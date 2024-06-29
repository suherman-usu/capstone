#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Callbacks are functions used to give a feedback about each epoch calculated metrics
from rl.callbacks import Callback

class ValidationCallback(Callback):

    def __init__(self):
        #Initially, the metrics are zero
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        self.short = 0
        self.long = 0
        self.shortAcc =0
        self.longAcc =0
        self.longPrec =0
        self.shortPrec =0
        self.marketRise =0
        self.marketFall =0

    def reset(self):
        #The metrics are also zero when the epoch ends
        self.episodes = 0
        self.rewardSum = 0
        self.accuracy = 0
        self.coverage = 0
        self.short = 0
        self.long = 0
        self.shortAcc =0
        self.longAcc =0
        self.longPrec =0
        self.shortPrec =0
        self.marketRise =0
        self.marketFall =0
        
    #all information is given by the environment: action, reward and market
    #Then, when the episode ends, metrics are calculated
    def on_episode_end(self, action, reward, market):
        
        #After the episode ends, increments the episodes 
        self.episodes+=1

        #Increments the reward
        self.rewardSum+=reward

        #If the action is not a hold, there is coverage because the agent decided 
        self.coverage+=1 if (action != 0) else 0

        #increments the accuracy if the reward is positive (we have a hit)
        self.accuracy+=1 if (reward >= 0 and action != 0) else 0
       
        
        #Increments the counter for short if the action is a short (id 2)
        self.short +=1 if(action == 2) else 0
        
        #Increments the counter for long if the action is a long (id 1)
        self.long +=1 if(action == 1) else 0
        
        #We will also calculate the accuracy for a given action. Here, it increments
        #the accuracy for short if the action is short and the reward is positive
        self.shortAcc +=1 if(action == 2 and reward >=0) else 0
        
        #Increments the accuracy for long if the action is long and the reward is positive
        self.longAcc +=1 if(action == 1 and reward >=0) else 0
        
        #If the market increases, increments the marketRise variable. If the prediction is 1 (long), increments the precision for long
        if(market>0):
            self.marketRise+=1
            self.longPrec+=1 if(action == 1) else 0

        #If market decreases, increments the marketFall. If the prediction is 2 (short), increments the precision for short   
        elif(market<0):
            self.marketFall+=1
            self.shortPrec+=1 if(action == 2) else 0

    #Function to show the metrics of the episode  
    def getInfo(self):
        #Start setting them to zero
        acc = 0
        cov = 0
        short = 0
        long = 0
        longAcc = 0
        shortAcc = 0
        longPrec = 0
        shortPrec = 0
        
        #If there is coverage, we will calculate the accuracy only related to when decisions were made. 
        #In other words, we dont calculate accuracy for hold operations
        if self.coverage > 0:
            acc = self.accuracy/self.coverage
        
        #Now, we calculate the mean coverage, short and long operations from the episodes
        if self.episodes > 0:
            cov = self.coverage/self.episodes
            short = self.short/self.episodes
            long = self.long/self.episodes

        #Calculate the mean accuracy for short operations. 
        #That is, the number of total short correctly predicted (self.shortAcc) 
        #divided by the total of shorts predicted (self.short)
        # #We need to correct this     
        if self.short > 0:
            shortAcc = self.shortAcc/self.short
        
        #Calculate the mean accuracy for long operations. 
        #That is, the number of total short correctly predicted (long.shortAcc) 
        #divided by the total of longs predicted (long.short)
        if self.long > 0:
            longAcc = self.longAcc/self.long


        if self.marketRise > 0:
            longPrec = self.longPrec/self.marketRise

        if self.marketFall > 0:
            shortPrec = self.shortPrec/self.marketFall

        #Returns the metrics to the user    
        return self.episodes,cov,acc,self.rewardSum,long,short,longAcc,shortAcc,longPrec,shortPrec


# In[11]:


#gym is the library of videogames used by reinforcement learning
import gym
from gym import spaces
import numpy
import pandas
from datetime import datetime
import Callback


class SpEnv1(gym.Env):
    #Just for the gym library. In a continuous environment, you can do infinite decisions. 
    #We dont want this because we have just three possible actions.
    continuous = False

    #Observation window is the time window regarding the "hourly" dataset 
    #ensemble variable tells to save or not the decisions at each walk

    def __init__(self, data, callback = None, ensamble = None, columnName = "iteration-1"):
        #Declare the episode as the first episode
        self.episode=1

        # opening the dataset      
        self.data=data

        #Load the data
        self.output=False

        #ensamble is the table of validation and testing
        #If its none, you will not save csvs of validation and testing    
        if(ensamble is not None): # managing the ensamble output (maybe in the wrong way)
            self.output=True
            self.ensamble=ensamble
            self.columnName = columnName
            
            #self.ensemble is a big table (before file writing) containing observations as lines and epochs as columns
            #each column will contain a decision for each epoch at each date. It is saved later.
            #We read this table later in order to make ensemble decisions at each epoch
            self.ensamble[self.columnName]=0

        #Declare low and high as vectors with -inf values 
        self.low = numpy.array([-numpy.inf])
        self.high = numpy.array([+numpy.inf])

        #Define the space of actions as 3
        #the action space is now 2 (hold and long)
        #self.action_spaces = space.Discrete(2) 
        self.action_space = gym.spaces.Box(low=numpy.array([0]),high= numpy.array([2]), dtype=int)
      
               
        #low and high are the minimun and maximum accepted values for this problem
        #Tonio used random values
        #We dont know what are the minimum and maximum values of Close-Open, so we put these values
        self.observation_space = spaces.Box(self.low, self.high, dtype=numpy.float32)

        self.currentObservation = 0
        #Defines that the environment is not done yet
        self.done = False
        #The limit is the number of open values in the dataset (could be any other value)
        self.limit = len(data)      
        
        #Initiates the values to be returned by the environment
        self.reward = None
        self.possibleGain = 0
        self.openValue = 0
        self.closeValue = 0
        self.callback=callback

    #This is the action that is done in the environment. 
    #Receives the action and returns the state, the reward and if its done 
    def step(self, action):
    
        #assert self.action_space.contains(action)

        #Initiates the reward, weeklist and daylist
        self.reward=0
    
        #Calculate the reward in percentage of growing/decreasing
        self.possibleGain = self.data.iloc[self.currentObservation]['delta_next_day']
        
        #Calculate the reward in percentage of growing/decreasing
        self.possibleGain = self.data.iloc[self.currentObservation]['delta_next_day']

        #If action is a LONG, calculate the reward
        #If action is a long, calculate the reward
        if(action == 1):
        #The reward must be subtracted by the cost of transaction
        #action=1
            self.reward = self.possibleGain

        #If action is a short, calculate the reward
        elif(action==2):
            self.reward = (-self.possibleGain)

        #If action is a hold, no reward
        elif(action==0):
            self.reward = 0
                   
        #Finish episode 
        self.done=True

        
        #Call the callback for the episode
        if(self.callback!=None and self.done):
            self.callback.on_episode_end(action,self.reward,self.possibleGain)
            
        #If its validation or test, save the outputs in the ensemble file that will be ensembled later    
        if(self.output):
            self.ensamble.at[self.data.iloc[self.currentObservation]['date_time'],self.columnName]=action
                   
        self.episode+=1   
        self.currentObservation+=1
        
        if(self.currentObservation>=self.limit):
            self.currentObservation=0
             
        #Return the state, reward and if its done or not
        return self.getObservation(), self.reward, self.done, {}
        
    #function done when the episode finishes
    #reset will prepare the next state (feature vector) and give it to the agent
    def reset(self):
    
        self.done = False
        self.reward = None
        self.possibleGain = 0
       
        return self.getObservation()
        

    def getObservation(self):

        predictionList = []
        predictionList=numpy.array(self.data.iloc[self.currentObservation]["prediction_0":"prediction_999"])
     
        
        return predictionList.ravel()
    
    def resetEnv(self):
        self.currentObservation=0
        self.episode=1


# In[12]:


#Imports the SPEnv library, which will perform the Agent actions themselves
from Callback import ValidationCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import LeakyReLU, PReLU
from keras.optimizers import *
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from keras_radam import RAdam
from math import floor
import pandas as pd
import datetime
import os
import numpy
numpy.random.seed(0)

class DeepQTrading1:
    
    #Class constructor
    #model: Keras model considered
    #explorations_iterations: a vector containing (i) probability of random predictions; (ii) how many iterations will be 
    #run by the algorithm (we run the algorithm several times-several iterations)  
    #outputFile: name of the file to print metrics of the training
    #ensembleFolderName: name of the file to print predictions
    #optimizer: optimizer to run 
        
    def __init__(self, model, nbActions, explorations_iterations, outputFile, ensembleFolderName, optimizer="adamax"):
        
        self.ensembleFolderName=ensembleFolderName
        self.policy = EpsGreedyQPolicy()
        self.explorations_iterations=explorations_iterations
        self.nbActions=nbActions
        self.model=model
        #Define the memory
        self.memory = SequentialMemory(limit=10000, window_length=1)
        #Instantiate the agent with parameters received
        self.agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=200, target_model_update=1e-1, enable_double_dqn=True,enable_dueling_network=True)
        
        #Compile the agent with the optimizer given as parameter
        if optimizer=="adamax":        
                self.agent.compile(Adamax(), metrics=['mae'])
        if optimizer=="adadelta":        
                self.agent.compile(Adadelta(), metrics=['mae'])
        if optimizer=="sgd":        
                self.agent.compile(SGD(), metrics=['mae'])
        if optimizer=="rmsprop":        
                self.agent.compile(RMSprop(), metrics=['mae'])
        if optimizer=="nadam":        
                self.agent.compile(Nadam(), metrics=['mae'])
        if optimizer=="adagrad":        
                self.agent.compile(Adagrad(), metrics=['mae'])
        if optimizer=="adam":        
                self.agent.compile(Adam(), metrics=['mae'])
        if optimizer=="radam":        
                self.agent.compile(RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5), metrics=['mae'])

        #Save the weights of the agents in the q.weights file
        #Save random weights
        self.agent.save_weights("q.weights", overwrite=True)

        #Load data
        self.train_data= pd.read_csv('./dataset/jpm/train_data2.csv')
        self.validation_data=pd.read_csv('./dataset/jpm/train_data2.csv')
        self.test_data=pd.read_csv('./dataset/jpm/test_data2.csv')
                
        #Call the callback for training, validation and test in order to show results for each iteration 
        self.trainer=ValidationCallback()
        self.validator=ValidationCallback()
        self.tester=ValidationCallback()
        self.outputFileName=outputFile

    def run(self):
        #Initiates the environments, 
        trainEnv=validEnv=testEnv=" "
         
        if not os.path.exists(self.outputFileName):
             os.makedirs(self.outputFileName)

        file_name=self.outputFileName+"/results-agent-training.csv"
        
        self.outputFile=open(file_name, "w+")
        #write the first row of the csv
        self.outputFile.write(
            "Iteration,"+
            "trainAccuracy,"+
            "trainCoverage,"+
            "trainReward,"+
            "trainLong%,"+
            "trainShort%,"+
            "trainLongAcc,"+
            "trainShortAcc,"+
            "trainLongPrec,"+
            "trainShortPrec,"+

            "validationAccuracy,"+
            "validationCoverage,"+
            "validationReward,"+
            "validationLong%,"+
            "validationShort%,"+
            "validationLongAcc,"+
            "validationShortAcc,"+
            "validLongPrec,"+
            "validShortPrec,"+
                
            "testAccuracy,"+
            "testCoverage,"+
            "testReward,"+
            "testLong%,"+
            "testShort%,"+
            "testLongAcc,"+
            "testShortAcc,"+
            "testLongPrec,"+
            "testShortPrec\n")      
        
            
        #Prepare the training and validation files for saving them later 
        ensambleValid=pd.DataFrame(index=self.validation_data[:].loc[:,'date_time'].drop_duplicates().tolist())
        ensambleTest=pd.DataFrame(index=self.test_data[:].loc[:,'date_time'].drop_duplicates().tolist())
            
        #Put the name of the index for validation and testing
        ensambleValid.index.name='date_time'
        ensambleTest.index.name='date_time'
            
        #Explorations are epochs considered, or how many times the agent will play the game.  
        for eps in self.explorations_iterations:

            #policy will use eps[0] (explorations), so the randomness of predictions (actions) will happen with eps[0] of probability 
            self.policy.eps = eps[0]
                
            #there will be 25 iterations or eps[1] in explorations_iterations)
            for i in range(0,eps[1]):
                    
                del(trainEnv)
                #Define the training, validation and testing environments with their respective callbacks
                trainEnv = SpEnv1(data=self.train_data, callback=self.trainer)
                
                del(validEnv)
                validEnv=SpEnv1(data=self.validation_data,ensamble=ensambleValid,callback=self.validator,columnName="iteration"+str(i))
                
                del(testEnv)  
                testEnv=SpEnv1(data=self.test_data, callback=self.tester,ensamble=ensambleTest,columnName="iteration"+str(i))

                #Reset the callback
                self.trainer.reset()
                self.validator.reset()
                self.tester.reset()

                #Reset the training environment
                trainEnv.resetEnv()
                
                #Train the agent
                #The agent receives as input one environment
                self.agent.fit(trainEnv,nb_steps=len(self.train_data),visualize=False,verbose=0)
                
                #Get the info from the train callback    
                (_,trainCoverage,trainAccuracy,trainReward,trainLongPerc,                                              trainShortPerc,trainLongAcc,trainShortAcc,trainLongPrec,trainShortPrec)=self.trainer.getInfo()
                
                print("Iteration " + str(i+1) + " TRAIN:  accuracy: " + str(trainAccuracy)+ " coverage: " + str(trainCoverage)+ " reward: " + str(trainReward))
                             
                #Reset the validation environment
                validEnv.resetEnv()               
                #Test the agent on validation data
                self.agent.test(validEnv,nb_episodes=len(self.validation_data),visualize=False,verbose=0)
                
                #Get the info from the validation callback
                (_,validCoverage,validAccuracy,validReward,validLongPerc,validShortPerc,
validLongAcc,validShortAcc,validLongPrec,validShortPrec)=self.validator.getInfo()
                #Print callback values on the screen
                print("Iteration " +str(i+1) + " VALIDATION:  accuracy: " + str(validAccuracy)+ " coverage: " + str(validCoverage)+ " reward: " + str(validReward))

                #Reset the testing environment
                testEnv.resetEnv()
                #Test the agent on testing data
                self.agent.test(testEnv,nb_episodes=len(self.test_data),visualize=False,verbose=0)
                #Get the info from the testing callback
                (_,testCoverage,testAccuracy,testReward,testLongPerc,testShortPerc,
testLongAcc,testShortAcc,testLongPrec,testShortPrec)=self.tester.getInfo()
                #Print callback values on the screen
                print("Iteration " +str(i+1) + " TEST:  acc: " + str(testAccuracy)+ " cov: " + str(testCoverage)+ " rew: " + str(testReward))
                print(" ")
                    
                #write the metrics in a text file
                self.outputFile.write(
                    str(i)+","+
                    str(trainAccuracy)+","+
                    str(trainCoverage)+","+
                    str(trainReward)+","+
                    str(trainLongPerc)+","+
                    str(trainShortPerc)+","+
                    str(trainLongAcc)+","+
                    str(trainShortAcc)+","+
                    str(trainLongPrec)+","+
                    str(trainShortPrec)+","+
                       
                    str(validAccuracy)+","+
                    str(validCoverage)+","+
                    str(validReward)+","+
                    str(validLongPerc)+","+
                    str(validShortPerc)+","+
                    str(validLongAcc)+","+
                    str(validShortAcc)+","+
                    str(validLongPrec)+","+
                    str(validShortPrec)+","+
                       
                    str(testAccuracy)+","+
                    str(testCoverage)+","+
                    str(testReward)+","+
                    str(testLongPerc)+","+
                    str(testShortPerc)+","+
                    str(testLongAcc)+","+
                    str(testShortAcc)+","+
                    str(testLongPrec)+","+
                    str(testShortPrec)+"\n")

        #Close the file                
        self.outputFile.close()

        if not os.path.exists("./Output/ensemble/"+self.ensembleFolderName):
             os.makedirs("./Output/ensemble/"+self.ensembleFolderName)

        ensambleValid.to_csv("./Output/ensemble/"+self.ensembleFolderName+"/ensemble_valid.csv")
        ensambleTest.to_csv("./Output/ensemble/"+self.ensembleFolderName+"/ensemble_test.csv")


    #Function to end the Agent
    def end(self):
        print("FINISHED")


# In[13]:


"""""
        This is the code of Reinforcement learning applied on outputs of an ensemble of classifiers
        There is an ensemble of 1000 CNNs that will output predictions for each day
        Therefore, our RL metalearner will be applied on these 1000 outputs
        
        We call it as the following:
        
        python main.py <number_of_actions> <number_of_explorations> <activation> <output_file> <optimizer>
        
        ex: python3 main.py 3 0.3 selu teste-rmsprop-0.3-selu rmsprop

        where:
                <number_of_actions>: number of actions done by the agent. 
                <number_of_explorations>: in the RL training, this is the probability that the action taken is random or it obeys the Q-values found previously 
                <activation>: activation function of the double q-network layer we use as RL agent 
                <output_file>: where results will be written 
                <optimizer>: optimization approach of the RL network

       Authors: Anselmo Ferreira, Alessandro Sebastian Podda and Andrea Corriga
       
       Please dont hesitate to cite our Applied Intelligence paper when using it for your research ;-)  
  
"""""

#os library is used to define the GPU to be used by the code, needed only in cerain situations (Better not to use it, use only if the main gpu is Busy)
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import LeakyReLU, PReLU, ReLU
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
import sys
import tensorflow as tf
from tensorflow.python.keras import backend as K
config =tf.compat.v1.ConfigProto
tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)


#There are three actions possible in the stock market
#Hold(id 0): do nothing.
#Long(id 1): It predicts that the stock market value will raise at the end of the day. 
#Short(id 2): It predicts that the stock market value will decrease at the end of the day.

#NN composes one flatten layer to get 1000 dimensional vectors as input
#One dense layer with 35 neurons with a given activation
#One final Dense Layer with the number of actions considered and linear activation

#sys.argv[1]: number of actions ---in1
#sys.argv[2]: probability of performing explorations ---in2
#sys.argv[3]: initializer ---in3
#sys.argv[4]: folder name where experiments results will be written ---in4
#sys.argv[5]: optimizer ---in5  3 0.3 relu teste-adam-0.3-relu  adam  
in1='3'
in2='0.3'
in3='relu'
in4='teste-adam-0.3-relu'
in5='adam'

model = Sequential()
model.add(Flatten(input_shape=(1,1000))) 
if(in3=="relu"):
    model.add(Dense(35,activation='relu'))    
if(in3=="sigmoid"):
    model.add(Dense(35,activation='sigmoid'))    
if(in3=="linear"):
    model.add(Dense(35,activation='linear'))
if(in3=="tanh"):
    model.add(Dense(35,activation='tanh'))
if(in3=="selu"):
    model.add(Dense(35,activation='selu'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(int(in1)))
model.add(Activation('linear'))



#Define the DeepQTrading class with the following parameters:
#explorations: operations are random with a given probability, and 25 iterations.
#in this case, iterations parameter is used because the agent acts on daily basis, so its better to repeat the experiments several
#times. 
#outputFile: where the results will be written

dqt = DeepQTrading1(
    model=model,
    nbActions=int(in1),
    explorations_iterations=[(round(float(in2)),25)],
    outputFile="./Output/csv/" + in4,
    ensembleFolderName=in4,
    optimizer=in5
    )

dqt.run()

dqt.end()



# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def majority_voting(df):
    
    local_df = df.copy()
    x=local_df.loc[:,'iteration0':'iteration24']
    local_df['ensemble']=x.mode(axis=1).iloc[:, 0]
    local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)
    return local_df
    
def ensemble(type, ensembleFolderName):

    dollSum=0
    rewSum=0
    posSum=0
    negSum=0
    covSum=0 
    numSum=0

    values=[]
    columns = ["Experiment","#Wins","#Losses","Dollars","Coverage","Accuracy"]
    
    sp500=pd.read_csv("./dataset/jpm/test_data.csv",index_col='date_time')
      
    df=pd.read_csv("./Output/ensemble/"+ensembleFolderName+"/ensemble_"+type+".csv",index_col='date_time')
        
    df=majority_voting(df)
      
    num=0
    rew=0
    pos=0
    neg=0
    doll=0
    cov=0

  
    #Lets iterate through each date and decision 
    for date, i in df.iterrows():
     
        #If the date in the predictions is in the index of sp500 (which is also a date) 
        if date in sp500.index:
             
            num+=1
              
            #If the output is 1 (long)
            if (i['ensemble']==1):
                   
                #If the close - open is positive at that day, we have earning money. Positives are equal to 1. Otherwise, no incrementation 
                pos+= 1 if (float(sp500.at[date,'delta_next_day'])) > 0 else 0

                #If close - open is negative at that day, we are losing money. Negatives are equal to 1. Otherwise, no incrementation 
                neg+= 1 if (float(sp500.at[date,'delta_next_day'])) < 0 else 0

                #Lets calculate the reward (positive or negative)
                rew+=float(sp500.at[date,'delta_next_day'])
                    
                #In dollars, we just multiply by the sp500 points by the differences 
                doll+=float(sp500.at[date,'delta_next_day'])

                #There is coverage (of course) 
                cov+=1

            #The same stuff happens for short.
            elif (i['ensemble']==2):
     
                pos+= 1 if float(sp500.at[date,'delta_next_day']) < 0 else 0
                neg+= 1 if float(sp500.at[date,'delta_next_day']) > 0 else 0
                    
                rew+=-float(sp500.at[date,'delta_next_day'])
                cov+=1
                doll+=-float(sp500.at[date,'delta_next_day'])
                    


    values.append([str(1),str(round(pos,2)),str(round(neg,2)),str(round(doll,2)),str(round(cov/num,2)),(str(round(pos/cov,2)) if (cov>0) else "")])
        
    #Now lets sum walk by walk 
    dollSum+=doll
    rewSum+=rew
    posSum+=pos
    negSum+=neg
    covSum+=cov
    numSum+=num

    
    #Now lets summarize everything showing the sum of values 
    values.append(["sum",str(round(posSum,2)),str(round(negSum,2)),str(round(dollSum,2)),str(round(covSum/numSum,2)),(str(round(posSum/covSum,2)) if (covSum>0) else "")])
    
    return values,columns
    
    
################


# In[15]:


from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from math import floor
#from ensemble import ensemble
from matplotlib.gridspec import GridSpec

#sys.argv[1] --input1
#sys.argv[2] --input2
#sys.argv[3] --input3
#sys.argv[4] --input4
input1='teste-adam-0.3-relu'
input2='results-adam-relu-explorations-0.3'
input3='1'
input4='0'

outputFile=str(input2)+".pdf"
numFiles=int(input3)
#Number of epochs in the algorithm
numEpochs=35
numPlots=10

pdf=PdfPages(outputFile)

#Configure the size of the picture that will be plotted
#Configure the size of the picture that will be plotted
plt.figure(figsize=((numEpochs/10)*(2),9*5))

#Open the file that was saved on folder csv/walks, containing information about each iteration in that walk 
#Lets show a summary of each walk
#For each walk, one column is plotted in a final pdf file
for i in range(1,numFiles+1):

    document = pd.read_csv("./Output/csv/"+ input1+"/results-agent-training.csv")
    plt.subplot(numPlots,numFiles,0*numFiles + i)
    #Draw information in that file. First of all, lets plot accuracy
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'testAccuracy'].tolist(),'r',label='Test')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'trainAccuracy'].tolist(),'b',label='Train')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'validationAccuracy'].tolist(),'g',label='Validation')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Accuracy')

    #Lets draw information about coverage, read from the csv file located at csv/walks
    plt.subplot(numPlots,numFiles,1*numFiles + i)
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'testCoverage'].tolist(),'r',label='Test')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'trainCoverage'].tolist(),'b',label='Train')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'validationCoverage'].tolist(),'g',label='Validation')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Coverage')

    # Information about reward
    plt.subplot(numPlots,numFiles,2*numFiles + i )
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'trainReward'].tolist(),'b',label='Train')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'validationReward'].tolist(),'g',label='Validation')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'testReward'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Reward')
    
    #Percentages of long
    plt.subplot(numPlots,numFiles,3*numFiles + i )
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'trainLong%'].tolist(),'b',label='Train')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'validationLong%'].tolist(),'g',label='Validation')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'testLong%'].tolist(),'r',label='Test')  
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))    
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long %')
    
    #Percentages of short
    plt.subplot(numPlots,numFiles,4*numFiles + i )
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'trainShort%'].tolist(),'b',label='Train')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'validationShort%'].tolist(),'g',label='Validation')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'testShort%'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short %')
    

    #Coverage
    plt.subplot(numPlots,numFiles,5*numFiles + i )
    plt.plot(document.loc[:, 'Iteration'].tolist(),list(map(lambda x: 1-x,document.loc[:, 'trainCoverage'].tolist())),'b',label='Train')
    plt.plot(document.loc[:, 'Iteration'].tolist(),list(map(lambda x: 1-x,document.loc[:, 'validationCoverage'].tolist())),'g',label='Validation')
    plt.plot(document.loc[:, 'Iteration'].tolist(),list(map(lambda x: 1-x,document.loc[:, 'testCoverage'].tolist())),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Hold %')
    

    #Accuracy of longs
    plt.subplot(numPlots,numFiles,6*numFiles + i )
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'trainLongAcc'].tolist(),'b',label='Train')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'validationLongAcc'].tolist(),'g',label='Validation')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'testLongAcc'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long Accuracy')
    
    #Accuracy of shorts
    plt.subplot(numPlots,numFiles,7*numFiles + i )
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'trainShortAcc'].tolist(),'b',label='Train')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'validationShortAcc'].tolist(),'g',label='Validation')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'testShortAcc'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short Accuracy')

    
    #Precisions of long
    plt.subplot(numPlots,numFiles,8*numFiles + i )
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'trainLongPrec'].tolist(),'b',label='Train')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'validLongPrec'].tolist(),'g',label='Validation')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'testLongPrec'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long Precision')
    
    #Precisions of short
    plt.subplot(numPlots,numFiles,9*numFiles + i )
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'trainShortPrec'].tolist(),'b',label='Train')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'validShortPrec'].tolist(),'g',label='Validation')
    plt.plot(document.loc[:, 'Iteration'].tolist(),document.loc[:, 'testShortPrec'].tolist(),'r',label='Test')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short Precision')

plt.suptitle("Experiment RL metalearner\n"
            +"Model: 35 neurons single layer\n"
            +"Input: 1000 predictions of CNNs\n"
            +"Memory-Window Length: 10000-1\n"
            +"Other changes: Does Short, Hold and Long\n"
            +"Explorations:" +input4 +"."
            ,size=19    
            ,weight=20
            ,ha='left'
            ,x=0.1
            ,y=0.99)

pdf.savefig()


#Now, lets try the ensemble
i=1

###########-------------------------------------------------------------------|Tabella Full Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*3.5,y*3.5))

plt.subplot(y,y,1)
plt.axis('off')
val,col=ensemble("test", input1)
t=plt.table(cellText=val, colLabels=col, fontsize=20, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Final Results")
#plt.suptitle("MAJORITY VOTING")
pdf.savefig()
###########--------------------------------------------------------------------------------------------------------------------
pdf.close()


# In[ ]:




