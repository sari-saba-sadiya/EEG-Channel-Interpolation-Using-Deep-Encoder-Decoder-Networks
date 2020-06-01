#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import keras
import keras.optimizers
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, MaxPooling2D,Dropout,Conv2D,BatchNormalization,Reshape,UpSampling2D,ZeroPadding2D,Conv2DTranspose
from keras.backend import resize_images
from sklearn.utils import class_weight
import json
import time
import os


# In[ ]:


datasetName = 'ecr_data2_'
foldBegin = 0
foldEnd = 8


# In[38]:


index = 0
h  = 14
hp = np.load('ecr_hyper_parameters.npy', allow_pickle=True)[()]


# In[3]:


hp['batch_perc'] = 0.01
hp['num_epochs'] = 100


# In[4]:


def broadcast_to_8x8(A):
    B = np.hstack((A[:,:2,:],np.tile(A[:,2,:].reshape(5,1,10),(1,2,1)),A[:,3:,:]))
    C = np.vstack((B[:2,:,:],np.tile(B[2,:,:].reshape(1,6,10),(2,1,1)),B[3:,:,:]))
    D = A[0,0,:].reshape((1,1,10)) #A1-A2 ear channel
    E = A[4,4,:].reshape((1,1,10)) #A1-A2 ear channel
    F = np.array(np.vstack((np.hstack((D,E)),np.hstack((E,D)))))
    G = np.tile(F,(4,4,1))
    G[1:7,1:7,:] = C
    return G[:,:,:8]


# In[5]:


def ecr_load_data(index,foldBegin,foldEnd,datasetName):
    l = range(10)
    trnIdx = [ii % len(l) for ii in range(foldBegin,foldEnd+1)]
    trnY = []
    trnX = []
    valY = []
    valX = []
    for ii in trnIdx:
        foldData = np.load(datasetName+str(ii)+'.npy', allow_pickle=True)[()]
        for subjData in foldData:
            for ttData in foldData[subjData]:
                [org,data] = foldData[subjData][ttData]
                for jj,tt in enumerate(list(data.keys())[:10]):
                    if jj % 10 == 0:
                        trnY.append(broadcast_to_8x8(org))
                        trnX.append(broadcast_to_8x8(np.nan_to_num(data[tt], copy=False)))
                    else:
                        valY.append(broadcast_to_8x8(org))
                        valX.append(broadcast_to_8x8(np.nan_to_num(data[tt], copy=False)))
    return np.array(trnX),np.array(trnY),np.array(valX),np.array(valY)


# In[6]:


exp_top    = './Checkpoints/' +  hp['experiment'][h] + '/topology/model.json'   # topology
exp_perf   = './Checkpoints/' +  hp['experiment'][h] +'/all_performance'       # performance
exp_weight = './Checkpoints/' +  hp['experiment'][h] +'/fold0/weights/nn_weights-200.hdf5'


# In[7]:


with open(exp_top, 'r') as json_file:
    architecture = json.load(json_file)
    nn = model_from_json(architecture)


# In[8]:


nn.load_weights(exp_weight, by_name=True)


# In[37]:


tic = time.time()
print("(ecr_transfer): at ecr_load_data")
xTrn,yTrn,xVal,yVal = ecr_load_data(index,foldBegin,foldEnd,datasetName)
toc = time.time()
print("(ecr_transfer): finished loading data after "+str(int(toc-tic))+" seconds")


# In[25]
batch_size = int(round(hp['batch_perc']*np.size(xTrn,0)))
print("(ecr_CNN): batch_size is:"+str(batch_size)+" input size is "+str(np.size(xTrn,0))+" batch perc "+str(hp['batch_perc']))


grad_desc_algorithm = keras.optimizers.SGD(lr=hp['lr'][h], decay=0, momentum=hp['momentum'][h], nesterov=hp['nesterov'][h])


# In[33]:


nn.compile(loss = 'mean_squared_error', optimizer = grad_desc_algorithm)


# In[ ]:


tic = time.time()
print("(ecr_transfer): fitting")
fit_nn = nn.fit(xTrn,        # Training Data X
                yTrn,        # Training Data Y
                validation_data = (xVal,
                                   yVal),  # Validation data tuple
                shuffle         = 1,              # shuffle the training data epoch before you use it
                initial_epoch   = 0,              # Starting Epoch (should awlways be 0)
                epochs          = hp['num_epochs'],     # Number of runs through the data 
                batch_size      = batch_size)     # Number of samples per gradient update. 
toc = time.time()
print("(ecr_transfer): finished training after "+str(int(toc-tic))+" seconds")


# In[19]:


valPerf_over_time = []


# In[30]:


print("(ecr_transfer): evaluating")
#EVALUATE TRAINING SET
loss = nn.evaluate(xTrn, yTrn, verbose=0)
valPerf_over_time.append(loss)


# In[31]:


valPerf_over_time = np.vstack(valPerf_over_time)


# In[ ]:


# SAVE THE RESULTS ############################################################
np.save('transfer_performance_'+datasetName+str(foldBegin)+'_'+str(foldEnd),valPerf_over_time)


# In[ ]:




