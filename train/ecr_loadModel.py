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
import sys

# In[2]:


index = int(sys.argv[1])

h  = 0
hp = np.load('ecr_hyper_parameters.npy', allow_pickle=True)[()]


# In[ ]:


hp['batch_perc'] = 0.001
hp['num_epochs'] = 1000


# In[3]:


def broadcast_to_8x8(A):
    B = np.hstack((A[:,:2,:],np.tile(A[:,2,:].reshape(5,1,10),(1,2,1)),A[:,3:,:]))
    C = np.vstack((B[:2,:,:],np.tile(B[2,:,:].reshape(1,6,10),(2,1,1)),B[3:,:,:]))
    D = A[0,0,:].reshape((1,1,10)) #A1-A2 ear channel
    E = A[4,4,:].reshape((1,1,10)) #A1-A2 ear channel
    F = np.array(np.vstack((np.hstack((D,E)),np.hstack((E,D)))))
    G = np.tile(F,(4,4,1))
    G[1:7,1:7,:] = C
    return G[:,:,:8]


# In[4]:


def ecr_load_data(index):
    #l = range(10)
    trnIdx = [index] #[ii % len(l) for ii in range(fold,fold+6)]
    #valIdx = [ii % len(l) for ii in range(fold+6,fold+8)]
    trnY = []
    trnX = []
    #valY = []
    #valX = []
    for ii in trnIdx:
        foldData = np.load('../data/ecr_data2_'+str(ii)+'.npy', allow_pickle=True)[()]
        for subjData in foldData:
            for ttData in foldData[subjData]:
                [org,data] = foldData[subjData][ttData]
                for tt in list(data.keys())[:10]:
                    trnY.append(broadcast_to_8x8(org))
                    trnX.append(broadcast_to_8x8(np.nan_to_num(data[tt], copy=False)))
    return np.array(trnX),np.array(trnY)


def evaluate_model(X,Y_test,Y_pred):
    test_loss = []
    for ii in range(len(X)):
        x,y = np.argwhere(X[ii][:,:,0]==0)[0]
        test_loss.append(keras.losses.mean_squared_error(Y_test[ii][x,y], Y_pred[ii][x,y]))
    return [['NMSE loss'],[np.mean(np.array(test_loss))]]


# In[5]:


exp_top    = '../Checkpoints/' +  hp['experiment'][h] + '/topology/model.json'   # topology
exp_perf   = '../Checkpoints/' +  hp['experiment'][h] +'/all_performance'       # performance
exp_weight = '../Checkpoints/' +  hp['experiment'][h] +'/fold0/weights/nn_weights-500.hdf5'


# In[6]:


with open(exp_top, 'r') as json_file:
    architecture = json.load(json_file)
    nn = model_from_json(architecture)


# In[7]:


nn.load_weights(exp_weight, by_name=True)


# In[ ]:


tic = time.time()
print("(ecr_transfer): at ecr_load_data")
xTrn,yTrn = ecr_load_data(index)
toc = time.time()
print("(ecr_transfer): finished loading data after "+str(int(toc-tic))+" seconds")


# In[ ]:


fullPerf_over_time = []


# In[ ]:
grad_desc_algorithm = keras.optimizers.SGD(lr=hp['lr'][h], decay=0, momentum=hp['momentum'][h], nesterov=hp['nesterov'][h])
nn.compile(loss = 'mean_squared_error', optimizer = grad_desc_algorithm)

print("(ecr_transfer): evaluating")
#EVALUATE TRAINING SET
loss = [0]
loss[0] = nn.evaluate(xTrn, yTrn, verbose=0)
fullPerf_over_time.append(loss)
yPred = nn.predict(xTrn,verbose=0)
[test_names, results] = evaluate_model(xTrn,yTrn, yPred)
#results = loss.extend(loss,np.array(results))
fullPerf_over_time.append(results)


# In[ ]:


##fullPerf_over_time = np.vstack(fullPerf_over_time)
fullPerf_over_time = np.array(fullPerf_over_time)

# In[ ]:


# SAVE THE RESULTS ############################################################
np.save('./transfer_performance_'+str(index),fullPerf_over_time)

