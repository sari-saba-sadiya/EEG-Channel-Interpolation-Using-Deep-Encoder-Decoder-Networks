#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:53:57 2020

@author: sarisadiya
"""


import numpy as np
import keras
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D,Dropout,Conv2D,BatchNormalization,Reshape,UpSampling2D,ZeroPadding2D,Conv2DTranspose
from sklearn.utils import class_weight
import json
import time
import os
import sys
import keras.backend as K
#Let the user know if they are not using a GPU
if K.tensorflow_backend._get_available_gpus() == []:
    print('YOU ARE NOT USING A GPU ON THIS DEVICE!')
else: 
    print('USING GPU!!!!!!')

h  = int(sys.argv[1])
hp = np.load('ecr_hyper_parameters.npy', allow_pickle=True)[()]

hp['batch_perc'] = 0.001
hp['num_epochs'] = 200


def broadcast_to_8x8(A):
    B = np.hstack((A[:,:2,:],np.tile(A[:,2,:].reshape(5,1,10),(1,2,1)),A[:,3:,:]))
    C = np.vstack((B[:2,:,:],np.tile(B[2,:,:].reshape(1,6,10),(2,1,1)),B[3:,:,:]))
    D = A[0,0,:].reshape((1,1,10)) #A1-A2 ear channel
    E = A[4,4,:].reshape((1,1,10)) #A1-A2 ear channel
    F = np.array(np.vstack((np.hstack((D,E)),np.hstack((E,D)))))
    G = np.tile(F,(4,4,1))
    G[1:7,1:7,:] = C
    return G[:,:,:8]

def ecr_load_data(fold):
    l = range(10)
    trnIdx = [ii % len(l) for ii in range(fold,fold+6)]
    valIdx = [ii % len(l) for ii in range(fold+6,fold+8)]
    trnY = []
    trnX = []
    valY = []
    valX = []
    for ii in trnIdx:
        foldData = np.load('ecr_data_'+str(ii)+'.npy', allow_pickle=True)[()]
        for subjData in foldData:
            for ttData in foldData[subjData]:
                [org,data] = foldData[subjData][ttData]
                for tt in list(data.keys())[:10]:
                    trnY.append(broadcast_to_8x8(org))
                    trnX.append(broadcast_to_8x8(np.nan_to_num(data[tt], copy=False)))
    for ii in valIdx:
        foldData = np.load('ecr_data_'+str(ii)+'.npy', allow_pickle=True)[()]
        for subjData in foldData:
            for ttData in foldData[subjData]:
                [org,data] = foldData[subjData][ttData]
                for tt in list(data.keys())[:10]:
                    valY.append(broadcast_to_8x8(org))
                    valX.append(broadcast_to_8x8(np.nan_to_num(data[tt], copy=False)))
    return np.array(trnX),np.array(trnY),np.array(valX),np.array(valY)


def callbacks(hp,h,fold):  
    
    ###########################################################################
    # Checkpoints 
    ###########################################################################
    filepath="Checkpoints/" + hp['experiment'][h] + "/fold" + str(fold) + "/weights/nn_weights-{epoch:02d}.hdf5" # Where are checkpoints saved
    checkpoint = keras.callbacks.ModelCheckpoint(
                 filepath, 
                 monitor='val_loss',                     # Validation set Loss           
                 verbose           = 0,                  # Display text 
                 save_weights_only = True,               # if True, only the model weights are saved
                 save_best_only    = False,              # if True, the latest-best model is overwritten
                 mode              = 'auto',             # used if 'save_best_only' is True  
                 period            = 5)                  # Epochs between checkpoints
    return checkpoint

def CNN_2D(hp,h):
    nn = Sequential()
    for i in range(0,hp['cnn2d_num_cnn_layers'][h]):
        #--------------------------------------------------------------------------
        # 3D Convolution
        if i == 0:
            if hp['cnn2d_strides_per_layer'][h][i][0] == 1:
                nn.add(Conv2D(hp['cnn2d_filters_per_layer'][h][i],                        # 8
                                           kernel_size        = hp['cnn2d_kernal_sizes_per_layer'][h][i],     # (3, 3, 3),
                                           strides            = hp['cnn2d_strides_per_layer'][h][i],          # (1, 1, 1),
                                           kernel_initializer = hp['weights_initialization'][h],
                                           bias_initializer   = 'zeros',
                                           input_shape        = hp['input_shape'], 
                                           data_format        ='channels_last',
                                           padding            ='same'))
            else:
                s = hp['cnn2d_strides_per_layer'][h][i][0]
                k = hp['cnn2d_kernal_sizes_per_layer'][h][i][0]
                d = hp['input_shape'][0]
                p = int(np.floor(((s-1)*d+k-s)/2))
                nn.add(ZeroPadding2D(padding=(p,p), input_shape = hp['input_shape']))
                nn.add(Conv2D(hp['cnn2d_filters_per_layer'][h][i],                        # 8
                                           kernel_size        = hp['cnn2d_kernal_sizes_per_layer'][h][i],     # (3, 3, 3),
                                           strides            = hp['cnn2d_strides_per_layer'][h][i],          # (1, 1, 1),
                                           kernel_initializer = hp['weights_initialization'][h],
                                           bias_initializer   = 'zeros',
                                           input_shape        = hp['input_shape'],
                                           data_format        ='channels_last',
                                           padding            ='same'))
        else:
            s = hp['cnn2d_strides_per_layer'][h][i][0]
            k = hp['cnn2d_kernal_sizes_per_layer'][h][i][0]
            d = nn.layers[-1].output_shape[1]
            p = int(np.floor(((s-1)*d+k-s)/2))
            if hp['cnn2d_strides_per_layer'][h][i][0] != 1:
                nn.add(ZeroPadding2D(padding=(p,p)))
            nn.add(Conv2D(hp['cnn2d_filters_per_layer'][h][i],                        # 8
                                       kernel_size        = hp['cnn2d_kernal_sizes_per_layer'][h][i],     # (3, 3, 3),
                                       strides            = hp['cnn2d_strides_per_layer'][h][i],          # (1, 1, 1),
                                       kernel_initializer = hp['weights_initialization'][h],
                                       bias_initializer   = 'zeros',
                                       padding            = 'same'))
        #Batch Noramlization
        if hp['cnn2d_batchnormalize_per_layer'][h][i]:
            nn.add(BatchNormalization( axis            = -1, 
                                momentum               = 0.99, 
                                epsilon                = 0.001, 
                                center                 = True, 
                                scale                  = True, 
                                beta_initializer       = 'zeros', 
                                gamma_initializer      = 'ones', 
                                moving_mean_initializer = 'zeros', 
                                moving_variance_initializer = 'ones', 
                                beta_regularizer       = None, 
                                gamma_regularizer      = None, 
                                beta_constraint        = None, 
                                gamma_constraint       = None))
        if hp['cnn2d_activations_per_layer'][h][i]:
            nn.add(Activation('relu'))
        if hp['cnn2d_maxpool_per_layer'][h][i]:
            nn.add(MaxPooling2D(pool_size   = (2,2),
                                padding     = 'same'))
        if hp['cnn2d_dropouts_per_layer'][h][i] != []:
            nn.add(Dropout(hp['cnn2d_dropouts_per_layer'][h][i]))
            
    while nn.layers[-1].output_shape[3] > 8:
        if 2*nn.layers[-1].output_shape[1] <= 8:
            nn.add(Conv2DTranspose(int(np.floor(nn.layers[-1].output_shape[3]/2)),
                                       kernel_size        = (2,2),
                                       strides            = (2,2),
                                       kernel_initializer = hp['weights_initialization'][h],
                                       bias_initializer   = 'zeros',
                                       data_format        ='channels_last',
                                       padding            ='valid'))
        else:
            k = 2
            s = 1
            d = nn.layers[-1].output_shape[1]
            p = int(np.floor(((s-1)*d+k-s)/2))
            nn.add(ZeroPadding2D(padding=(p,p)))
            nn.add(Conv2DTranspose(int(np.floor(nn.layers[-1].output_shape[3]/2)),
                                       kernel_size        = (2,2),
                                       strides            = (1,1),
                                       kernel_initializer = hp['weights_initialization'][h],
                                       bias_initializer   = 'zeros',
                                       data_format        ='channels_last',
                                       padding            ='same'))
        #Batch Noramlization
        if hp['cnn2d_batchnormalize_per_layer'][h][i]:
            nn.add(BatchNormalization( axis            = -1, 
                                momentum               = 0.99, 
                                epsilon                = 0.001, 
                                center                 = True, 
                                scale                  = True, 
                                beta_initializer       = 'zeros', 
                                gamma_initializer      = 'ones', 
                                moving_mean_initializer = 'zeros', 
                                moving_variance_initializer = 'ones', 
                                beta_regularizer       = None, 
                                gamma_regularizer      = None, 
                                beta_constraint        = None, 
                                gamma_constraint       = None))

        #Activation
        if hp['cnn2d_activations_per_layer'][h][i]:
            nn.add(Activation('relu'))

            
    while 2*nn.layers[-1].output_shape[1] <= 8:
        nn.add(Conv2DTranspose(int(np.floor(nn.layers[-1].output_shape[3])),
                                   kernel_size        = (2,2),
                                   strides            = (2,2),
                                   kernel_initializer = hp['weights_initialization'][h],
                                   bias_initializer   = 'zeros',
                                   data_format        ='channels_last',
                                   padding            ='same'))            
            
        
        #Batch Noramlization
        if hp['cnn2d_batchnormalize_per_layer'][h][i]:
            nn.add(BatchNormalization( axis            = -1, 
                                momentum               = 0.99, 
                                epsilon                = 0.001, 
                                center                 = True, 
                                scale                  = True, 
                                beta_initializer       = 'zeros', 
                                gamma_initializer      = 'ones', 
                                moving_mean_initializer = 'zeros', 
                                moving_variance_initializer = 'ones', 
                                beta_regularizer       = None, 
                                gamma_regularizer      = None, 
                                beta_constraint        = None, 
                                gamma_constraint       = None))


        #Activation
        if hp['cnn2d_activations_per_layer'][h][i]:
            nn.add(Activation('relu'))
    nn.add(Dense(nn.layers[-1].output_shape[1], activation='relu'))
    return nn

def evaluate_model(Y_test,Y_pred):
    test_loss = keras.losses.mean_squared_error(Y_test, Y_pred)
    return [['NMSE loss'],[np.mean(np.array(test_loss))]]


os.mkdir('Checkpoints/' + hp['experiment'][h])                       # make the directory '<time_val>'
os.mkdir('Checkpoints/' +  hp['experiment'][h] + '/topology')         # make the directory 'topology'
os.mkdir('Checkpoints/' +  hp['experiment'][h] +'/all_performance')          # make the directory 'overall'


for fold in range(1):
    os.mkdir('Checkpoints/' +  hp['experiment'][h] + '/fold' + str(fold))                      # make the director for this fold
    os.mkdir('Checkpoints/' +  hp['experiment'][h] + '/fold' + str(fold) + '/weights')         # make the directory for weights
    os.mkdir('Checkpoints/' +  hp['experiment'][h] + '/fold' + str(fold) + '/performance')     # make the directory for performance

    tic = time.time()
    print("(ecr_CNN): at ecr_load_data")
    xTrn,yTrn,xVal,yVal = ecr_load_data(fold)
    toc = time.time()
    print("(ecr_CNN): finished loading data after "+str(int(toc-tic))+" seconds")

    hp['input_shape'] = (8,8,8)
    nn = CNN_2D(hp,h)
    nn.summary() 
   
    if nn.layers[-1].output_shape[1:] != (8,8,8):
        print("(ecr_CNN): size Error!!!!!")
        raise Exception('size Error!')
    
    print("(ecr_CNN): Save topology")
    json_nn = nn.to_json()
    with open('Checkpoints/' + hp['experiment'][h]  + '/topology/model.json', 'w') as outfile:
        json.dump(json_nn, outfile)
        
    print("(ecr_CNN): creating an optimizer")
    grad_desc_algorithm = keras.optimizers.SGD(lr=hp['lr'][h], decay=0, momentum=hp['momentum'][h], nesterov=hp['nesterov'][h])

    print("(ecr_CNN): compiling")
    nn.compile(loss = 'mean_squared_error', optimizer = grad_desc_algorithm, metrics = [keras.losses.mean_squared_error])

    batch_size = int(round(hp['batch_perc']*np.size(xTrn,0)))
    print("(ecr_CNN): batch_size is:"+str(batch_size)+" input size is "+str(np.size(xTrn,0))+" batch perc "+str(hp['batch_perc']))
    
    checkpoint = callbacks(hp,h,fold)
    
    print("(ecr_CNN): fitting")
    fit_nn = nn.fit(xTrn,        # Training Data X
                    yTrn,        # Training Data Y
                    validation_data = (xVal,
                                       yVal),  # Validation data tuple
                    shuffle         = 1,              # shuffle the training data epoch before you use it
                    initial_epoch   = 0,              # Starting Epoch (should awlways be 0)
                    epochs          = hp['num_epochs'],     # Number of runs through the data 
                    batch_size      = batch_size,     # Number of samples per gradient update. 
                    verbose         = 1,              # display options to console
                    callbacks=[checkpoint])            # We want to save checkpoints

    
    train_over_time = []
    val_over_time = []
    
    print("(ecr_CNN): evaluating")
    #EVALUATE TRAINING SET
    loss = nn.evaluate(xTrn, yTrn, verbose=0)
    yPred = nn.predict(xTrn,verbose=0)
    [test_names, results] = evaluate_model(yTrn, yPred)
    results = np.append(loss[0],np.array(results))
    train_over_time.append(results)

    #EVALUATE VALIDATION SET
    loss = nn.evaluate(xVal, yVal, verbose=0)
    yPred = nn.predict(xVal,verbose=0)
    [test_names, results] = evaluate_model(yVal, yPred)
    results = np.append(loss[0],np.array(results))
    val_over_time.append(results)
 
    train_over_time = np.vstack(train_over_time)
    val_over_time = np.vstack(val_over_time)
    
    # SAVE THE RESULTS
    np.save('Checkpoints/' + hp['experiment'][h]  + '/fold' + str(fold) +'/performance/train_performance',train_over_time)
    np.save('Checkpoints/' + hp['experiment'][h]  + '/fold' + str(fold) +'/performance/val_performance',val_over_time) 
    np.save('Checkpoints/' + hp['experiment'][h]  + '/fold' + str(fold) +'/performance/metric_names',test_names)
