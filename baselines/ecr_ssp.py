#!/usr/bin/env python
# coding: utf-8


import scipy.io
import glob
import numpy as np
import scipy.stats
import json
import time
import os
import sys
from scipy.special import legendre
import keras


fold  = int(sys.argv[1])


def load_val_data(fold):
    l = range(10)
    trnIdx = [fold]
    trn = []
    for ii in trnIdx:
        foldData = np.load('ecr_data_'+str(ii)+'.npy', allow_pickle=True)[()]
        for subjData in foldData:
            for ttData in foldData[subjData]:
                [org,data] = foldData[subjData][ttData]
                trn.append(org)
    return trn


class baselines:
    # for ssp function following:
    # https://github.com/openroc/eeglab/blob/master/branches/
    # eeglab10/external/bioelectromagnetism_ligth/eeg_lap_sph_spline.m
    def __init__(self,orgEEG,system='10-20_system'):
        
        ignoreCh = set(['EEGA2A1','EDFAnnotations','ECGECG']);
        
        self.sysMap = scipy.io.loadmat('../../ECR/maps/'+system+'.mat')
        mapChan = list(set([ channel[0] for channel in self.sysMap['map'].flatten().tolist()]))
        self.eegChan = [orgEEG['chanlocs'][0][0,ii][0][0].replace(' ','').replace('-','') for ii in range(np.size(orgEEG['chanlocs'][0]))]
        distLabels = [x[0] for x in list(orgEEG['chanlocs'][0].dtype.fields.items())]
        xIdx = distLabels.index('X')
        yIdx = distLabels.index('Y')
        zIdx = distLabels.index('Z')
        thtIdx = distLabels.index('sph_theta')
        phiIdx = distLabels.index('sph_phi')
        self.distances = {}
        for ii,chan in enumerate(self.eegChan):
            if chan in ignoreCh:
                continue
            self.distances[chan] = {}
            self.distances[chan]['x'] = float(orgEEG['chanlocs'][0][0,ii][xIdx])
            self.distances[chan]['y'] = float(orgEEG['chanlocs'][0][0,ii][yIdx])
            self.distances[chan]['z'] = float(orgEEG['chanlocs'][0][0,ii][zIdx])
            self.distances[chan]['tht'] = np.radians(float(orgEEG['chanlocs'][0][0,ii][thtIdx]))
            self.distances[chan]['phi'] = np.radians(float(orgEEG['chanlocs'][0][0,ii][phiIdx]))
        self.sort_chan = list(self.distances.keys())
        self.sort_chan.sort()
        self.m = 4
        self.calc_cosines()
        self.calc_g()
        return
            
    def EUD(self,EEG,p=1):
        mse = []
        for intChan in  self.distances.keys():
            [ii],[jj]=np.where(self.sysMap['map'] == [intChan])
            orgChan = EEG[ii,jj,:].reshape(np.shape(EEG)[2])
            newChan = np.zeros(np.shape(EEG)[2])
            normChan = 0
            for chan in self.distances.keys():
                dij = 0
                if chan == intChan:
                    continue
                dij += (self.distances[chan]['x']-self.distances[intChan]['x'])**2
                dij += (self.distances[chan]['y']-self.distances[intChan]['y'])**2
                dij += (self.distances[chan]['z']-self.distances[intChan]['z'])**2
                dij = np.sqrt(dij)
                [ii],[jj]=np.where(self.sysMap['map'] == [chan])
                newChan += EEG[ii,jj,:].reshape(np.shape(EEG)[2]) / (dij**p)
                normChan += 1/(dij**p)
            newChan = newChan / normChan
            orgChan = scipy.stats.zscore(orgChan)
            newChan = scipy.stats.zscore(newChan)
            mse.append(float(keras.losses.mean_squared_error(orgChan,newChan)))
        return mse
    
    def new_EUD(self,EEG,ii,jj,p=1):
        orgChan = EEG[ii,jj,:].reshape(np.shape(EEG)[2])
        newChan = np.zeros(np.shape(EEG)[2])
        intChan = self.sysMap['map'][ii,jj][0]
        normChan = 0
        for chan in self.distances.keys():
            dij = 0
            if chan == intChan:
                continue
            dij += (self.distances[chan]['x']-self.distances[intChan]['x'])**2
            dij += (self.distances[chan]['y']-self.distances[intChan]['y'])**2
            dij += (self.distances[chan]['z']-self.distances[intChan]['z'])**2
            dij = np.sqrt(dij)
            [ii],[jj]=np.where(self.sysMap['map'] == [chan])
            newChan += EEG[ii,jj,:].reshape(np.shape(EEG)[2]) / (dij**p)
            normChan += 1/(dij**p)
        newChan = newChan / normChan
        orgChan = scipy.stats.zscore(orgChan)
        newChan = scipy.stats.zscore(newChan)
        return newChan
    
    def GCD(self,EEG):
        mse = []
        for intChan in  self.distances.keys():
            [ii],[jj]=np.where(self.sysMap['map'] == [intChan])
            orgChan = EEG[ii,jj,:].reshape(np.shape(EEG)[2])
            newChan = np.zeros(np.shape(EEG)[2])
            for chan in self.distances.keys():
                dij = 0
                if chan == intChan:
                    continue
                dij += np.sin((self.distances[intChan]['phi']-self.distances[chan]['phi'])/2)**2
                dij += np.cos(self.distances[intChan]['phi'])*self.distances[chan]['phi']*                     np.sin((self.distances[intChan]['tht']-self.distances[chan]['tht'])/2)**2
                #print(dij)
                dij = np.sqrt(dij)
                dij = 2*np.arcsin(dij)
                [ii],[jj]=np.where(self.sysMap['map'] == [chan])
                newChan += dij*EEG[ii,jj,:].reshape(np.shape(EEG)[2])
            orgChan = scipy.stats.zscore(orgChan,2)
            newChan = scipy.stats.zscore(newChan,2)
            mse.append(float(keras.losses.mean_squared_error(orgChan,newChan)))
        return mse
    
    def EGL(self,EEG):
        mse = []
        for intChan in  self.distances.keys():
            [ii],[jj]=np.where(self.sysMap['map'] == [intChan])
            orgChan = EEG[ii,jj,:].reshape(np.shape(EEG)[2])
            newChan = np.zeros(np.shape(EEG)[2])
            for chan in self.distances.keys():
                dij = 0
                if chan == intChan:
                    continue
                dij = vincenty.vincenty((self.distances[intChan]['phi'],self.distances[intChan]['tht']),                               (self.distances[chan]['phi'],self.distances[chan]['tht']))
                [ii],[jj]=np.where(self.sysMap['map'] == [chan])
                newChan += dij*EEG[ii,jj,:].reshape(np.shape(EEG)[2])
            orgChan = scipy.stats.zscore(orgChan)
            newChan = scipy.stats.zscore(newChan)
            mse.append(float(keras.losses.mean_squared_error(orgChan,newChan)))
        return mse
    
    def new_EGL(self,EEG,ii,jj,p=1):
        orgChan = EEG[ii,jj,:].reshape(np.shape(EEG)[2])
        newChan = np.zeros(np.shape(EEG)[2])
        intChan = self.sysMap['map'][ii,jj][0]
        normChan = 0
        for chan in self.distances.keys():
            dij = 0
            if chan == intChan:
                continue
            dij = vincenty.vincenty((self.distances[intChan]['phi'],self.distances[intChan]['tht']),                           (self.distances[chan]['phi'],self.distances[chan]['tht']))
            [ii],[jj]=np.where(self.sysMap['map'] == [chan])
            newChan += EEG[ii,jj,:].reshape(np.shape(EEG)[2]) / (dij**p)
            normChan += 1/(dij**p)
        newChan = newChan / normChan
        orgChan = scipy.stats.zscore(orgChan)
        newChan = scipy.stats.zscore(newChan)
        return newChan
    
    def calc_cosines(self):
        self.cosines = np.zeros((len(self.sort_chan),len(self.sort_chan)))
        for ii,chanA in enumerate(self.sort_chan):
            for jj,chanB in enumerate(self.sort_chan):
                x1 = self.distances[chanA]['x']
                y1 = self.distances[chanA]['y']
                z1 = self.distances[chanA]['z']
                x2 = self.distances[chanB]['x']
                y2 = self.distances[chanB]['y']
                z2 = self.distances[chanB]['z']
                self.cosines[ii,jj] = (x1*x2+y1*y2+z1*z2)/                 (np.sqrt(x1**2 + y1**2 + z1**2)*np.sqrt(x2**2 + y2**2 + z2**2))
        return
    
    def calc_g(self):
        self.g = np.zeros((len(self.sort_chan),len(self.sort_chan)))
        for p in range(1,8):
            Pn = legendre(p)
            for ii,chanA in enumerate(self.sort_chan):
                for jj,chanB in enumerate(self.sort_chan):
                    self.g[ii,jj] += (1/(4*np.pi))*                     (2*p+1)/((p**self.m)*((p+1)**self.m))                     * Pn(self.cosines[ii,jj]) 
        return
    
    def calc_coeff(self,V_list):
        coeff = []
        for V in V_list:
            c = []
            for ii,intChan in enumerate(self.sort_chan):
                g = np.delete(self.g, ii, axis=0)
                g = np.delete(g, ii, axis=1)
                v = np.delete(V, ii, axis=0)
                Gx = np.ones((len(self.sort_chan),len(self.sort_chan)))
                Gx[1:,1:] = g
                Gx[0,0] = 0
                CoV = np.insert(v,0,0)
                c.append(np.linalg.solve(Gx,CoV))
            coeff.append(c)
        return coeff
            
            
            
            
    def new_SSM(self,EEG,ii,jj,m=4):
        orgChan = EEG[ii,jj,:].reshape(np.shape(EEG)[2])
        newChan = np.zeros(np.shape(EEG)[2])
        intChan = self.sysMap['map'][ii,jj][0]            

# This file is from the original data and simply carries meta
# information such as channel names and locations
files = glob.glob('./subject*_1_intrp.mat')
orgEEG = scipy.io.loadmat(files[0])['EEG'][0]
baseline = baselines(orgEEG)


trn = load_val_data(fold)

mse = []


for data in trn:
    full_data = []
    for chan in  baseline.sort_chan:
        [ii],[jj]=np.where(baseline.sysMap['map'] == [chan])
        full_data.append(data[ii,jj,:].reshape(np.shape(data)[2]))

    chan_unsorted = baseline.sort_chan
    data_unsorted = full_data
    data = []
    for chan in baseline.sort_chan:
        ii = chan_unsorted.index(chan)
        data.append(data_unsorted[ii])
    V_tmp = np.stack(data, axis=0)
    print('here',np.shape(V_tmp))
    V = [V_tmp[:,ii] for ii in range(np.shape(V_tmp)[1])]
    coeff = baseline.calc_coeff(V)
    ssp = []
    gt = []
    for jj in range(len(V)):
        sspT = []
        gtT = []
        for ii in range(19):
            g = np.delete(baseline.g[ii,:], ii, axis=0)
            g = np.insert(g,0,1)
            sspT.append(np.dot(coeff[jj][ii],g))
            gtT.append(V[jj][ii])
        ssp.append(np.array(sspT))
        gt.append(np.array(gtT))
    ssp = np.vstack(ssp)
    gt = np.vstack(gt)
    for ii in range(19):
        mse.append(float(keras.losses.mean_squared_error(scipy.stats.zscore(ssp[:,ii]),                                                         scipy.stats.zscore(gt[:,ii]))))



np.save('./SSP_data'+str(fold),mse)
