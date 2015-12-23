# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 13:57:28 2015

@author: shih
"""
import numpy as np
import copy


def dynamicRangeCheck(ts, wSize=5, iteration=1):
    TS = copy.copy(ts)
    '''
    % for noisy data to remove some peaks before using moving avg.
    % input
    %   TS - numpy ndarray, column-wise
    %   wSize - window size for range checking
    %   iteration - how many rounds for data smoothing 
    %                  in case that each series may come from different source
    %                  an array of integers is available
    % output
    %   data - cleaned data    
    '''
    
    
    iteration = iteration*np.ones((1,6))
    
    std1 = [2,2,2,3,3,3];
    std2 = [.25, .25, .25, .5, .5, .5];
    
    
    for i in range(TS.shape[1]):
        base_m = np.mean(TS[:,i])
        base_s = np.std(TS[:,i])
        m = base_m
        s = base_s
        
        boo = (TS[:, i] > (base_m + std1[i]*base_s)) | (TS[:, i] < (base_m - std1[i]*base_s))
        TS[boo, i] = base_m;       # replace large peak
        steady = (TS[:,i] < (base_m + std2[i]*base_s)) & (TS[:, i] > (base_m - std2[i]*base_s))
        
        steady = np.reshape(TS[steady, i],(len(TS[steady, i]),1))
        if steady.shape[0]==0:
            steady = np.array([base_m])
        
        for j in range(wSize):            
            if (TS[j,i] > (base_m + .5*base_s)) | (TS[j, i] < (base_m - .5*base_s)):
                TS[j, i] = steady[np.random.choice(len(steady), 1)];
        
        for ite in range(int(iteration[0,i])):
            for j in range(wSize, TS.shape[0]-wSize):
                # suppose that at the begining of each time series is steady
                if (TS[j, i] >= m+2*s) | (TS[j, i] <= m-2*s):       # smooth medium peak
                    TS[j, i] = np.mean([TS[int(j-round(wSize/2)), i], TS[int(j+round(wSize/2)), i]])
                    
                if abs(TS[j, i]-TS[j-1, i]) > 1*base_s:          # smooth small peak 
                    TS[j, i] = np.mean([TS[int(j-round(wSize/2)), i], TS[int(j+round(wSize/2)), i]])
                
                #print [i, j]
                #raw_input()
                m = np.mean(TS[j-wSize+1:j,i])
                s = np.std(TS[j-wSize+1:j,i])
    return TS            

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    temp = np.genfromtxt('E:\Dropbox\inel_project\Data\ecomini\ccl0925.csv', delimiter=',')
    
    temp_ind=temp[:,0]    
    temp_ind = temp_ind.reshape((len(temp_ind),1))
    #print temp_ind.shape
    #raw_input()
    
    #result = dynamicRangeCheck(temp[:,1:7])
    
    result = np.hstack((temp_ind, dynamicRangeCheck(temp[:,1:7])))
    #print result.shape
    plt.figure()
    plt.plot(temp[:,3])
    plt.figure()
    plt.plot(result[:,3])
    
