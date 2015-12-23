# -*- coding: utf-8 -*-
import os.path
import sys
sys.path.append('lib')
import random as rd
import numpy as np
from datetime import datetime
from Envelope import envelope
# from Demo_UI import Base
from Processing import Training,SVCTraining
from Sampling import OverSampling
from Vectorization import Vectorize
from PreProcessing import Preprocessing
from sklearn.linear_model import LogisticRegression
from dynamicRangeCheck import dynamicRangeCheck
import copy
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedKFold
def GetFeatures(data,trainingLabel,dataPool,en_params):
    features = np.zeros((data.shape[1], 1))
    # Vectorization
    for x in data:
        features = np.insert(features, features.shape[1], Vectorize(x), axis=1)
    
    # -1 label num for each axis
    num_n1=np.sum(np.array(trainingLabel)==-1)/float(len(dataPool[0]))
    num_n1=int(num_n1)
    
    features = np.delete(features, 0, axis=1)
    # Envelope
    for idx in range(len(dataPool[0])):
        tmp = []
        axis_en_params=[]
        if len(en_params)>idx:
            axis_en_params=en_params[idx]
        for i in range(len(dataPool)):
            tmp.extend(dataPool[i][idx].tolist())

        envelopeResult = np.array(envelope(np.array(trainingLabel[idx*(len(tmp)+num_n1):(idx+1)*len(tmp)+idx*num_n1]), tmp, data[idx].tolist(), axis_en_params))
        if len(en_params)<=idx:
            en_params.append(axis_en_params)
        features = np.insert(features, features.shape[1], envelopeResult.T, axis=1)
        # print 'feature shape'
        # print features.shape

    return features
def TrainingByParams(dataPool, anomalyPool, trainingLabel,params,en_params):
    features = []
    labels = []
    rangeOfData = [0]
    modelPool = []
    p_pool = []
    p_table = []
    LogRegPool = []
    p_min_table=[]

    # The flag of current person
    currentGuy = 0
    #print_flag = 0
    for data in dataPool:
        
        vectorFeature=GetFeatures(data,trainingLabel,dataPool,en_params)
        
        if len(features)==0:
            features=np.zeros((1, vectorFeature.shape[1]))
        features = np.insert(features, features.shape[0], vectorFeature, 0)
        # Create testing lable
        labels.extend([currentGuy for _ in xrange(vectorFeature.shape[0])])
        
        currentGuy +=1
        rangeOfData.append(rangeOfData[len(rangeOfData)-1] + data.shape[1])
    ## add anomally.csv
    for data in anomalyPool:
        
        vectorFeature=GetFeatures(data,trainingLabel,dataPool,en_params)
        
        if len(features)==0:
            features=np.zeros((1, vectorFeature.shape[1]))
        features = np.insert(features, features.shape[0], vectorFeature, 0)
        # Create testing lable
        labels.extend([-1 for _ in xrange(vectorFeature.shape[0])])
        
        # rangeOfData.append(rangeOfData[len(rangeOfData)-1] + data.shape[1])
    
    features = np.delete(features, 0, axis=0)
    
    ## output feature file
    # out = open('door_features.csv', 'w')
    # for each_data_f in features:
    #     out.write(','.join([str(x) for x in each_data_f]) + '\n')
    # out.close()
    # out = open('door_labels.csv', 'w')
    # out.write(','.join([str(x) for x in labels]) + '\n')
    # out.close()
    
    # Max-Min Normalize
    # scaleRange = np.abs(np.max(features, 0) - np.min(features, 0))
    # scaleMin = np.min(features, 0)

    # Max and Min is 0, avoiding to divide by zero
    # scaleRange[scaleRange == 0] = 1
    # features = (features - scaleMin)/scaleRange
    
    ##
    # for i in xrange(len(set(labels))):
        # filename='feature'+str(i)+'.csv'
        # out = open(filename, 'w')
        # target_features=features[np.array(labels)==i]
        # for j in xrange(target_features.shape[0]):
            # out.write(','.join([str(x) for x in target_features[j]]) + '\n')
    ##
    # Logistic_C=np.array([1e-2,1e-1,1,1e1,1e2])
    target_percent=0.95
    logis_prob=[]
    scaleRange=[]
    for i in range(len(dataPool)):
        sample=features[rangeOfData[i]:rangeOfData[i+1]]
        # trainFileNamesT=[]
        # trainFileNamesF=[]
        # for j in range(len(dataPool)):
            # if i==j:
                # trainFileNamesT.append(r'feature'+str(j)+'.csv')
            # else:
                # trainFileNamesF.append(r'feature'+str(j)+'.csv')
        input_labels=np.zeros(len(labels))
        input_labels[np.array(labels)==i]=1
        bestModel, normalizer,p_val_mean, p_val_std=SVCTraining(sample, features,input_labels)
        p_vals=[]
        LogReg=[]
        scaleMin=[]
        # label  = np.array([0 for _ in range(features.shape[0])])
        # label[rangeOfData[i]:rangeOfData[i+1]] = 1
        # # OverSampling
        # # sample = OverSampling(np.insert(features, features.shape[1], label, axis=1))
        # sample=features[rangeOfData[i]:rangeOfData[i+1]]
        # model, p_val, p_vals, p_min_val = Training(sample, features, params[i])
        # df_val=np.array(p_vals)
        # ## Logistic Regression
        # # LogReg = LogisticRegression(C=1e-3,fit_intercept=False)
        # # LogReg.fit(np.array(df_val), label)
        # ##
        # LogReg = LogisticRegression(C=1e2)
        # LogReg.fit(np.array(df_val), label)
        
        ## tune Logistic_C by 5-fold
        # fold_num=5
        # all_tErr=[]
        # for c_ind in xrange(len(Logistic_C)):
            # LogReg = LogisticRegression(C=Logistic_C[c_ind])
            
            # meanErr=0
            # kf = StratifiedKFold(y=label, n_folds=fold_num)
            # for train, test in kf:
                # LogReg.fit(np.array(df_val[train]), label[train])
                # probs=LogReg.predict_proba(df_val[train])[:,1]
                # probs.sort()
                # target_prob=probs[int(np.floor((1-target_percent)*len(probs)))]
                # test_probs=LogReg.predict_proba(df_val[test])[:,1]
                # true_label=label[test]
                # test_label=np.zeros(len(test_probs))
                # test_label[test_probs>target_prob]=1
                # tErr=np.sum(np.abs(test_label-true_label))/len(label)
                # # print 'tErr'+str(tErr)
                # meanErr=meanErr+tErr
            # meanErr=meanErr/fold_num
            # print 'C='+str(Logistic_C[c_ind])+' meanErr='+str(meanErr)
            # all_tErr.append(meanErr)
        # all_tErr=np.array(all_tErr)
        # best_c=Logistic_C[all_tErr==min(all_tErr)][0]
        # print 'best_c='+str(best_c)
        # # choice best c
        # LogReg = LogisticRegression(C=best_c)
        # LogReg.fit(np.array(df_val), label)
        #### plot LogReg
        # x1=[]
        # y1=[]
        # x0=[]
        # y0=[]
        # for j in xrange(np.array(p_vals).shape[0]):
            # if label[j]==1:
                # x1.append(p_vals[j])
                # y1.append(LogReg.predict_proba(np.array(p_vals))[j][1])
            # else:
                # x0.append(p_vals[j])
                # y0.append(LogReg.predict_proba(np.array(p_vals))[j][1])
        
        # ## find 95% self prob.
        # tmp_prob = copy.copy(y1)
        # tmp_prob.sort()
        # target_prob=tmp_prob[int(np.floor((1-target_percent)*len(tmp_prob)))]
        # print 'label:'+str(i)+' prob:'+str(target_prob)
        # ## add to prob. threshold
        # logis_prob.append(target_prob)
        ## plot
        # plt.figure()
        # plt.plot(x0,y0,'bo')
        # plt.plot(x1,y1,'ro')
        # plt.title('Logistic_C='+str(Logistic_C)+' ; '+str(int(target_percent*100))+'%self-prob.='+str(target_prob))
        # plt.show()
        ####
        
        p_pool.append(p_val_mean)
        p_table.append(p_vals)
        p_min_table.append(p_val_std)
        modelPool.append(bestModel)
        LogRegPool.append(LogReg)
        scaleRange.append(normalizer)

    print "Finish"
    return modelPool, p_pool, p_table, features, labels, scaleRange, scaleMin, rangeOfData, LogRegPool, logis_prob,p_min_table

def DataRepresent(dataPool, trainingLabel, rawdata, scaleRange, scaleMin,en_params):
    # Preprocessing
    temp_ind=rawdata[:,0]    
    temp_ind = temp_ind.reshape((len(temp_ind),1))
    rawdata = np.hstack((temp_ind, dynamicRangeCheck(rawdata[:,1:7])))
    [axis1, axis2, axis3, axis4, axis5, axis6] = Preprocessing(rawdata, maxLen=200, n=5)
    testingData = np.array([axis1, axis2, axis3, axis4, axis5, axis6])

    features=GetFeatures(testingData,trainingLabel,dataPool,en_params)
    # Max-min Normalize
    #features = (features-scaleMin)/scaleRange

    return features

def LoadTrainingData(namelist,anomally_list):
    print namelist
    dataPool = []
    anomalyPool=[]
    trainingLabel = []
    i = 0
    # Load the data in numpy's type
    for name in namelist:
        data = np.genfromtxt(name, delimiter=',')
        print data.shape
        # Do preprocessing & moving average
        temp_ind=data[:,0]    
        temp_ind = temp_ind.reshape((len(temp_ind),1))
        data = np.hstack((temp_ind, dynamicRangeCheck(data[:,1:7])))
        [axis1, axis2, axis3, axis4, axis5, axis6] = Preprocessing(data, maxLen=200, n=5)

        cleanedData = np.array([axis1, axis2, axis3, axis4, axis5, axis6])

        # Collect data which has been processing
        dataPool.append(cleanedData)
        # Create training label
        trainingLabel.extend([i for _ in range(axis1.shape[0])])
        i+=1
    
    # anomaly
    for name in anomally_list:
        data = np.genfromtxt(name, delimiter=',')
        print data.shape
        # Do preprocessing & moving average
        temp_ind=data[:,0]    
        temp_ind = temp_ind.reshape((len(temp_ind),1))
        data = np.hstack((temp_ind, dynamicRangeCheck(data[:,1:7])))
        [axis1, axis2, axis3, axis4, axis5, axis6] = Preprocessing(data, maxLen=200, n=5)

        cleanedData = np.array([axis1, axis2, axis3, axis4, axis5, axis6])

        # Collect data which has been processing
        anomalyPool.append(cleanedData)
        # Create training label
        trainingLabel.extend([-1 for _ in range(axis1.shape[0])])
    trainingLabel = trainingLabel * len(dataPool[0])
    
    return dataPool, anomalyPool, trainingLabel

def Train(namelist,params,en_params,anomally_list):
    dataPool, anomalyPool, trainingLabel= LoadTrainingData(namelist,anomally_list)
    ## Training Model
    modelPool, p_pool, p_table, testingData, _, scaleRange, scaleMin, rangeOfData, LogRegPool, logis_prob, p_min_table = TrainingByParams(dataPool, anomalyPool, trainingLabel,params,en_params)

    ## Use intruder data
    #data = np.genfromtxt(intruder, delimiter=',')
    #print data.shape
    ## Do preprocessing & moving average
    #features = DataRepresent(dataPool, trainingLabel, data, scaleRange, scaleMin)
    ## Random sampling
    #features = features[rd.sample(range(len(features)), 1), :]
    #print features.shape
    #Testing(LogRegPool, modelPool, p_tabel, features, [-1 for _ in range(len(features))])
    #print "finish"

    return modelPool, p_pool, dataPool, trainingLabel, scaleRange, scaleMin, LogRegPool, logis_prob, p_min_table

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: python ReadSerial.py <fileName>"
        exit(1)

    namelist = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    test_data = np.genfromtxt(sys.argv[5], delimiter=',')

    modelPool, p_table, dataPool, trainingLabel, scaleRange, scaleMin, LogRegPool = Train(namelist)
    
    test_feature = DataRepresent(dataPool, trainingLabel, np.array(test_data), scaleRange, scaleMin)
    pVal, probs = Testing(LogRegPool, modelPool, p_table, test_feature, [1])
