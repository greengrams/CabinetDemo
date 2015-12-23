import sys

import numpy as np

from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import pandas as pd

name = {0:"Han", 1:"Jhow", 2:"Jing", 3:"Rick"}

def Training(data, testing, params):
    clf = svm.OneClassSVM(nu=params[0], kernel="rbf", gamma=params[1])
    model = clf.fit(data)
    # p_val = model.decision_function(testing[:, :-1])
    t_val = model.decision_function(data)
    # print t_val.shape
    # print str(np.max(t_val))+' '+str(np.min(t_val))
    p_val = model.decision_function(testing)
    # print str(np.max(p_val))+' '+str(np.min(p_val))
    ##
    # param = '-s 2 -t 2 -n ' + str(params[0]) + ' -g ' + str(params[1])
    # model = svm_train([1 for _ in range(len(data))], data.tolist(), param)
    # p_label, p_acc, p_val = svm_predict(testing[:,-1].tolist(), testing[:, :-1].tolist(), model)
    print 'std'+str(np.std(t_val))
    return model, np.mean(t_val), p_val, np.std(t_val)

def Testing(LogRegPool, modelPool, p_mean_table, testingFeature, logis_prob,p_std_table):
    probs = []
    tmp_probs = np.array([])
    tmp_scaler = np.array([])
    for feature in testingFeature:
        pVal = -1
        idx = 0
        # tmp = []
        # print feature
        for (model, LogReg) in zip(modelPool, LogRegPool):
            # p_label, _, p_val = svm_predict([1], [feature.tolist()], model)
            p_val = model.decision_function(feature)
            print model.predict(feature)
            standard_scaler_val=(p_val-p_mean_table[idx])/(p_std_table[idx])
            print standard_scaler_val
            tmp_scaler = np.insert(tmp_scaler, tmp_scaler.shape[0], standard_scaler_val)
            # print p_val[0][0]/p_table[idx]
            # print LogReg.predict_proba(np.array(p_val))
            tmp_probs = np.insert(tmp_probs, tmp_probs.shape[0], LogReg.predict_proba(np.array(p_val))[0][1])
            # tmp.append(p_val[0][0]/p_table[idx])
            idx += 1
        #if np.sum(np.array(tmp) > 0) != 0:
        #    pVal = np.where(tmp == np.max(tmp))[0][0]
        #if np.max(tmp_probs) > 0.5 and pVal == -1:
        #    tmp_probs[tmp_probs==np.max(tmp_probs)] -= 0.5
        ##
        # if np.max(tmp_probs) > 0.3:
        #     pVal=np.where(tmp_probs == np.max(tmp_probs))[0][0]
        ## by logistic
        # tmp_probs=np.array(tmp_probs)
        # label_index=np.array(range(len(LogRegPool)))
        # label_index=label_index[tmp_probs>logis_prob]
        # if len(label_index)>0:
            # a=tmp_probs[label_index]
            # print a
            # pVal=label_index[np.where(a==max(a))][0]
            ## let 0123 to 1234
            # pVal=pVal+1
        ##
        if (sum(tmp_scaler>-2)>0):
            pVal=np.where(tmp_scaler==max(tmp_scaler))
            # let 0123 to 1234
            pVal=pVal[0][0]+1
        for x in tmp_probs.tolist():
            probs.append((str(np.around(x * 100, decimals=3)) + '%' , str(np.around(x / (np.sum(np.array(tmp_probs))) * 100, decimals=3)) + '%'))
    return pVal, probs

def modelTrain(features,labels):
# #     Location = r'D:/slipper1016/frank.csv'
    # traindf = pd.DataFrame()
    # label = np.array([], dtype=np.int64).reshape(0,1)
    # for item in trainFileNamesT:
        # tempTraindfTrue = pd.read_csv(item, header=None)
        # print tempTraindfTrue.values.shape
        # tempLabelTrue = np.zeros((len(tempTraindfTrue),1))+1
        # traindf=traindf.append(tempTraindfTrue, ignore_index=True)
        # label = np.vstack([label, tempLabelTrue])
# #         label = label.append(tempLabelTrue, ignore_index=True)
    # for item in trainFileNamesF:
        # tempTraindfTrue = pd.read_csv(item, header=None)
        # print tempTraindfTrue
        # tempLabelTrue = np.zeros((len(tempTraindfTrue),1))
        # traindf=traindf.append(tempTraindfTrue, ignore_index=True)
# #         label = label.append(tempLabelTrue, ignore_index=True)
        # label = np.vstack([label, tempLabelTrue])
    # traindfArray = traindf.values
    # label = np.ravel(label)
    traindfArray=features
    label=np.array(labels)
    #for tempidx,item in enumerate(traindfArray):
        #print tempidx,item
    print traindfArray.shape
    normalizer = StandardScaler().fit(traindfArray)
    traindfArray=normalizer.transform(traindfArray)
    param_grid = [  
      {'C': np.linspace(1.0, 500.0, num=10)/1000, 'gamma': np.linspace(1.0, 1000.0, num=10)/10000, 'kernel': ['rbf']},
     ]
    print 'label0',np.sum(label==0),'label1',np.sum(label==1)
    weight_=float(np.sum(label==0))/float(np.sum(label==1))
    print weight_
    zero_weight_=float(2.0)/weight_
    print 'zero_weight:'+str(zero_weight_)
    clf = grid_search.GridSearchCV(SVC(class_weight={0:zero_weight_}), param_grid,scoring='f1', cv=KFold(len(label), n_folds=5,shuffle =True,random_state =10),verbose =1)
    clf.fit(traindfArray, label)
    bestModel = clf.best_estimator_
    print bestModel.get_params()
    print 'score:'+str(clf.best_score_)
    return bestModel,normalizer

def SVCTraining(data,features,labels):
    bestModel,normalizer=modelTrain(features,labels)
    # scale
    data=normalizer.transform(data)
    p_val = bestModel.decision_function(data)
    print p_val
    print 'std: '+str(np.std(p_val))
    return bestModel, normalizer,np.mean(p_val), np.std(p_val)

def SVCTesting(LogRegPool, modelPool, p_mean_table, testingFeature, logis_prob,p_std_table,scaleRange):
    # probs = []
    # tmp_probs = np.array([])
    for feature in testingFeature:
        pVal = -1
        idx = 0
        # tmp = []
        # print feature
        tmp_scaler = np.array([])
        tmp_predict=[]
        for (model,normalizer) in zip(modelPool,scaleRange):
            e_feature=normalizer.transform(feature)
            p_val=model.decision_function(e_feature)
            e_predict=model.predict(e_feature)
            print e_predict
            tmp_predict.append(int(e_predict))
            # print p_val
            # print p_mean_table[idx]
            # print p_std_table[idx]
            # p_val = model.decision_function(feature)
            standard_scaler_val=(p_val-p_mean_table[idx])/(p_std_table[idx])
            print standard_scaler_val
            tmp_scaler = np.insert(tmp_scaler, tmp_scaler.shape[0], standard_scaler_val)
            # print p_val[0][0]/p_table[idx]
            # print LogReg.predict_proba(np.array(p_val))
            # tmp_probs = np.insert(tmp_probs, tmp_probs.shape[0], LogReg.predict_proba(np.array(p_val))[0][1])
            # tmp.append(p_val[0][0]/p_table[idx])
            idx += 1
        #if np.sum(np.array(tmp) > 0) != 0:
        #    pVal = np.where(tmp == np.max(tmp))[0][0]
        #if np.max(tmp_probs) > 0.5 and pVal == -1:
        #    tmp_probs[tmp_probs==np.max(tmp_probs)] -= 0.5
        ##
        # if np.max(tmp_probs) > 0.3:
        #     pVal=np.where(tmp_probs == np.max(tmp_probs))[0][0]
        ## by logistic
        # tmp_probs=np.array(tmp_probs)
        # label_index=np.array(range(len(LogRegPool)))
        # label_index=label_index[tmp_probs>logis_prob]
        # if len(label_index)>0:
            # a=tmp_probs[label_index]
            # print a
            # pVal=label_index[np.where(a==max(a))][0]
            ## let 0123 to 1234
            # pVal=pVal+1
        ##
        predict_index=np.array(range(len(tmp_predict)))
        tmp_predict=np.array(tmp_predict)
        predict_index=predict_index[tmp_predict==1]
        if (len(predict_index)>0):
            a=tmp_scaler[tmp_predict==1]
            pVal=predict_index[np.where(a==max(a))][0]
            # let 0123 to 1234
            pVal=pVal+1
        # for x in tmp_probs.tolist():
            # probs.append((str(np.around(x * 100, decimals=3)) + '%' , str(np.around(x / (np.sum(np.array(tmp_probs))) * 100, decimals=3)) + '%'))
    return pVal
