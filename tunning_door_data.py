# -*- coding: utf-8 -*-
import sys
sys.path.append('lib')
import os
import numpy as np
from DoorFunction import *
from Processing import *
if __name__ == '__main__':
    namelist=[]
    for i in xrange(len(sys.argv)):
        if not (i==0):
            namelist.append(sys.argv[i])
    now_path=os.path.dirname(os.path.abspath(__file__))
    dataset_path=os.path.join(now_path,'DataSet')
    namelist=[os.path.join(dataset_path,f_) for f_ in namelist]
    
    dataPool, trainingLabel=LoadTrainingData(namelist)
    
    
# python tunning_door_data.py shih.csv jhow.csv jing.csv rick.csv