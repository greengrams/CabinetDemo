import SocketServer
import json
import sys
import atexit
import win32api
sys.path.append('lib')
import os
import numpy as np
from AlignmentDoor import AlignmentDoor
from DoorFunction import Train,DataRepresent
from Processing import SVCTesting

__author__ = 'maeglin89273'

import socket as sk
import wukong_client

HOST = ''
PORT = 3082
server = False

BUFFER_SIZE = 512

filename = 'tmp.csv'

testing_data_buffer=[]

init_state=0
save_state=0
seq_count=0
before_buffer=[]
start_threshold=25
jsonstr_buffer=''
Counter = 0

modelPool=[]
dataPool=[]
p_pool=[]
trainingLabel=[]
scaleRange=[]
scaleMin=[]
LogRegPool=[]
p_min_table=[]

ad=AlignmentDoor()

## 150 rows/person without peak features
# params = [[0.1868, 0.0506], [0.1245, 0.0759], [0.0554, 0.0759], [0.0830,0.0759]]
## 150 rows/person by all features
# params = [[0.1868, 0.0225], [0.1868, 0.0759], [0.0830, 0.0506], [0.0830,0.01]]
## 250 rows/person without peak features
# params = [[0.1096, 0.0506], [0.0487, 0.0506], [0.8325, 0.0759], [0.1096,0.0506]]
## 250 rows/person by all features
# params = [[0.0512, 0.0100], [0.2048, 0.0759], [0.0064, 0.0506], [0.1024,0.0100]]
##

## 0:right door ; 1:left door
params = \
{\
    0:[[0.0064, 0.0100], [0.0032, 0.0100], [0.0032, 0.0150], [0.0064,0.0100]],\
    1:[[0.0064, 0.0100], [0.0008, 0.0100], [0.2048, 0.0338], [0.0016,0.0225]]\
}
# en_params = \
# {\
    # 0:[[2.1,1.9,2.1,2.3],[2.1,3,3,2.2],[2.3,3,3,2.1],[3,2,3,3],[2.2,3,2.2,2.4],[2.2,3,2.2,2.6]],\
    # 1:[[2.3,3,1.7,2.1],[3,3,1.9,2.3],[1.3,2.2,1.7,1.4],[2.6,3,1.8,2],[1.5,3,1.8,2.3],[3,3,3,1.2]]\
# 
en_params=[]
# logis_prob=\
# {\
    # 0:[0.2937,0.11795,0.16898,0.35369],\
    # 1:[0.321451409643,0.558429421174,0.361306837043,0.320023927309]\
# }
## auto find prob. threshold
logis_prob=[]
device=0
device_port={0:3082,1:3081}

def direct_to_model(raw_data):
    global init_state
    global save_state
    global seq_count
    global before_buffer
    global start_threshold
    global modelPool
    global dataPool
    global trainingLabel
    global p_pool
    global scaleRange
    global scaleMin
    global LogRegPool
    global testing_data_buffer
    global en_params
    global device
    global logis_prob
    global p_min_table
    global Counter
    
    #print init_flag
    tmp=[float(a) for a in raw_data['FFA2'].split(",")]
    original_data=[]
    original_data.extend(tmp)
    original_data.append(float(raw_data['Timestamp']))
    original_data.append(float(raw_data['Label']))
    if len(before_buffer)<15:
        before_buffer.append(original_data)
    else:
        before_buffer.remove(before_buffer[0])
        before_buffer.append(original_data)
    
    init_flag=ad.init_shift(tmp)
    if (not init_flag):
        if init_state==0:
            init_state=1
            print 'init... please wait...'
    else:
        if init_state==1:
            init_state=2
            print 'init OK!'
        start_flag=ad.is_start(tmp)
        Counter=Counter+1
        # data = [raw_data['FFA2'], raw_data['Timestamp'], raw_data['Label'],int(start_flag)]
        
        if not ((tmp[4]>-start_threshold)&(save_state==0)):
            ad.update_sum_val(tmp)
        # change state area
        if (tmp[4]<-start_threshold) & (save_state==0) & start_flag:
            # open state
            # start save & output buffer
            testing_data_buffer=[]
            save_state=1
            for each_data in before_buffer:
                data = [seq_count]
                data.extend(each_data)
                data.append(save_state)
                seq_count=seq_count+1
                testing_data_buffer.append(data)
            ##ad.open_init()
            print 'open'
            wukong_client.send(device,int(-2))
        elif (save_state==1) & (not start_flag):
            # stay state
            save_state=2
            print 'stay'
        elif (tmp[4]>start_threshold) & (save_state==2) & start_flag:
            # close state
            save_state=3
            print 'close'
        elif (save_state==3) & (not start_flag) & (ad.is_close()):
            save_state=0
            seq_count=0
            ad.clean_start()
            print 'stop'
            
            test_data = np.array(testing_data_buffer)
            #test_feature = DataRepresent(dataPool, trainingLabel, np.array(test_data), scaleRange, scaleMin,en_params)
            #pVal, probs = Testing(LogRegPool, modelPool, p_pool, test_feature,logis_prob,p_min_table)
            test_feature = DataRepresent(dataPool, trainingLabel, np.array(test_data), scaleRange, scaleMin,en_params)
            pVal= SVCTesting(LogRegPool, modelPool, p_pool, test_feature,logis_prob,p_min_table,scaleRange)
            
            print '=========='
            print str(pVal)
            print '=========='
            wukong_client.send(device,int(pVal))
            Counter = 0
        
        if Counter == 100:
            wukong_client.send(device, 0)
        
        if not (save_state==0):
            data = [seq_count]
            data.extend(original_data)
            data.append(save_state)
            seq_count=seq_count+1
            testing_data_buffer.append(data)
        ##else:
            ##ad.update_init_buffer(tmp)

class TCPHandler(SocketServer.BaseRequestHandler):
    def setup(self):
        print "Connect"
    def handle(self):
        global jsonstr_buffer
        while(True):
            json_data = self.request.recv(1024)
            if not json_data:
                break
            #print json_data
            try:
                raw_data = json.loads(json_data)
                #print raw_data['FFA2']
                try:
                    direct_to_model(raw_data)
                except ValueError as e:
                    print "ValueError({0}): {1}".format(e.errno, e.strerror)
            except(ValueError):
                json_data=jsonstr_buffer+json_data
                jsonstr_buffer=''
                replace_str=json_data.replace("}{", "}||{")
                datas=replace_str.split("||")
                for e_data in datas:
                    try:
                        raw_data = json.loads(e_data)
                        try:
                            direct_to_model(raw_data)
                        except ValueError as e:
                            print "ValueError({0}): {1}".format(e.errno, e.strerror)
                    except(ValueError):
                        jsonstr_buffer=jsonstr_buffer+e_data
        ## udp server
        # json_data = self.request[0]
        # raw_data = json.loads(json_data)
        # direct_to_model(raw_data)
    def finish(self):
        global jsonstr_buffer
        print "Disconnect"
        jsonstr_buffer=''

def start_server():
    global modelPool
    global dataPool
    global trainingLabel
    global p_pool
    global scaleRange
    global scaleMin
    global LogRegPool
    global en_params
    global device
    global params
    global logis_prob
    global p_min_table
    global server
    ad.clean_init()
    
    #device=int(sys.argv[1])
    #PORT=device_port[device]
    device=0
    PORT=device_port[device]
    
    namelist=[]
    for i in xrange(len(sys.argv)):
        if (i>1):
            namelist.append(sys.argv[i])
    now_path=os.path.dirname(os.path.abspath(__file__))
    dataset_path=os.path.join(now_path,'DataSet')
    namelist=[os.path.join(dataset_path,f_) for f_ in namelist]
    
    anomally_list=[]
    if sys.argv[1]!='0':
        anomally_list=[os.path.join(dataset_path,sys.argv[1])]
    
    modelPool, p_pool, dataPool, trainingLabel, scaleRange, scaleMin, LogRegPool, logis_prob, p_min_table = Train(namelist,params[device],en_params,anomally_list)
    
    print 'current ip address: ' + sk.gethostbyname(sk.gethostname()) + ':' + str(PORT)
    
    server = SocketServer.TCPServer((HOST, PORT), TCPHandler)
    server.serve_forever()

def exit_fuction(sig):
    print 'exit function'
    global server
    if server:
        server.server_close()
        print 'close server'

if __name__ == '__main__':
    # atexit.register(exit_fuction,0)
    win32api.SetConsoleCtrlHandler(exit_fuction, True)
    start_server()
# python test_data_server.py 0 shihright.csv jhow.csv jingright.csv rickright.csv
# python test_data_server.py 1 shihleft.csv jhow.csv jingleft.csv rickleft.csv
