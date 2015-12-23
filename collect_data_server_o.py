import SocketServer
import json
import sys
sys.path.append('lib')
from AlignmentDoor import AlignmentDoor

__author__ = 'maeglin89273'

import socket as sk

HOST = ''
PORT = 3075

BUFFER_SIZE = 512

filename = 'tmp.csv'

out=[]
sum_file=[]
# err_f=[]
jsonstr_buffer=''
init_state=0
save_state=0
seq_count=0
before_buffer=[]
start_threshold=25
open_count=0

ad=AlignmentDoor()

def direct_to_model(raw_data):
    global init_state
    global save_state
    global seq_count
    global before_buffer
    global start_threshold
    global open_count
    global out
    global sum_file
    tmp=[float(a) for a in raw_data['FFA2'].split(",")]
    ## save sum_data
    sum_data=[]
    sum_data.append(ad.get_sum_val())
    sum_data.append(ad.get_min_sum_val())
    sum_data.append(ad.get_noise_shift())
    sum_data.append(ad.get_mask_val())
    sum_data.extend(tmp)
    sum_data.append(raw_data['Timestamp'])
    sum_file.write(','.join([str(x) for x in sum_data]) + '\n')
    ##
    data = [raw_data['FFA2'], raw_data['Timestamp'], raw_data['Label']]
    if len(before_buffer)<15:
        before_buffer.append(data)
    else:
        before_buffer.remove(before_buffer[0])
        before_buffer.append(data)
    
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
        # data = [raw_data['FFA2'], raw_data['Timestamp'], raw_data['Label'],int(start_flag)]
        
        if not ((tmp[4]>-start_threshold)&(save_state==0)):
            ad.update_sum_val(tmp)
        # change state area
        if (tmp[4]<-start_threshold) & (save_state==0) & start_flag:
            # open state
            # start save & output buffer
            save_state=1
            for each_data in before_buffer:
                data = [seq_count]
                data.extend(each_data)
                data.append(save_state)
                seq_count=seq_count+1
                out.write(','.join([str(x) for x in data]) + '\n')
            ##ad.open_init()
            open_count=open_count+1
            print open_count
            print 'open'
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
        
        if not (save_state==0):
            data = [seq_count,raw_data['FFA2'], raw_data['Timestamp'], raw_data['Label']]
            data.append(save_state)
            seq_count=seq_count+1
            out.write(','.join([str(x) for x in data]) + '\n')
        ## else:
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
            except(ValueError):
                json_data=jsonstr_buffer+json_data
                jsonstr_buffer=''
                replace_str=json_data.replace("}{", "}||{")
                datas=replace_str.split("||")
                for e_data in datas:
                    try:
                        raw_data = json.loads(e_data)
                    except(ValueError):
                        jsonstr_buffer=jsonstr_buffer+e_data
                    direct_to_model(raw_data)
            direct_to_model(raw_data)
            ## udp server
        # json_data = self.request[0]
        # raw_data = json.loads(json_data)
        # direct_to_model(raw_data)
    def finish(self):
        print "Disconnect"

def start_server():
    global filename
    global out
    global sum_file
    # global err_f
    print 'current ip address: ' + sk.gethostbyname(sk.gethostname()) + ':' + str(PORT)
    ad.clean_init()
    
    filename=sys.argv[1]
    
    out = open(filename, 'a')
    sum_file = open('sum_'+filename, 'a')
    # err_f=open('err_'+filename, 'w')
    server = SocketServer.TCPServer((HOST, PORT), TCPHandler)
    server.serve_forever()


if __name__ == '__main__':
    start_server()
