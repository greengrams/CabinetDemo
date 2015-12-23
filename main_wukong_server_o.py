import SocketServer
import json
import sys
sys.path.append('lib')
import os
import numpy as np

import socket as sk
import serial

HOST = ''
PORT = 4070
usb_port='/dev/ttyUSB0'
usb_ser=serial.Serial(usb_port,9600)
state_pool=[0]*4

class UDPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        global state_pool
        json_data = self.request[0]
        raw_data = json.loads(json_data)
        device_num=raw_data['device']
        state_value=raw_data['state']
        print device_num, state_value
        state_str=''
        if (int(device_num)==3) and (int(state_value)!=0):
            state_pool=change_state(3,-1)
        else:
            if (int(device_num)==0) or (state_pool[0]!=-2):
                state_pool=change_state(int(device_num),int(state_value))
        if (state_pool[0]!=-2):
            for ind in xrange(len(state_pool)):
                state_str=state_str+str(state_pool[ind])
                if not(ind==(len(state_pool)-1)):
                    state_str=state_str+','
            print '##'+state_str+'@@\n'
            usb_ser.write('##'+state_str+'@@\n')
        self.request[1].sendto('send OK', self.client_address)

def change_state(target_index,value):
    global state_pool
    state_pool[target_index]=value
    return state_pool

def start_server():
    print 'current ip address: ' + sk.gethostbyname(sk.gethostname()) + ':' + str(PORT)
    server = SocketServer.UDPServer((HOST, PORT), UDPHandler)
    server.serve_forever()


if __name__ == '__main__':
    start_server()
