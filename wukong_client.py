import socket
import json


def send(device,state):
    HOST, PORT = '127.0.0.1', 4075
    
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Connect to server and send data
    json_obj=json.dumps({'device': device, 'state': state})
    
    s.connect((HOST, PORT))
    s.sendall(json_obj + "\n")
    try:
        received = s.recv(1024)
        print received
    except:
        print 'send fail...'
    s.close()
    
if __name__ == '__main__':
    send(1,-1)