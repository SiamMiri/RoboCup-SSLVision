import socket 
import struct
from threading import Thread
from UDPSockets_SSLClient_ProtoBuf.ClientSocketProtoBuf import Client_Socket_ProtoBuf
from google.protobuf.internal.encoder import _VarintEncoder
from google.protobuf.internal.decoder import _DecodeVarint

class UDP_Connection():
    GROUP = "224.1.1.1"
    PORT  = 5003
    ADDRESS = (GROUP, PORT)
    # https://en.wikipedia.org/wiki/Hop_(networking)#:~:text=bridges)%20are%20not.-,Hop%20limit,TTL%20or%20hop%20limit%20fields.
    '''Known as time to live (TTL) in IPv4, 
    and hop limit in IPv6, this field specifies
    a limit on the number of hops a packet is allowed
    before being discarded. Routers modify IP packets as 
    they are forwarded, decrementing the respective
    TTL or hop limit fields. Routers do not 
    forward packets with a resultant field of 0 or less.
    This prevents packets from following a loop forever.'''
    
    TTL = 2 # Hop restriction in network
    
    def __init__(self):

        self.socket_ssl = socket.socket(socket.AF_INET,
                                  socket.SOCK_DGRAM,
                                  socket.IPPROTO_UDP)
        
    
    def __del__(self):
        pass
    
    def send(self, payload:dict= {}):
        message = self.convert_data_to_ProtoBuf_format(payload)
        
        self.socket_ssl.setsockopt(socket.IPPROTO_IP,
                                   socket.IP_MULTICAST_TTL,
                                   UDP_Connection.TTL)
        self.socket_ssl.connect(UDP_Connection.ADDRESS)
        s = message.SerializeToString()

        totallen = 4 + len(s) 
        pack1 = struct.pack('>I', totallen) # the first part of the message is length
    
        self.socket_ssl.sendall(pack1 + s)
        # self.socket_ssl.sendto(message, (UDP_Connection.GROUP, UDP_Connection.PORT))

    def convert_data_to_ProtoBuf_format(self, message:dict={}):
        ProtoBuf_Message = Client_Socket_ProtoBuf(message)
        return ProtoBuf_Message.ProtoBuf
    def receive(self):
        self.socket_ssl.setsockopt(socket.SOL_SOCKET,
                                   socket.SO_REUSEADDR,
                                   1)
        self.socket_ssl.bind(("", UDP_Connection.PORT))
        mreq = struct.pack("4sl", socket.inet_aton(UDP_Connection.GROUP), socket.INADDR_ANY)
        self.socket_ssl.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        while True:
            print(self.socket_ssl.recv(10240))  
            
    def run(self):
        self.socket_ssl.setsockopt(socket.SOL_SOCKET,
                                   socket.SO_REUSEADDR,
                                   1)
        self.socket_ssl.bind(("", UDP_Connection.PORT))
        mreq = struct.pack("4sl", socket.inet_aton(UDP_Connection.GROUP), socket.INADDR_ANY)
        self.socket_ssl.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        while True:
            print(self.socket_ssl.recv(10240))
    
    def send_thread(self, payload:bytes=b''):
        pass
    
    def receive_thread(self):
        self.run()