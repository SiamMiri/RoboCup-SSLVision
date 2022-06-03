import socket 
import struct
from threading import Thread

class UDP_Connection(Thread):
    GROUP = "224.1.1.1"
    PORT  = 5003
    
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
    
    def send(self, payload:bytes= b''):
        self.socket_ssl.setsockopt(socket.IPPROTO_IP,
                                   socket.IP_MULTICAST_TTL,
                                   UDP_Connection.TTL)
        self.socket_ssl.sendto(payload, (UDP_Connection.GROUP, UDP_Connection.PORT))

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
    
co = UDP_Connection()
co.send(payload=bytearray(14564))
co.receive_thread()