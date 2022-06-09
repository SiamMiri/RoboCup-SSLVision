from UDPSockets_SSLClient_ProtoBuf import ssl_client_pb2 as SSL_RoboCup_ProtoBuf

class Client_Socket_ProtoBuf():
    def __init__(self, *args):
        self.ProtoBuf = None
        if len(args[0]) == 8:       
            self.ssl_robot_detection_message(args[0])
            
    
    def __del__(self):
        pass
    
    def ssl_robot_detection_message(self, *args):        
        message_dic = SSL_RoboCup_ProtoBuf.SSL_DetectionRobot()
        message_dic.confidence  = args[0]["confidence"]
        message_dic.robot_id    = args[0]["robot_id"]
        message_dic.x           = args[0]["x"]
        message_dic.y           = args[0]["y"]
        message_dic.orientation = args[0]["orientation"]
        message_dic.pixel_x     = args[0]["pixel_x"]
        message_dic.pixel_y     = args[0]["pixel_y"]
        message_dic.height      = args[0]["height"]        
        self.ProtoBuf =  message_dic

