# import ImageProcessing
import math
import numpy as np
from ImageProcessing.ImageProcessing import Image_Processing
from UDPSockets_SSLClient_ProtoBuf.UDPConnection import UDP_Connection
import time 

class Detect_Robot_Ball():
    
    ROTATE_ROBAT_SINGLE_IMAGE = False
    Robot_Pattern_Dict= { "1"  : {"TOP_RIGHT": 'red'  , "TOP_LEFT": 'red'  , "DOWN_LEFT": 'green', "DOWN_RIGHT": 'red'},
                          "2"  : {"TOP_RIGHT": 'red'  , "TOP_LEFT": 'green', "DOWN_LEFT": 'green', "DOWN_RIGHT": 'red'},
                          "3"  : {"TOP_RIGHT": 'green', "TOP_LEFT": 'green', "DOWN_LEFT": 'green', "DOWN_RIGHT": 'red'},
                          "4"  : {"TOP_RIGHT": 'green', "TOP_LEFT": 'red'  , "DOWN_LEFT": 'green', "DOWN_RIGHT": 'red'},
                          "5"  : {"TOP_RIGHT": 'red'  , "TOP_LEFT": 'red'  , "DOWN_LEFT": 'red'  , "DOWN_RIGHT": 'green'},
                          "6"  : {"TOP_RIGHT": 'red'  , "TOP_LEFT": 'green', "DOWN_LEFT": 'red'  , "DOWN_RIGHT": 'green'},
                          "7"  : {"TOP_RIGHT": 'green', "TOP_LEFT": 'green', "DOWN_LEFT": 'red'  , "DOWN_RIGHT": 'green'},
                          "8"  : {"TOP_RIGHT": 'green', "TOP_LEFT": 'red'  , "DOWN_LEFT": 'red'  , "DOWN_RIGHT": 'green'},
                          "9"  : {"TOP_RIGHT": 'green', "TOP_LEFT": 'green', "DOWN_LEFT": 'green', "DOWN_RIGHT": 'green'},
                          "10" : {"TOP_RIGHT": 'red'  , "TOP_LEFT": 'red'  , "DOWN_LEFT": 'red'  , "DOWN_RIGHT": 'red'  },
                          "11" : {"TOP_RIGHT": 'red'  , "TOP_LEFT": 'red'  , "DOWN_LEFT": 'green', "DOWN_RIGHT": 'green'},
                          "12" : {"TOP_RIGHT": 'green', "TOP_LEFT": 'green', "DOWN_LEFT": 'red'  , "DOWN_RIGHT": 'red'  },
                          "13" : {"TOP_RIGHT": 'red'  , "TOP_LEFT": 'green', "DOWN_LEFT": 'green', "DOWN_RIGHT": 'green'},
                          "14" : {"TOP_RIGHT": 'red'  , "TOP_LEFT": 'green', "DOWN_LEFT": 'red'  , "DOWN_RIGHT": 'red'  },
                          "15" : {"TOP_RIGHT": 'green', "TOP_LEFT": 'red'  , "DOWN_LEFT": 'green', "DOWN_RIGHT": 'green'},
                          "16" : {"TOP_RIGHT": 'green', "TOP_LEFT": 'red'  , "DOWN_LEFT": 'red'  , "DOWN_RIGHT": 'red'  }}
    PRINT_DEBUG  = False
    SEND_DATA_TO_SERVER = False
    
    def __init__(self) -> None: #, package_color_pixel_position:list = None
        Image_Processing.ROTATE_ROBOT_IMAGE = Detect_Robot_Ball.ROTATE_ROBAT_SINGLE_IMAGE
        self.processImage = Image_Processing(self)
        self.upd_connection = UDP_Connection()

        # object global variable
        self.pack                           = None # package_color_pixel_position
        self.shortest_line_pixel_pos_list   = None
        self.crop_robot_image               = None
        self.ROBOT_ID                       = 0
        self.startTime                      = 0
        self.endTime                     = 0
        
    def __del__(self):
        pass
    
    def detect_robot(self, frame : np.array = None):
        self.startTime = time.time()
        if frame is not None:
            self.processImage._start_process(field_frame = frame)
          
    def list_min_dist_between_circle(self, circle_pixel_pos_pack:list = None ):
        length_list = {}
        if circle_pixel_pos_pack is not None:
            if 'prime' in circle_pixel_pos_pack.keys():
                circle_keys = list(circle_pixel_pos_pack.keys())
                # circle_keys = 1
                print(circle_pixel_pos_pack.keys())
                length_list.update({ f"{circle_keys[0]}__{circle_keys[1]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[0]][0] - circle_pixel_pos_pack[circle_keys[1]][0])**2 + (circle_pixel_pos_pack[circle_keys[0]][1] - circle_pixel_pos_pack[circle_keys[1]][1])**2) })
                length_list.update({ f"{circle_keys[0]}__{circle_keys[2]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[0]][0] - circle_pixel_pos_pack[circle_keys[2]][0])**2 + (circle_pixel_pos_pack[circle_keys[0]][1] - circle_pixel_pos_pack[circle_keys[2]][1])**2) })
                length_list.update({ f"{circle_keys[0]}__{circle_keys[3]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[0]][0] - circle_pixel_pos_pack[circle_keys[3]][0])**2 + (circle_pixel_pos_pack[circle_keys[0]][1] - circle_pixel_pos_pack[circle_keys[3]][1])**2) })
                length_list.update({ f"{circle_keys[1]}__{circle_keys[2]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[1]][0] - circle_pixel_pos_pack[circle_keys[2]][0])**2 + (circle_pixel_pos_pack[circle_keys[1]][1] - circle_pixel_pos_pack[circle_keys[2]][1])**2) })
                length_list.update({ f"{circle_keys[1]}__{circle_keys[3]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[1]][0] - circle_pixel_pos_pack[circle_keys[3]][0])**2 + (circle_pixel_pos_pack[circle_keys[1]][1] - circle_pixel_pos_pack[circle_keys[3]][1])**2) })
                length_list.update({ f"{circle_keys[2]}__{circle_keys[3]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[2]][0] - circle_pixel_pos_pack[circle_keys[3]][0])**2 + (circle_pixel_pos_pack[circle_keys[2]][1] - circle_pixel_pos_pack[circle_keys[3]][1])**2) })
                if Detect_Robot_Ball.PRINT_DEBUG:
                    print("##############################")
                    print(f'length_all:   {length_list}')
                    print("##############################")
            else:
                length_list.update({ "TOP_RIGHT__TOP_LEFT"      : math.sqrt((circle_pixel_pos_pack["TOP_RIGHT"][0] - circle_pixel_pos_pack["TOP_LEFT"][0])**2   + (circle_pixel_pos_pack["TOP_RIGHT"][1] - circle_pixel_pos_pack["TOP_LEFT"][1])**2) })
                length_list.update({ "TOP_RIGHT__DOWN_LEFT"     : math.sqrt((circle_pixel_pos_pack["TOP_RIGHT"][0] - circle_pixel_pos_pack["DOWN_LEFT"][0])**2  + (circle_pixel_pos_pack["TOP_RIGHT"][1] - circle_pixel_pos_pack["DOWN_LEFT"][1])**2)})
                length_list.update({ "TOP_RIGHT__DOWN_RIGHT"    : math.sqrt((circle_pixel_pos_pack["TOP_RIGHT"][0] - circle_pixel_pos_pack["DOWN_RIGHT"][0])**2 + (circle_pixel_pos_pack["TOP_RIGHT"][1] - circle_pixel_pos_pack["DOWN_RIGHT"][1])**2)})
                length_list.update({ "TOP_LEFT__DOWN_LEFT"      : math.sqrt((circle_pixel_pos_pack["TOP_LEFT"][0]  - circle_pixel_pos_pack["DOWN_LEFT"][0])**2  + (circle_pixel_pos_pack["TOP_LEFT"][1]  - circle_pixel_pos_pack["DOWN_LEFT"][1])**2)})
                length_list.update({ "TOP_LEFT__DOWN_RIGHT"     : math.sqrt((circle_pixel_pos_pack["TOP_LEFT"][0]  - circle_pixel_pos_pack["DOWN_RIGHT"][0])**2 + (circle_pixel_pos_pack["TOP_LEFT"][1]  - circle_pixel_pos_pack["DOWN_RIGHT"][1])**2)})
                length_list.update({ "DOWN_LEFT__DOWN_RIGHT"    : math.sqrt((circle_pixel_pos_pack["DOWN_LEFT"][0] - circle_pixel_pos_pack["DOWN_RIGHT"][0])**2 + (circle_pixel_pos_pack["DOWN_LEFT"][1] - circle_pixel_pos_pack["DOWN_RIGHT"][1])**2)})
                if Detect_Robot_Ball.PRINT_DEBUG:
                    print("##############################")
                    print(f'len_one:   {length_list["TOP_RIGHT__TOP_LEFT"]}')
                    print(f'len_two:   {length_list["TOP_RIGHT__DOWN_LEFT"]}')
                    print(f'len_three: {length_list["TOP_RIGHT__DOWN_RIGHT"]}')
                    print(f'len_four:  {length_list["TOP_LEFT__DOWN_LEFT"]}')
                    print(f'len_five:  {length_list["TOP_LEFT__DOWN_RIGHT"]}')
                    print(f'len_six:   {length_list["DOWN_LEFT__DOWN_RIGHT"]}')
                    print("##############################")
        else:
            print("circle_pixel_pos_pack is Empty")
        
        min_len = min(length_list, key=length_list.get)
        length_list = sorted(length_list)

        if Detect_Robot_Ball.PRINT_DEBUG:
            print(f"The min_len is {min_len}")
        
        
        if 'prime' in circle_pixel_pos_pack.keys():
            min_len = min_len.split("__")
            self.shortest_line_pixel_pos_list = [min_len[0], min_len[1]]
        else:
            min_len = min_len.split("__")
            self.shortest_line_pixel_pos_list = [min_len[0], min_len[1]]
        
        if Detect_Robot_Ball.PRINT_DEBUG:
            print(f"The min length of line is {self.shortest_line_pixel_pos_list}")
            
        angle = self.angle_between_shortest_line_circles(self.shortest_line_pixel_pos_list, circle_pixel_pos_pack)
        print(f'The final Angle is : {angle}')
        self.ROBOT_ID  += 1
        # self.show_robot()
        return angle
    
    def angle_between_shortest_line_circles(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        ppcsl = pixel_position_circle_shortest_line[0] + "__" + pixel_position_circle_shortest_line[1]
        # print(f'ppcsl: {ppcsl}')
        if self.check_circle_position(listCirclePosiotn=pixel_position_circle_shortest_line):
            
            if ppcsl == "TOP_LEFT__TOP_RIGHT" or ppcsl == "TOP_RIGHT__TOP_LEFT":
                ''' first assumption the position of the circles are in top right and top left'''
                return self.angle_circle_position_topRight_topLeft(pixel_position_circle_shortest_line = pixel_position_circle_shortest_line,
                                                                    pixel_position_value_shortest_line  = pixel_position_value_shortest_line)
            
            elif ppcsl == "TOP_LEFT__DOWN_LEFT" or ppcsl == "DOWN_LEFT__TOP_LEFT":
                ''' second assumption '''
                return self.angle_circle_position_topLeft_downLeft(pixel_position_circle_shortest_line = pixel_position_circle_shortest_line,
                                                                    pixel_position_value_shortest_line  = pixel_position_value_shortest_line)
            
            elif ppcsl == "DOWN_RIGHT__DOWN_LEFT" or ppcsl == "DOWN_LEFT__DOWN_RIGHT":
                ''' third assumption '''
                return self.angle_circle_position_downLeft_downRight(pixel_position_circle_shortest_line = pixel_position_circle_shortest_line,
                                                                    pixel_position_value_shortest_line  = pixel_position_value_shortest_line)
            
            elif ppcsl == "TOP_RIGHT__DOWN_RIGHT" or ppcsl == "DOWN_RIGHT__TOP_RIGHT":      
                ''' fourth assumption '''
                return self.angle_circle_position_topRight_downRight(pixel_position_circle_shortest_line = pixel_position_circle_shortest_line,
                                                                pixel_position_value_shortest_line  = pixel_position_value_shortest_line)
                
            elif ppcsl == "TOP_RIGHT__prime":
                ''' fifth assumption '''
                return self.angle_circle_position_topRight_topRight(pixel_position_circle_shortest_line = pixel_position_circle_shortest_line,
                                                                pixel_position_value_shortest_line  = pixel_position_value_shortest_line) 
                
            elif ppcsl == "TOP_LEFT__prime":
                ''' sixth assumption '''
                return self.angle_circle_position_topLeft_topLeft(pixel_position_circle_shortest_line = pixel_position_circle_shortest_line,
                                                                pixel_position_value_shortest_line  = pixel_position_value_shortest_line)
            
            elif ppcsl == "DOWN_LEFT__prime":
                ''' seventh assumption '''
                return self.angle_circle_position_downLeft_downLeft(pixel_position_circle_shortest_line = pixel_position_circle_shortest_line,
                                                                pixel_position_value_shortest_line  = pixel_position_value_shortest_line)
            elif ppcsl == "DOWN_RIGHT__prime": 
                ''' eighth assumption '''
                return  self.angle_circle_position_downRight_downRight(pixel_position_circle_shortest_line = pixel_position_circle_shortest_line,
                                                                    pixel_position_value_shortest_line  = pixel_position_value_shortest_line)
            else:
                return None 
        else:  
            return None
        
    def check_circle_position(self, listCirclePosiotn):
        ''' Durchmesser ist Ungültig '''
        if listCirclePosiotn[0]     == "TOP_RIGHT"  and listCirclePosiotn[0] == "DOWN_LEFT":
            return False
        elif listCirclePosiotn[0]   == "DOWN_LEFT"  and listCirclePosiotn[0] == "TOP_RIGHT":
            return False
        elif listCirclePosiotn[0]   == "TOP_LEFT"   and listCirclePosiotn[0] == "DOWN_RIGHT":
            return False
        elif listCirclePosiotn[0]   == "DOWN_RIGHT" and listCirclePosiotn[0] == "TOP_LEFT":
            return False
        else:
            return True
        
    def angle_circle_position_topRight_topLeft(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        
        Angle = None
        
        if pixel_position_circle_shortest_line[0] == "TOP_RIGHT" and pixel_position_circle_shortest_line[1] == "TOP_LEFT":
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["TOP_LEFT"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] > pixel_position_value_shortest_line["TOP_LEFT"][1]:
                    print("1 is right and down")
                    print("2 is left and up")
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]])
                    Angle =  ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle
                else:
                    print("1 is right and up")
                    print("2 is left and down")
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]])
                    Angle =  ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle

            else:
                print("THIS CONFIG IS NOT POSSIBLE !!!")

        if pixel_position_circle_shortest_line[1] == "TOP_RIGHT" and pixel_position_circle_shortest_line[0] == "TOP_LEFT":
            print("first assumption 2")
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["TOP_LEFT"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] > pixel_position_value_shortest_line["TOP_LEFT"][1]:
                    print("1 is right and down")
                    print("2 is left and up")
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]])
                    Angle =  ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle
                else:
                    print("1 is right and up")
                    print("2 is left and down")
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]])
                    Angle =  ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle
            else:
                print("THIS CONFIG IS NOT POSSIBLE !!!") 
        
        return Angle
                
    def angle_circle_position_topLeft_downLeft(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        if pixel_position_circle_shortest_line[0] == "DOWN_LEFT" and pixel_position_circle_shortest_line[1] == "TOP_LEFT":
            print("second assumption 1")
            if pixel_position_value_shortest_line["TOP_LEFT"][0] < pixel_position_value_shortest_line["DOWN_LEFT"][0]:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["DOWN_LEFT"][1]:
                    print("2 is up and back")
                    print("3 is down and frot")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                    
                    
                else:
                    print("2 is down and back")
                    print("3 is up and front")
                    print("THIS CONFIG IS NOT POSSIBLE !!!")
                    
            else:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["DOWN_LEFT"][1]:
                    print("2 is up and front")
                    print("3 is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]])
                    Angle = 90 - Angle + 90
                    print(Angle)
                    return Angle
                

        if pixel_position_circle_shortest_line[1] == "DOWN_LEFT" and pixel_position_circle_shortest_line[0] == "TOP_LEFT":
            print("second assumption 2")
            if pixel_position_value_shortest_line["TOP_LEFT"][0] < pixel_position_value_shortest_line["DOWN_LEFT"][0]:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["DOWN_LEFT"][1]:
                    print("2 is up and back")
                    print("3 is down and frot")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                    
                    
                else:
                    print("2 is down and back")
                    print("3 is up and front")
                    print("THIS CONFIG IS NOT POSSIBLE !!!")
                    
            else:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["DOWN_LEFT"][1]:
                    print("2 is up and front")
                    print("3 is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]])
                    Angle = 90 - Angle + 90
                    print(Angle)
                    return Angle
    
    def angle_circle_position_downLeft_downRight(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        if pixel_position_circle_shortest_line[0] == "DOWN_LEFT" and pixel_position_circle_shortest_line[1] == "DOWN_RIGHT":
            print("third assumption 1")
            if pixel_position_value_shortest_line["DOWN_LEFT"][0] < pixel_position_value_shortest_line["DOWN_RIGHT"][0]:
                if pixel_position_value_shortest_line["DOWN_LEFT"][1] < pixel_position_value_shortest_line["DOWN_RIGHT"][1]:
                    print("3 is up and back")
                    print("4 is down and front")
                    # TODO: Check the 10 degree error
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                else:
                    print("3 is down and back")
                    print("4 is up and front")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
            else:
                print("THIS IS NOT POSSIBLE!!")

        if pixel_position_circle_shortest_line[1] == "DOWN_LEFT" and pixel_position_circle_shortest_line[0] == "DOWN_RIGHT":
            print("third assumption 2")
            if pixel_position_value_shortest_line["DOWN_LEFT"][0] < pixel_position_value_shortest_line["DOWN_RIGHT"][0]:
                if pixel_position_value_shortest_line["DOWN_LEFT"][1] < pixel_position_value_shortest_line["DOWN_RIGHT"][1]:
                    print("3 is up and back")
                    print("4 is down and front")
                    # TODO: Check the 10 degree error
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                else:
                    print("3 is down and back")
                    print("4 is up and front")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
            else:
                print("THIS IS NOT POSSIBLE!!")
    
    def angle_circle_position_topRight_downRight(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        if pixel_position_circle_shortest_line[0] == "DOWN_RIGHT"  and pixel_position_circle_shortest_line[1] == "TOP_RIGHT":
            print("fourth assumption 1")
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["DOWN_RIGHT"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] < pixel_position_value_shortest_line["DOWN_RIGHT"][1]:
                    print("1 is up and front")
                    print("4 is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("1 is up and back")
                print("4 is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                Angle = ( Angle + 90 + 90 ) * -1
                print(Angle)
                return Angle
        
        if pixel_position_circle_shortest_line[1] == "DOWN_RIGHT" and pixel_position_circle_shortest_line[0] == "TOP_RIGHT":
            print("fourth assumption 2")
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["DOWN_RIGHT"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] < pixel_position_value_shortest_line["DOWN_RIGHT"][1]:
                    print("1 is up and front")
                    print("4 is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("1 is up and back")
                print("4 is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                Angle = ( Angle + 90 + 90 ) * -1
                print(Angle)
                return Angle
            
    def angle_circle_position_topRight_topRight(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        if pixel_position_circle_shortest_line[0] == "TOP_RIGHT"  and pixel_position_circle_shortest_line[1] == "prime":
            print("fifth assumption 1")
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] > pixel_position_value_shortest_line["prime"][1]:
                    print("1 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("1 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = ( 180 + Angle ) * -1
                print(Angle)
                return Angle
            
        if pixel_position_circle_shortest_line[0] == "prime"  and pixel_position_circle_shortest_line[1] == "TOP_RIGHT":
            print("fifth assumption 2")
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] > pixel_position_value_shortest_line["prime"][1]:
                    print("1 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("1 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = ( 180 + Angle ) * -1
                print(Angle)
                return Angle
    
    def angle_circle_position_topLeft_topLeft(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        if pixel_position_circle_shortest_line[0] == "TOP_LEFT"  and pixel_position_circle_shortest_line[1] == "prime":
            print("seventh assumption 1")
            if pixel_position_value_shortest_line["TOP_LEFT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    print("2 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = ( 90 - Angle ) + 90
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("2 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = ( 90 - Angle ) + 90
                print(Angle)
                return Angle
            
        if pixel_position_circle_shortest_line[0] == "prime"  and pixel_position_circle_shortest_line[1] == "TOP_LEFT":
            print("seventh assumption 2")
            if pixel_position_value_shortest_line["TOP_LEFT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    print("2 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = ( 90 - Angle ) + 90
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("2 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = ( 90 - Angle ) + 90
                print(Angle)
                return Angle
    
    def angle_circle_position_downLeft_downLeft(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        if pixel_position_circle_shortest_line[0] == "DOWN_LEFT"  and pixel_position_circle_shortest_line[1] == "prime":
            print("Eighth assumption 1")
            if pixel_position_value_shortest_line["DOWN_LEFT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["DOWN_LEFT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    print("3 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]])
                    Angle = Angle * -1 
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("3 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                # Angle = ( 90 + Angle ) + 90
                Angle = Angle * -1 
                print(Angle)
                return Angle
            
        if pixel_position_circle_shortest_line[0] == "prime"  and pixel_position_circle_shortest_line[1] == "DOWN_LEFT":
            print("Eighth assumption 2")
            if pixel_position_value_shortest_line["DOWN_LEFT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["DOWN_LEFT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    print("3 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = Angle * -1 
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("3 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = Angle * -1 
                print(Angle)
                return Angle
    
    def angle_circle_position_downRight_downRight(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        if pixel_position_circle_shortest_line[0] == "DOWN_RIGHT"  and pixel_position_circle_shortest_line[1] == "prime":
            print("Ninth assumption 1")
            if pixel_position_value_shortest_line["DOWN_RIGHT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["DOWN_RIGHT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    print("4 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = Angle * -1 
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("4 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                # Angle = ( 90 + Angle ) + 90
                Angle = Angle * -1 
                print(Angle)
                return Angle
            
        if pixel_position_circle_shortest_line[0] == "prime"  and pixel_position_circle_shortest_line[1] == "DOWN_RIGHT":
            print("Ninth assumption 2")
            if pixel_position_value_shortest_line["DOWN_RIGHT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["DOWN_RIGHT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    print("4 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = Angle * -1 
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("4 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = Angle * -1 
                print(Angle)
                return Angle

    def gradient(self, pt1 ,pt2):
        return(pt2[1] - pt1[1])/ (pt2[0] - pt1[0]) 
    
    def getAngle(self, PT1 = None, PT2 = None, PT3 = None):
        pt1 = PT1
        pt2 = PT2
        pt3 = PT3
        n1 = self.gradient(pt1 , pt2)
        n2 = self.gradient(pt1 ,pt3)
        angleR = math.atan((n2 - n1)/(1 + (n2*n1)))
        angleD = round(math.degrees(angleR))
        return angleD
    
    def match_robot(self, frameRobot:np.array = None):
        
        circle_pixel_pos_pack = {"TOP_RIGHT":    "",
                                 "TOP_LEFT":     "",
                                 "DOWN_LEFT":    "",
                                 "DOWN_RIGHT":   ""}
               
        contours_red , mask   = self.processImage.find_contours_mask(frame=frameRobot, circle_color="Masked_Red")
        contours_green , mask = self.processImage.find_contours_mask(frame=frameRobot, circle_color="Masked_Green")
        are_of_circle_min, are_of_circle_max = self.processImage._calculate_area_of_circle(frame=frameRobot,circle_color="Red")
        
        list_circle_cordinate = []             
        
        """ contours for red area  """            
        for red_contour in contours_red:
            red_area = self.processImage.calculate_contours_area(contours=red_contour)
            if red_area < are_of_circle_max and red_area > are_of_circle_min:
                moment = self.processImage.calculate_moment(contours=red_contour)
                cx_red = int(moment["m10"]/moment["m00"])
                cy_red = int(moment["m01"]/moment["m00"])
                if cx_red > frameRobot.shape[0]/2 and cy_red < frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["TOP_RIGHT"] = "red"
                if cx_red < frameRobot.shape[0]/2 and cy_red < frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["TOP_LEFT"] = "red"
                if cx_red < frameRobot.shape[0]/2 and cy_red > frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["DOWN_LEFT"] = "red"
                if cx_red > frameRobot.shape[0]/2 and cy_red > frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["DOWN_RIGHT"] = "red"

        list_circle_cordinate.clear()
        """ contours for green area """             
        for green_contour in contours_green:
            green_area = self.processImage.calculate_contours_area(contours=green_contour)
            if green_area < are_of_circle_max and green_area > are_of_circle_min:
                moment = self.processImage.calculate_moment(contours=green_contour)
                print(f"green_area {green_area}")
                cx_green = int(moment["m10"]/moment["m00"])
                cy_green = int(moment["m01"]/moment["m00"])
                if cx_green > frameRobot.shape[0]/2 and cy_green < frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["TOP_RIGHT"] = "green"
                if cx_green < frameRobot.shape[0]/2 and cy_green < frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["TOP_LEFT"] = "green"
                if cx_green < frameRobot.shape[0]/2 and cy_green > frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["DOWN_LEFT"] = "green"
                if cx_green > frameRobot.shape[0]/2 and cy_green > frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["DOWN_RIGHT"] = "green"

        return self.loop_robot_id_list(color_pattern_list = circle_pixel_pos_pack)
    
    def loop_robot_id_list(self, color_pattern_list: dict = None):
        for Roboid in Detect_Robot_Ball.Robot_Pattern_Dict:
            if Detect_Robot_Ball.Robot_Pattern_Dict[Roboid] == color_pattern_list:
                return Roboid
        if Detect_Robot_Ball.PRINT_DEBUG:       
            print(f"IT IS NOT ROBOT PATERN, PATERN: {color_pattern_list}")
            
    def send_data_to_server(self, ssl_message:dict=None):
        if Detect_Robot_Ball.SEND_DATA_TO_SERVER:
            self.endTime = time.time()
            if Detect_Robot_Ball.PRINT_DEBUG:
                print(f"Data is sending. time takes: {self.endTime - self.startTime}")
                print(f"Data SSL is {ssl_message}")
            udp__connection = UDP_Connection()
            udp__connection.send(payload=ssl_message)