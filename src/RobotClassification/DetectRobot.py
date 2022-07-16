# Author: Siamak Mirifar

from asyncio.log import logger
import math
import numpy as np
import time

class Detect_Robot():
    
    # Dic List of the robots ID
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
    
    PRINT_DEBUG         = False
    
    def __init__(self, FuncRotateImage, FuncFindContours, FuncCalContoursArea, FuncCalMoment) -> None: #,CropImageQueue package_color_pixel_position:list = None
        """_summary_

        Args:
            FuncRotateImage (_type_): _description_ 
            Method for rotating image from Image Processing class
            
            FuncFindContours (_type_): _description_
            Method to define contours of the cropped image from image processing class
            
            CircleArea (_type_): _description_
            Maximum and Minimum of robot id circle
            
            FuncCalContoursArea (_type_): _description_
            Method to calculate the concourse area
            
            FuncCalMoment (_type_): _description_
            Method for finding Moment of the contours
        """        

        # object global variable
        self.shortest_line_pixel_pos_list   = None  # shortest line between circle's id
        
        # External Function
        self.FuncRotateImage                = FuncRotateImage
        self.FuncFindContours               = FuncFindContours
        self.FuncCalContoursArea            = FuncCalContoursArea
        self.FuncCalMoment                  = FuncCalMoment

        # External Variable 
        self.CircleArea                     = None
        self.bInitCircleArea                = False # onetime set circle area
            
    def detect_robot(self, frame_data : dict, CircleArea:list=None ):# , dataQueue 
        """_summary_

        Args:
            frame_data (dict): _description_: 
            Imported frame data from image processing
            
            CircleArea (list, optional): _description_. Defaults to None.
            CircleArea imported from Image Processing class

        Returns:
            _type_: _description_:
            return robots id and 
        """
        if self.bInitCircleArea != True: # Define circle area just once
            self.CircleArea = CircleArea
            self.bInitCircleArea = True
            
        ssl_list_queue = {} # Dic of robots data
        if frame_data != None:
            for circlePack in frame_data:
                # Get Robots Angle
                Angle       = self.list_min_dist_between_circle(circle_pixel_pos_pack = frame_data[circlePack][3])
                
                # Rotate Robots Cropped Image
                rotateImg   = self.FuncRotateImage(frame_data[circlePack][2], degree=Angle)
                
                # Set Robot Id
                RobotId     = self.match_robot(frameRobot = rotateImg)
                
                # Update Robot list
                ssl_list_queue[circlePack] = [RobotId, Angle, frame_data[circlePack][0], frame_data[circlePack][1]]
        else:
            return ssl_list_queue
        
        return ssl_list_queue
        
    def list_min_dist_between_circle(self, circle_pixel_pos_pack:list = None ):
        """_summary_
        This method calculate minimum len between circles to calculate angle
        Args:
            circle_pixel_pos_pack (list, optional): _description_. Defaults to None.
            dic of all color and their position and the pixel position value
        Returns:
            _type_: _description_:
            return Angle
        """        
        length_list = {}
        if circle_pixel_pos_pack is not None:
            if 'prime' in circle_pixel_pos_pack.keys(): # if we have position the calculation would be as follow
                circle_keys = list(circle_pixel_pos_pack.keys())
                # circle_keys = 1
                length_list.update({ f"{circle_keys[0]}__{circle_keys[1]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[0]][0] - circle_pixel_pos_pack[circle_keys[1]][0])**2 + (circle_pixel_pos_pack[circle_keys[0]][1] - circle_pixel_pos_pack[circle_keys[1]][1])**2) })
                length_list.update({ f"{circle_keys[0]}__{circle_keys[2]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[0]][0] - circle_pixel_pos_pack[circle_keys[2]][0])**2 + (circle_pixel_pos_pack[circle_keys[0]][1] - circle_pixel_pos_pack[circle_keys[2]][1])**2) })
                length_list.update({ f"{circle_keys[0]}__{circle_keys[3]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[0]][0] - circle_pixel_pos_pack[circle_keys[3]][0])**2 + (circle_pixel_pos_pack[circle_keys[0]][1] - circle_pixel_pos_pack[circle_keys[3]][1])**2) })
                length_list.update({ f"{circle_keys[1]}__{circle_keys[2]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[1]][0] - circle_pixel_pos_pack[circle_keys[2]][0])**2 + (circle_pixel_pos_pack[circle_keys[1]][1] - circle_pixel_pos_pack[circle_keys[2]][1])**2) })
                length_list.update({ f"{circle_keys[1]}__{circle_keys[3]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[1]][0] - circle_pixel_pos_pack[circle_keys[3]][0])**2 + (circle_pixel_pos_pack[circle_keys[1]][1] - circle_pixel_pos_pack[circle_keys[3]][1])**2) })
                length_list.update({ f"{circle_keys[2]}__{circle_keys[3]}" : math.sqrt((circle_pixel_pos_pack[circle_keys[2]][0] - circle_pixel_pos_pack[circle_keys[3]][0])**2 + (circle_pixel_pos_pack[circle_keys[2]][1] - circle_pixel_pos_pack[circle_keys[3]][1])**2) })
                if Detect_Robot.PRINT_DEBUG:
                    print("##############################")
                    print(f'length_all:   {length_list}')
                    print("##############################")
            else: # find minimum length in list without prime position
                length_list.update({ "TOP_RIGHT__TOP_LEFT"      : math.sqrt((circle_pixel_pos_pack["TOP_RIGHT"][0] - circle_pixel_pos_pack["TOP_LEFT"][0])**2   + (circle_pixel_pos_pack["TOP_RIGHT"][1] - circle_pixel_pos_pack["TOP_LEFT"][1])**2) })
                length_list.update({ "TOP_RIGHT__DOWN_LEFT"     : math.sqrt((circle_pixel_pos_pack["TOP_RIGHT"][0] - circle_pixel_pos_pack["DOWN_LEFT"][0])**2  + (circle_pixel_pos_pack["TOP_RIGHT"][1] - circle_pixel_pos_pack["DOWN_LEFT"][1])**2)})
                length_list.update({ "TOP_RIGHT__DOWN_RIGHT"    : math.sqrt((circle_pixel_pos_pack["TOP_RIGHT"][0] - circle_pixel_pos_pack["DOWN_RIGHT"][0])**2 + (circle_pixel_pos_pack["TOP_RIGHT"][1] - circle_pixel_pos_pack["DOWN_RIGHT"][1])**2)})
                length_list.update({ "TOP_LEFT__DOWN_LEFT"      : math.sqrt((circle_pixel_pos_pack["TOP_LEFT"][0]  - circle_pixel_pos_pack["DOWN_LEFT"][0])**2  + (circle_pixel_pos_pack["TOP_LEFT"][1]  - circle_pixel_pos_pack["DOWN_LEFT"][1])**2)})
                length_list.update({ "TOP_LEFT__DOWN_RIGHT"     : math.sqrt((circle_pixel_pos_pack["TOP_LEFT"][0]  - circle_pixel_pos_pack["DOWN_RIGHT"][0])**2 + (circle_pixel_pos_pack["TOP_LEFT"][1]  - circle_pixel_pos_pack["DOWN_RIGHT"][1])**2)})
                length_list.update({ "DOWN_LEFT__DOWN_RIGHT"    : math.sqrt((circle_pixel_pos_pack["DOWN_LEFT"][0] - circle_pixel_pos_pack["DOWN_RIGHT"][0])**2 + (circle_pixel_pos_pack["DOWN_LEFT"][1] - circle_pixel_pos_pack["DOWN_RIGHT"][1])**2)})
                if Detect_Robot.PRINT_DEBUG:
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
        
        # get min of the list and sorted the list
        min_len = min(length_list, key=length_list.get)
        length_list = sorted(length_list)

        if Detect_Robot.PRINT_DEBUG:
            print(f"The min_len is {min_len}")
        
        ''' #FIXME: This section might not be need any more
        if 'prime' in circle_pixel_pos_pack.keys():
            min_len = min_len.split("__")
            self.shortest_line_pixel_pos_list = [min_len[0], min_len[1]]
        else:
            min_len = min_len.split("__")
            self.shortest_line_pixel_pos_list = [min_len[0], min_len[1]]
        ''' 
        min_len = min_len.split("__")
        self.shortest_line_pixel_pos_list = [min_len[0], min_len[1]]
                
        if Detect_Robot.PRINT_DEBUG:
            print(f"The min length of line is {self.shortest_line_pixel_pos_list}")
            
        # Get angle
        angle = self.angle_between_shortest_line_circles(self.shortest_line_pixel_pos_list, circle_pixel_pos_pack)
        
        if Detect_Robot.PRINT_DEBUG:
            print(f'The final Angle is : {angle}')
            
        return angle
    
    def angle_between_shortest_line_circles(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        """_summary_
        base on pixel position on the cropped image and the minimum length between
        the input circles this method calculates angle
        Args:
            pixel_position_circle_shortest_line (_type_, optional): _description_. Defaults to None.
            package of the shortest line to detect angle
            
            pixel_position_value_shortest_line (_type_, optional): _description_. Defaults to None.
            package of the circle position with all values
        Returns:
            _type_: _description_:
            angle
        """        
        ppcsl = pixel_position_circle_shortest_line[0] + "__" + pixel_position_circle_shortest_line[1]
        # print(f'ppcsl: {ppcsl}')
        if self.check_circle_position(listCirclePosition=pixel_position_circle_shortest_line):
            
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
        
    def check_circle_position(self, listCirclePosition):
        ''' Durchmesser ist Ungültig '''
        if listCirclePosition[0]     == "TOP_RIGHT"  and listCirclePosition[0] == "DOWN_LEFT":
            return False
        elif listCirclePosition[0]   == "DOWN_LEFT"  and listCirclePosition[0] == "TOP_RIGHT":
            return False
        elif listCirclePosition[0]   == "TOP_LEFT"   and listCirclePosition[0] == "DOWN_RIGHT":
            return False
        elif listCirclePosition[0]   == "DOWN_RIGHT" and listCirclePosition[0] == "TOP_LEFT":
            return False
        else:
            return True
        
    def angle_circle_position_topRight_topLeft(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        """_summary_
        calculate angle between shortest line in position Top_Right
        and Top_Left
        
        Args:
            pixel_position_circle_shortest_line (_type_, optional): _description_. Defaults to None.
            pixel_position_value_shortest_line (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_:
            return Angle
        """
        Angle = None
        
        if pixel_position_circle_shortest_line[0] == "TOP_RIGHT" and pixel_position_circle_shortest_line[1] == "TOP_LEFT":
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["TOP_LEFT"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] > pixel_position_value_shortest_line["TOP_LEFT"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("1 is right and down")
                        print("2 is left and up")
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]])
                    Angle =  ( 180 - Angle )
                    return Angle
                else:
                    if Detect_Robot.PRINT_DEBUG:
                        print("1 is right and up")
                        print("2 is left and down")
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]])
                    Angle =  ( 180 - Angle )
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
        """_summary_
        calculate angle between shortest line in position Top_Right
        and Down_Left
        
        Args:
            pixel_position_circle_shortest_line (_type_, optional): _description_. Defaults to None.
            pixel_position_value_shortest_line (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_:
            return Angle
        """
        if pixel_position_circle_shortest_line[0] == "DOWN_LEFT" and pixel_position_circle_shortest_line[1] == "TOP_LEFT":
            print("second assumption 1")
            if pixel_position_value_shortest_line["TOP_LEFT"][0] < pixel_position_value_shortest_line["DOWN_LEFT"][0]:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["DOWN_LEFT"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("2 is up and back")
                        print("3 is down and frot")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]])
                    Angle = Angle * -1
                    return Angle                
                    
                else:
                    if Detect_Robot.PRINT_DEBUG:
                        print("2 is down and back")
                        print("3 is up and front")
                    print("THIS CONFIG IS NOT POSSIBLE !!!")
                    
            else:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["DOWN_LEFT"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("2 is up and front")
                        print("3 is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]])
                    Angle = 90 - Angle + 90
                    return Angle
                

        if pixel_position_circle_shortest_line[1] == "DOWN_LEFT" and pixel_position_circle_shortest_line[0] == "TOP_LEFT":
            if Detect_Robot.PRINT_DEBUG:
                print("second assumption 2")
            if pixel_position_value_shortest_line["TOP_LEFT"][0] < pixel_position_value_shortest_line["DOWN_LEFT"][0]:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["DOWN_LEFT"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("2 is up and back")
                        print("3 is down and front")
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
        """_summary_
        calculate angle between shortest line in position Down_Lef
        and Down_Right
        
        Args:
            pixel_position_circle_shortest_line (_type_, optional): _description_. Defaults to None.
            pixel_position_value_shortest_line (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_:
            return Angle
        """
        if pixel_position_circle_shortest_line[0] == "DOWN_LEFT" and pixel_position_circle_shortest_line[1] == "DOWN_RIGHT":
            if pixel_position_value_shortest_line["DOWN_LEFT"][0] < pixel_position_value_shortest_line["DOWN_RIGHT"][0]:
                if pixel_position_value_shortest_line["DOWN_LEFT"][1] < pixel_position_value_shortest_line["DOWN_RIGHT"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("3 is up and back")
                        print("4 is down and front")
                    # TODO: Check the 10 degree error
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    return Angle
                else:
                    if Detect_Robot.PRINT_DEBUG:
                        print("3 is down and back")
                        print("4 is up and front")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    return Angle
            else:
                print("THIS IS NOT POSSIBLE!!")

        if pixel_position_circle_shortest_line[1] == "DOWN_LEFT" and pixel_position_circle_shortest_line[0] == "DOWN_RIGHT":
            if pixel_position_value_shortest_line["DOWN_LEFT"][0] < pixel_position_value_shortest_line["DOWN_RIGHT"][0]:
                if pixel_position_value_shortest_line["DOWN_LEFT"][1] < pixel_position_value_shortest_line["DOWN_RIGHT"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("3 is up and back")
                        print("4 is down and front")
                    # TODO: Check the 10 degree error
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    return Angle
                else:
                    if Detect_Robot.PRINT_DEBUG:
                        print("3 is down and back")
                        print("4 is up and front")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    return Angle
            else:
                print("THIS IS NOT POSSIBLE!!")
    
    def angle_circle_position_topRight_downRight(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        """_summary_
        calculate angle between shortest line in position Top_Right
        and Down_Left
        
        Args:
            pixel_position_circle_shortest_line (_type_, optional): _description_. Defaults to None.
            pixel_position_value_shortest_line (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_:
            return Angle
        """
        if pixel_position_circle_shortest_line[0] == "DOWN_RIGHT"  and pixel_position_circle_shortest_line[1] == "TOP_RIGHT":
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["DOWN_RIGHT"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] < pixel_position_value_shortest_line["DOWN_RIGHT"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("1 is up and front")
                        print("4 is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                    Angle = Angle * -1
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                if Detect_Robot.PRINT_DEBUG:
                    print("1 is up and back")
                    print("4 is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                Angle = ( Angle + 90 + 90 ) * -1
                return Angle
        
        if pixel_position_circle_shortest_line[1] == "DOWN_RIGHT" and pixel_position_circle_shortest_line[0] == "TOP_RIGHT":
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["DOWN_RIGHT"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] < pixel_position_value_shortest_line["DOWN_RIGHT"][1]:
                    if Detect_Robot.PRINT_DEBUG:
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
                if Detect_Robot.PRINT_DEBUG:
                    print("1 is up and back")
                    print("4 is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], # 
                                    PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                    PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]])
                Angle = ( Angle + 90 + 90 ) * -1
                return Angle
            
    def angle_circle_position_topRight_topRight(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        """_summary_
        calculate angle between shortest line in position Top_Right
        and Top_Right (Prime)
        
        Args:
            pixel_position_circle_shortest_line (_type_, optional): _description_. Defaults to None.
            pixel_position_value_shortest_line (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_:
            return Angle
        """
        if pixel_position_circle_shortest_line[0] == "TOP_RIGHT"  and pixel_position_circle_shortest_line[1] == "prime":
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] > pixel_position_value_shortest_line["prime"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("1 is up and front")
                        print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = ( 180 + Angle ) * -1
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                if Detect_Robot.PRINT_DEBUG:
                    print("1 is up and back")
                    print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = ( 180 + Angle ) * -1
                return Angle
            
        if pixel_position_circle_shortest_line[0] == "prime"  and pixel_position_circle_shortest_line[1] == "TOP_RIGHT":
            if pixel_position_value_shortest_line["TOP_RIGHT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["TOP_RIGHT"][1] > pixel_position_value_shortest_line["prime"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("1 is up and front")
                        print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = ( 180 + Angle ) * -1
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                if Detect_Robot.PRINT_DEBUG:
                    print("1 is up and back")
                    print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["TOP_RIGHT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["TOP_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = ( 180 + Angle ) * -1
                return Angle
    
    def angle_circle_position_topLeft_topLeft(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        """_summary_
        calculate angle between shortest line in position Top_Left
        and Top_Left (Prime)
        
        Args:
            pixel_position_circle_shortest_line (_type_, optional): _description_. Defaults to None.
            pixel_position_value_shortest_line (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_:
            return Angle
        """
        if pixel_position_circle_shortest_line[0] == "TOP_LEFT"  and pixel_position_circle_shortest_line[1] == "prime":
            if pixel_position_value_shortest_line["TOP_LEFT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("2 is up and front")
                        print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = ( 90 - Angle ) + 90
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                if Detect_Robot.PRINT_DEBUG:
                    print("2 is up and back")
                    print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = ( 90 - Angle ) + 90
                return Angle
            
        if pixel_position_circle_shortest_line[0] == "prime"  and pixel_position_circle_shortest_line[1] == "TOP_LEFT":
            if pixel_position_value_shortest_line["TOP_LEFT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["TOP_LEFT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("2 is up and front")
                        print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = ( 90 - Angle ) + 90
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                if Detect_Robot.PRINT_DEBUG:
                    print("2 is up and back")
                    print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["TOP_LEFT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["TOP_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = ( 90 - Angle ) + 90
                return Angle
    
    def angle_circle_position_downLeft_downLeft(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        """_summary_
        calculate angle between shortest line in position Down_Left
        and Down_Left (Prime)
        
        Args:
            pixel_position_circle_shortest_line (_type_, optional): _description_. Defaults to None.
            pixel_position_value_shortest_line (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_:
            return Angle
        """
        if pixel_position_circle_shortest_line[0] == "DOWN_LEFT"  and pixel_position_circle_shortest_line[1] == "prime":
            if pixel_position_value_shortest_line["DOWN_LEFT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["DOWN_LEFT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("3 is up and front")
                        print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]])
                    Angle = Angle * -1 
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                if Detect_Robot.PRINT_DEBUG:
                    print("3 is up and back")
                    print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                # Angle = ( 90 + Angle ) + 90
                Angle = Angle * -1 
                return Angle
            
        if pixel_position_circle_shortest_line[0] == "prime"  and pixel_position_circle_shortest_line[1] == "DOWN_LEFT":
            if pixel_position_value_shortest_line["DOWN_LEFT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["DOWN_LEFT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("3 is up and front")
                        print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = Angle * -1 
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                if Detect_Robot.PRINT_DEBUG:
                    print("3 is up and back")
                    print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["DOWN_LEFT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["DOWN_LEFT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = Angle * -1 
                return Angle
    
    def angle_circle_position_downRight_downRight(self, pixel_position_circle_shortest_line = None, pixel_position_value_shortest_line =  None):
        """_summary_
        calculate angle between shortest line in position Down_Right
        and Down_Right (Prime)
        
        Args:
            pixel_position_circle_shortest_line (_type_, optional): _description_. Defaults to None.
            pixel_position_value_shortest_line (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_:
            return Angle
        """
        if pixel_position_circle_shortest_line[0] == "DOWN_RIGHT"  and pixel_position_circle_shortest_line[1] == "prime":
            if pixel_position_value_shortest_line["DOWN_RIGHT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["DOWN_RIGHT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("4 is up and front")
                        print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = Angle * -1 
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                if Detect_Robot.PRINT_DEBUG:
                    print("4 is up and back")
                    print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                # Angle = ( 90 + Angle ) + 90
                Angle = Angle * -1 
                return Angle
            
        if pixel_position_circle_shortest_line[0] == "prime"  and pixel_position_circle_shortest_line[1] == "DOWN_RIGHT":
            if pixel_position_value_shortest_line["DOWN_RIGHT"][0] > pixel_position_value_shortest_line["prime"][0]:
                if pixel_position_value_shortest_line["DOWN_RIGHT"][1] < pixel_position_value_shortest_line["prime"][1]:
                    if Detect_Robot.PRINT_DEBUG:
                        print("4 is up and front")
                        print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                          PT2 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], #
                                          PT3 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                    Angle = Angle * -1 
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                if Detect_Robot.PRINT_DEBUG:
                    print("4 is up and back")
                    print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [pixel_position_value_shortest_line["prime"][0], pixel_position_value_shortest_line["prime"][1]], # 
                                      PT2 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["DOWN_RIGHT"][1]], #
                                      PT3 = [pixel_position_value_shortest_line["DOWN_RIGHT"][0], pixel_position_value_shortest_line["prime"][1]])
                Angle = Angle * -1 
                return Angle

    def gradient(self, pt1 ,pt2):
        """_summary_
        return gradient between two point
        Args:
            pt1 (_type_): _description_ Point Ont
            pt2 (_type_): _description_ Point Two
        """        
        return(pt2[1] - pt1[1])/ (pt2[0] - pt1[0]) 
    
    def getAngle(self, PT1 = None, PT2 = None, PT3 = None):
        """_summary_
        
        Args:
            PT1 (_type_, optional): _description_. Defaults to None.
            PT2 (_type_, optional): _description_. Defaults to None.
            PT3 (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_:
            return angle
        """        
        pt1 = PT1
        pt2 = PT2
        pt3 = PT3
        n1 = self.gradient(pt1 , pt2)
        n2 = self.gradient(pt1 ,pt3)
        angleR = math.atan((n2 - n1)/(1 + (n2*n1)))
        angleD = round(math.degrees(angleR))
        return angleD
    
    def match_robot(self, frameRobot:np.array = None):
        """_summary_

        Args:
            frameRobot (np.array, optional): _description_. Defaults to None.
            get rotated and cropped image to detect robot ID
        Returns:
            _type_: _description_
        """        
        circle_pixel_pos_pack = {"TOP_RIGHT":    "",
                                 "TOP_LEFT":     "",
                                 "DOWN_LEFT":    "",
                                 "DOWN_RIGHT":   ""}
               
        # Get red and green color contours of cropped and rotated image
        contours_red , _   = self.FuncFindContours(frame=frameRobot, circle_color="Masked_Red")
        contours_green , _ = self.FuncFindContours(frame=frameRobot, circle_color="Masked_Green")
        
        are_of_circle_min = self.CircleArea[0]
        are_of_circle_max = self.CircleArea[1]
        
        numRedCircle = 0 # for robots with all red ID
                
        """ contours for red area  """            
        for red_contour in contours_red:
            red_area = self.FuncCalContoursArea(contours=red_contour)
            if red_area < are_of_circle_max and red_area > are_of_circle_min:
                moment = self.FuncCalMoment(contours=red_contour)
                cx_red = int(moment["m10"]/moment["m00"])
                cy_red = int(moment["m01"]/moment["m00"])
                
                # Set circle position in img base on contours
                if cx_red > frameRobot.shape[0]/2 and cy_red < frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["TOP_RIGHT"] = "red"
                    numRedCircle += 1
                if cx_red < frameRobot.shape[0]/2 and cy_red < frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["TOP_LEFT"] = "red"
                    numRedCircle += 1
                if cx_red < frameRobot.shape[0]/2 and cy_red > frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["DOWN_LEFT"] = "red"
                    numRedCircle += 1
                if cx_red > frameRobot.shape[0]/2 and cy_red > frameRobot.shape[1]/2:
                    circle_pixel_pos_pack["DOWN_RIGHT"] = "red"
                    numRedCircle += 1
        
        # return if we on all red robot id
        if numRedCircle == 4:
            return self.loop_robot_id_list(color_pattern_list = circle_pixel_pos_pack)
        
        """ contours for green area """             
        for green_contour in contours_green:
            green_area = self.FuncCalContoursArea(contours=green_contour)
            if green_area < are_of_circle_max and green_area > are_of_circle_min:
                moment = self.FuncCalMoment(contours=green_contour)
                cx_green = int(moment["m10"]/moment["m00"])
                cy_green = int(moment["m01"]/moment["m00"])
                if cx_green > frameRobot.shape[0]/2 and cy_green < frameRobot.shape[1]/2:
                    if circle_pixel_pos_pack["TOP_RIGHT"] != "red":
                        circle_pixel_pos_pack["TOP_RIGHT"] = "green"
                    else:
                        print("Robot ID Detection, Position is occupied by red color")
                if cx_green < frameRobot.shape[0]/2 and cy_green < frameRobot.shape[1]/2:
                    if circle_pixel_pos_pack["TOP_LEFT"] != "red":
                        circle_pixel_pos_pack["TOP_LEFT"] = "green"
                    else:
                        print("Robot ID Detection, Position is occupied by red color")
                if cx_green < frameRobot.shape[0]/2 and cy_green > frameRobot.shape[1]/2:
                    if circle_pixel_pos_pack["DOWN_LEFT"] != "red":
                        circle_pixel_pos_pack["DOWN_LEFT"] = "green"
                    else:
                        print("Robot ID Detection, Position is occupied by red color")
                if cx_green > frameRobot.shape[0]/2 and cy_green > frameRobot.shape[1]/2:
                    if circle_pixel_pos_pack["DOWN_RIGHT"] != "red":
                        circle_pixel_pos_pack["DOWN_RIGHT"] = "green"
                    else:
                        print("Robot ID Detection, Position is occupied by red color")

        return self.loop_robot_id_list(color_pattern_list = circle_pixel_pos_pack)
    
    def loop_robot_id_list(self, color_pattern_list: dict = None):
        """_summary_
        Method get list of circle position from rotated image to match them with
        predefined list to find ID
        
        Args:
            color_pattern_list (dict, optional): _description_. Defaults to None.
            list of the circle position
        Returns:
            _type_: _description_:
            return robot Id
        """        
        for RobotID in Detect_Robot.Robot_Pattern_Dict:
            if Detect_Robot.Robot_Pattern_Dict[RobotID] == color_pattern_list:
                return RobotID
        if Detect_Robot.PRINT_DEBUG:       
            print(f"IT IS NOT ROBOT PATTERN, PATTERN: {color_pattern_list}")