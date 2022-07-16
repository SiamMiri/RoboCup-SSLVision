import json
import cv2
import numpy as np
import time
import imutils
from math import atan2, cos, sin, sqrt, pi, acos
from skimage.transform import (hough_line, hough_line_peaks)
import logging
from src.CythonCode import cython_me

from os import path
import os
from datetime import date, datetime

class Image_Processing(): 
    """ This class get images from DetectRobotBall """
    
    # This Parameters are belongs to the CLASS
    ########## Dimension in centimeters ########
#   0 ################## X ######################
    #                                           #
    #                                           #
    #                                           #
#   Y                 center                    #
    #                                           #
    #                                           #
    #                                           #
    #############################################
    Field_Size_X_Direction  = 277
    Field_Size_Y_Direction  = 188

    # Application Control Parameters
    MASK_COLOR_THRESHOLD        = True     # This Value should be as True to work correctly !!!
    PRINT_DEBUG                 = False    # Control debug session 
    def __init__(self):
        
        self._Frame_Data        = {}    # Robots Data

        # Minimum and Maximum area of the Robots
        # Red and Green circle have the same area
        
        self.area_of_circle_min      = 0 # for green and red
        self.area_of_circle_max      = 0 # for green and red
        self.area_of_circle_min_blue = 0
        self.area_of_circle_max_blue = 0
        self.boolArea = False       # Check if the Max and Min area has been calculated
            
        # Coefficient to convert pixel to centimeter
        self.xCoef              = 0
        self.yCoef              = 0
        
        # Width and hight frame size
        self.xFrameSizePixel    = 0 
        self.yFrameSizePixel    = 0
        
        self.ifSet              = False # False if image cm and pixel are not applied
        
        self.json_color_range   = None  # Read Color Config .json file to detect robots pattern
        
        self.json_frame_config        = None
        
        self.frameNew           = None  # save new frame as class object variable        

        self.__read_color_config_json_file() # Load HSV color json file
    
    def __read_color_config_json_file(self):
        """_summary_
        Read json file which contains HSV color range config
        and frame filter configuration
        """        
        try:
        # try to load the json file if exist
            with open("./src/Config/Robo_Color_Config.json") as color_config:
                self.json_color_range = json.load(color_config)
        # Catch Err in this case might be naming diff in json file and print defined
        except Exception as e:
            self.json_color_range = None
            print(f'Could Not Find Color Config .json File {e}')
        
        try:
        # try to load the json file if exist
            with open("./src/Config/CameraConfig.json") as color_config:
                self.json_frame_config = json.load(color_config)
        # Catch Err in this case might be naming diff in json file and print defined
        except Exception as e:
            self.json_frame_config = None
            print(f'Could Not Find Color Config .json File {e}')
            

    def _start_process(self, field_frame: np.array = None):
        """_summary_

        Args:
            field_frame (np.array, optional): _description_. Defaults to None. 
            frame of the field  should be passed'for process

        Returns:
            _type_: _description_: list of data as dic which contains info of the robots in the field
        """        
        
        # set coefficient of the width and hight of the field respect to the frame size
        # These values will be set just ones
        if self.ifSet != True: 
            # Set Pixel Value
            try:
                self.xFrameSizePixel = int(field_frame.shape[0]) # Setting the length of the X direction
                self.yFrameSizePixel = int(field_frame.shape[0]) # Setting the length of the Y direction

                self.xCoef = Image_Processing.Field_Size_X_Direction/self.xFrameSizePixel
                self.yCoef = Image_Processing.Field_Size_Y_Direction/self.yFrameSizePixel
                self.ifSet = True # set pixel and centimeter coefficient is done
                
            except Exception as e: 
                print(f"_start_process {e}")
        
        return self._detect_blue_circle(frame = field_frame)
                
    def _detect_blue_circle(self, frame: cv2.VideoCapture.read = None):
        
        # Clear frame data for new set of data
        self._Frame_Data.clear()
        
        self.frameNew = frame
        '''NOTE: This part of function is currently move to Cython Function
        if there is a need to move to python version active this part and
        the next mention part, do not forget to comment Cython Function'''
        # Constants:
        # blue_color_num          = 0
        # blue_color_dict         = f'Blue_Circle_{blue_color_num}'
        # if_is_ball              = False
        # crop_img = None

        # The X_Position and Y_Position of the circle blue 
        # cx_blue = 0 
        # cy_blue = 0 
        # moment  = 0

        # FIXME: This function take time, it should be optimized
        # finding blue counter
        contours_blue, mask_blue = self.find_contours_mask(frame= frame, circle_color= "Blue")
        
        '''NOTE: This part of function is currently move to Cython Function'''
        # frame[mask_blue > 0] = (255, 0 , 0)
        
        if self.boolArea != True:
            self.area_of_circle_min_blue ,self.area_of_circle_max_blue = self._calculate_area_of_circle(frame =frame, circle_color = "Blue") 
        
        lenContoursBlue = len(contours_blue)
        try:     
            if Image_Processing.PRINT_DEBUG:
                print(f"blue_area_of_circle_min {self.area_of_circle_min_blue}")
                print(f"blue_self.area_of_circle_max {self.area_of_circle_max_blue}")
            
            # Cython Function, if the function is moved to python this function should be committed
            cython_me.loop_blue_circle(lenContoursBlue, contours_blue, self.area_of_circle_max_blue, self.area_of_circle_min_blue, self._find_red_green_circle)

            '''NOTE: This part of function is currently move to Cython Function
            """ contours for blue area """
            for contours in contours_blue:
                
                blue_area = cv2.contourArea(contours)
                if blue_area == 0:
                    continue
                if Image_Processing.PRINT_DEBUG:
                    print(f"blue_are {blue_area}")
                if blue_area < area_of_circle_max and blue_area > area_of_circle_min:
                    if Image_Processing.MASK_COLOR_THRESHOLD:
                        frame[mask_blue > 0] = (255, 0 , 0)
                    
                    moment = cv2.moments(contours) 

                    cx_blue = int(moment["m10"]/moment["m00"])
                    cy_blue = int(moment["m01"]/moment["m00"])
                    
                    blue_color_num          += 1
                    blue_color_dict         = f'Blue_Circle_{blue_color_num}'
                    self._find_red_green_circle(cy_blue, cx_blue, blue_color_dict)
                    '''
            return self._Frame_Data
        except Exception as e:
            print(f"_detect_blue_circle: {e}")                  
    
    def _calculate_area_of_circle(self, frame: np.array = None, circle_color:str = None):
        """_summary_
        Fix Variable in this function are suited for the 1080 pixel image, it might
        not work on other resolution
        Args:
            frame (np.array, optional): _description_. Defaults to None. The frame field is loaded to
            this method, the size of the robots circle(id pattern) are dependent to img size
            
            circle_color (str, optional): _description_. Defaults to None. Area which is dedicated to each
            circle can be differ from blue and green/red 

        Returns:
            _type_: _description_: return maximum and min area of the which are accepted to detect robots
        """        
        area_of_circle_min = None
        area_of_circle_max = None

        if circle_color == "Blue":
            # Calculate the area of the circle 
            area_of_circle_min = frame.shape[0]/200
            area_of_circle_max = frame.shape[0]/100

            area_of_circle_min = pi*area_of_circle_min**2
            area_of_circle_max = pi*area_of_circle_max**2

        if circle_color == "Red" or circle_color == "Green":
            area_of_circle_min = frame.shape[0]/12
            area_of_circle_max = frame.shape[0]/6

            area_of_circle_min = pi*area_of_circle_min**2
            area_of_circle_max = pi*area_of_circle_max**2

        return area_of_circle_min , area_of_circle_max

    def find_contours_mask(self, frame = None, circle_color:str = None):
        """_summary_

        Args:
            frame (_type_, optional): _description_. Defaults to None. the contours will be
            detected of a given frame
            circle_color (str, optional): _description_. Defaults to None. The required color
            should be given to detect mask and contours

        Returns:
            _type_: _description_: Mask and contours of the frame
        """        
        contours = None
        mask     = None
        
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert frame to hsv format for setting mask
        
        # Source: https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
        if circle_color == "Blue":
            low_blue             = np.array(self.json_color_range["Low_Blue"] , np.uint8)
            upper_blue           = np.array(self.json_color_range["Up_Blue"]  , np.uint8) 
            # define masks
            mask                 = cv2.inRange(frameHSV, low_blue       ,upper_blue)
            # find contours
            contours             = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            contours             = imutils.grab_contours(contours)
            return contours, mask
        
        # Color: Red
        elif circle_color == "Red":
            low_red         = np.array(self.json_color_range["Low_Red"], np.uint8)
            upper_red       = np.array(self.json_color_range["Up_Red"], np.uint8)
            # define masks
            mask            = cv2.inRange(frameHSV, low_red        ,upper_red)
            # find contours
            contours        = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            contours        = imutils.grab_contours(contours)
            return contours, mask
            
        elif circle_color == "Green":
            # Color: green
            low_green       = np.array(self.json_color_range["Low_Green"], np.uint8)
            upper_green     = np.array(self.json_color_range["Up_Green"], np.uint8)
            # define masks
            mask            = cv2.inRange(frameHSV, low_green      ,upper_green)
            # find contours
            contours        = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            contours        = imutils.grab_contours(contours)
            return contours, mask
            
        elif circle_color == "Black":
            # Color: black
            low_black       = np.array([0, 0, 0], np.uint8)
            upper_black     = np.array([180, 255, 145], np.uint8)
            # define masks
            mask            = cv2.inRange(frameHSV, low_black      ,upper_black)
            # find contours
            contours        = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            contours        = imutils.grab_contours(contours)
            return contours, mask

        elif circle_color == "Masked_Red":
            low_red         = np.array([0, 26, 104], np.uint8)
            upper_red       = np.array([11, 255, 255], np.uint8)
            # define masks
            mask            = cv2.inRange(frameHSV, low_red        ,upper_red)
            contours        = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            contours        = imutils.grab_contours(contours)
            return contours, mask
            
        elif circle_color == "Masked_Green":
            # Color: green
            low_green       = np.array([30,  175 , 140], np.uint8)
            upper_green     = np.array([105, 255, 255], np.uint8)
            mask            = cv2.inRange(frameHSV, low_green      ,upper_green)
            contours        = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            contours        = imutils.grab_contours(contours)
            return contours, mask
        else:
            return contours, mask
    
    def set_image_filter(self, frame  : cv2.VideoCapture.read, filterJsonFile = None, Blur  : bool = False,GaussianBlur  : bool = False , Segmentation : bool  = False , Res : bool = False):
        """_summary_

        Args:
            frame (cv2.VideoCapture.read): _description_: frame which need to apply filter
            
            filterJsonFile (_type_, optional): _description_. Defaults to None. loaded json file which
            contains filter specification
            
            Blur (bool, optional): _description_. Defaults to False.
            GaussianBlur (bool, optional): _description_. Defaults to False.
            Segmentation (bool, optional): _description_. Defaults to False.
            Res (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_: return frame after filtering image
        """        
        self.ifSet = True
        if filterJsonFile != None :
            ''' Blur Image '''
            if Blur is not False:
                frame = cv2.blur(src=frame, ksize=(filterJsonFile["Blur"][0], filterJsonFile["Blur"][1]))
                print("Blur is applied")
            

            ''' Blured Image '''
            if GaussianBlur is not False:
                frame = cv2.GaussianBlur(frame, (filterJsonFile["GaussianBlur"][0], filterJsonFile["GaussianBlur"][1]), 0)
                print("GaussianBlur is applied")
            
            
            ''' Segmentation '''
            if Segmentation is not False:
                print("Segmentation is applied")
            
                # reshape the image to a 2D array of pixels and 3 color values (RGB) to push it in to the kmeans
                new_frame = np.float32(frame.reshape((-1, 3)))

                # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
                # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

                _, labels, centers = cv2.kmeans(new_frame, filterJsonFile["Segmentation"], None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                labels = labels.reshape((frame.shape[:-1]))
                reduced = np.uint8(centers)[labels]

                # frame = [np.hstack([frame, reduced])]
                # frame = np.vstack(frame)
                # frame = reduced
                for i, c in enumerate(centers):
                    mask  = cv2.inRange(labels, i, i)
                    mask  = np.dstack([mask] * 3)  # Make it 3 channel
                    frame = cv2.bitwise_and(reduced, mask)
                '''
                for i, c in enumerate(centers):
                    mask = cv2.inRange(labels, i, i)
                    mask = np.dstack([mask] * 3)  # Make it 3 channel
                    ex_img = cv2.bitwise_and(frame, mask)
                    ex_reduced = cv2.bitwise_and(reduced, mask)
                    frame.append(np.hstack([ex_img, ex_reduced]))
                    print("Segmentation is applied")
                '''
            
            ''' Resolution '''
            if Res is not False:
                try:
                    x = (filterJsonFile["resize_frame"][0], filterJsonFile["resize_frame"][1])
                    if Image_Processing.PRINT_DEBUG:
                        print(f"Resolution changed to: {x}")
                    frame = cv2.resize(frame, x)
                except Exception as e:
                    print(f'set_image_filter: Could not resize image {e}')
        else: 
            print("filter .json file is not loaded or it is corrupted")

        return frame

    def _crop_robot_circle(self, img : cv2.VideoCapture.read, pos_y , pos_x, if_is_ball):
        """_summary_
        This function first cropped image as square and then delete extra part and make it 
        circle image
        
        Args:
            img (cv2.VideoCapture.read): _description_
            img/frame which should be cropped
            
            pos_y (_type_): _description_ x pixel position of the img which required to
            be cropped. It is center of the cropped image
            pos_x (_type_): _description_ y pixel position of the img which required to
            be cropped. It is center of the cropped image
            
            if_is_ball (_type_): _description_ set to true if ball is required to be detected
            as it has different are

        Returns:
            _type_: _description_ cropped img/frame
        """        
        if if_is_ball:
            pos_y   = pos_y - int(img.shape[0]/40) 
            pos_x   = pos_x - int(img.shape[1]/65)
            radius  = int(img.shape[1] / 65)

            # crop image as a square
            img = img[pos_y:pos_y+radius*2, pos_x:pos_x+radius*2]
            # create a mask
            mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8) 
            # create circle mask, center, radius, fill color, size of the border
            cv2.circle(mask,(radius,radius), radius, (255,255,255),-1)
            # get only the inside pixels
            fg = cv2.bitwise_or(img, img, mask=mask)

            mask = cv2.bitwise_not(mask)
            background = np.full(img.shape, 255, dtype=np.uint8)
            bk = cv2.bitwise_or(background, background, mask=mask)
            crop_img = cv2.bitwise_or(fg, bk)
            return crop_img
        else:
            pos_y   = pos_y - int(img.shape[0]/40) 
            pos_x   = pos_x - int(img.shape[1]/65)
            radius  = int(img.shape[1] / 65)

            # crop image as a square
            img = img[pos_y:pos_y+radius*2, pos_x:pos_x+radius*2]
            # create a mask
            mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8) 
            # create circle mask, center, radius, fill color, size of the border
            cv2.circle(mask,(radius,radius), radius, (255,255,255),-1)
            # get only the inside pixels
            fg = cv2.bitwise_or(img, img, mask=mask)

            mask = cv2.bitwise_not(mask)
            background = np.full(img.shape, 255, dtype=np.uint8)
            bk = cv2.bitwise_or(background, background, mask=mask)
            crop_img = cv2.bitwise_or(fg, bk)
            return crop_img
    
    def _find_red_green_circle(self, cy: int = None, cx :int = None, blue_color_dict:int = None): # , Robo_Num: int = None,  img : cv2.VideoCapture.read = None,
        """_summary_

        Args:
            cy (int, optional): _description_. Defaults to None. x position of the blue color. It is
            required for cropping image
            cx (int, optional): _description_. Defaults to None. y position of the blue color. It is
            required for cropping image
            
            blue_color_dict (int, optional): _description_. Defaults to None. Dictionary
            for updating the self._Frame_Data 
        """        
        # Crop image with the given coordinate to find circle's of robots ID
        img = self._crop_robot_circle(self.frameNew, cy, cx, False)

        # constants
        num_of_circle   = 0
        num_of_red      = 0
        num_of_green    = 0
        
        """ List of color position in cropped image 
            prime is used when one side of the frame is already
            occupied by circle """
        circlePack = {"TOP_RIGHT":      [],
                        "TOP_LEFT":     [],
                        "DOWN_LEFT":    [],
                        "DOWN_RIGHT":   [],
                        "prime":        []}
        
        # finding Red Color and Green Color Contours 
        contours_red,   mask_red   = self.find_contours_mask(frame= img, circle_color= "Red")
        contours_green, mask_green = self.find_contours_mask(frame= img, circle_color= "Green")
        
                        
        if Image_Processing.MASK_COLOR_THRESHOLD:
            img[mask_green > 0] = (0  , 255 , 0  )
            img[mask_red   > 0] = (0  , 0   , 255)
            
        # Area for both green and blue Color is the same
        if self.boolArea != True:  
            self.area_of_circle_min, self.area_of_circle_max = self._calculate_area_of_circle(frame=img, circle_color= "Red")
            self.boolArea = True
        if Image_Processing.PRINT_DEBUG:
            print(f'area_min: {self.area_of_circle_min}')
            print(f'area_max: {self.area_of_circle_max}')
             
        try:
            """ contours for red area  """            
            for contours in contours_red:
                red_area = cv2.contourArea(contours)
                if red_area == 0:
                    continue
                if red_area < self.area_of_circle_max and red_area > self.area_of_circle_min:
                    if Image_Processing.PRINT_DEBUG:
                        print(f"red_area {red_area}")
                    moment = cv2.moments(contours) # NOTE: check me again 
                    cx_red = int(moment["m10"]/moment["m00"])
                    cy_red = int(moment["m01"]/moment["m00"])
                    if Image_Processing.PRINT_DEBUG:
                        print(f"cx_red {cx_red}")
                        print(f"cy_red {cy_red}")
                    
                    # set the position of the circle
                    position, circlePack = self.set_circle_position(img.shape[0], img.shape[1], cx_red, cy_red, circlePack)
                    if position != None:
                        num_of_circle += 1
                        num_of_red    += 1

                    if Image_Processing.PRINT_DEBUG:
                        print(f'Red Position is: {position}')

            # For robots that has 4 red circle as a ID
            if num_of_red == 4:
                # Debug
                if Image_Processing.PRINT_DEBUG:
                    print(f'circle_pack: {circlePack}')
                # clear circle package from any empty list
                for x in list(circlePack.keys()):
                    if circlePack[x] == []:
                        del circlePack[x]
                
                # updating frame data dictionary
                myDict = {blue_color_dict : [cx, cy, img, circlePack]}
                self._Frame_Data.update(myDict)
                return # return as the num of circle is complete and eq to 4
            
            """ contours for green area """             
            for contours in contours_green:
                green_area = cv2.contourArea(contours)
                if green_area == 0:
                    continue
                if Image_Processing.PRINT_DEBUG:
                    print(f"Green_are: {green_area}")
                
                if green_area < self.area_of_circle_max and green_area > self.area_of_circle_min:
                    moment = cv2.moments(contours) # NOTE: check me again 
                    cx_green = int(moment["m10"]/moment["m00"])
                    cy_green = int(moment["m01"]/moment["m00"])
                    if Image_Processing.PRINT_DEBUG:
                        print(f"cx_green {cx_green}")
                        print(f"cy_green {cy_green}")

                    # set the position of the circle
                    position, circlePack = self.set_circle_position(img.shape[0], img.shape[1], cx_green, cy_green, circlePack)
                    
                    if position != None:
                        num_of_circle += 1
                        num_of_green   += 1

                    if Image_Processing.PRINT_DEBUG:
                        print(f'Green Position is: {position}')
            
            # clear circle package from any empty list
            for x in list(circlePack.keys()):
                if circlePack[x] == []:
                    del circlePack[x]
            
            # Debug
            if Image_Processing.PRINT_DEBUG:
                print(f'circle_pack: {circlePack}')
              
            
            # updating frame data dictionary
            if num_of_circle == 4 :
                myDict = {blue_color_dict : [cx, cy, img, circlePack]}
                self._Frame_Data.update(myDict)

        except Exception as e:
            print(f'ERROR: _find_red_green_circle: {e}')
   
    def rotate_image_by_degree(self, frame=None, degree=None):
        """_summary_
        This method is called externally after finding angle from robot detection class
        Args:
            frame (_type_, optional): _description_. Defaults to None.  Input frame
            degree (_type_, optional): _description_. Defaults to None. Degree to be rotated

        Returns:
            _type_: _description_ : return rotated image by given degree
        """        
        # rotate image after calculation
        (h, w) = frame.shape[:2]
        M = cv2.getRotationMatrix2D((frame.shape[1] // 2, frame.shape[1] // 2), degree, 1.0)
        imgRotated = cv2.warpAffine(frame, M, (w, h))
        return imgRotated
    
    def set_circle_position(self,img_shape_x, img_shape_y, x, y, circlePack):
        """_summary_
        this function regardless of the circle color set their position to the given 
        dictionary
        Args:
            img_shape_x (_type_): _description_ x value of the img size (pixel)
            img_shape_y (_type_): _description_ y value of the img size (pixel)
            x (_type_): _description_ x pixel position of the circle 
            y (_type_): _description_ y pixel position of the circle
            circlePack (_type_): _description_ circle dictionary which required to
            be set by position

        Returns:
            _type_: _description_ dictionary of the circle position 
        """        
        if x >= img_shape_x//2:
            if y <= img_shape_y//2:
                if len(circlePack["TOP_RIGHT"]) == 0:
                    circlePack["TOP_RIGHT"] = [x, y]
                    
                    return "TOP_RIGHT", circlePack
                elif len(circlePack["prime"]) == 0: 
                    circlePack["prime"] = [x, y]
                    
                    return "prime", circlePack
        
        if x <= img_shape_x//2:
            if y <= img_shape_y//2:
                if len(circlePack["TOP_LEFT"]) == 0:
                    circlePack["TOP_LEFT"] = [x, y]
                    
                    return "TOP_LEFT", circlePack
                elif len(circlePack["prime"]) == 0: 
                    circlePack["prime"] = [x, y]
                    
                    return "prime", circlePack
    
        if x <= img_shape_x//2:
            if y >= img_shape_y//2:
                if len(circlePack["DOWN_LEFT"]) == 0:
                    circlePack["DOWN_LEFT"] = [x, y]
                    
                    return "DOWN_LEFT", circlePack
                elif len(circlePack["prime"]) == 0: 
                    circlePack["prime"] = [x, y]
                    
                    return "prime", circlePack

        if x >= img_shape_x//2:
            if y >= img_shape_y//2:
                if len(circlePack["DOWN_RIGHT"]) == 0:
                    circlePack["DOWN_RIGHT"] = [x, y]
                    
                    return "DOWN_RIGHT", circlePack
                elif len(circlePack["prime"]) == 0: 
                    circlePack["prime"] = [x, y]
                    
                    return "prime", circlePack

        return None, None

    def convert_pixel_to_centimeter(self, xVal = None, yVal = None):  
        """_summary_

        Args:
            xVal (_type_, optional): _description_. Defaults to None. x pixel position 
            yVal (_type_, optional): _description_. Defaults to None. y pixel position

        Returns:
            _type_: _description_ x and y position in centimeters
        """        
        x = xVal * self.xCoef
        y = yVal * self.yCoef
            
        return x, y

    def saveFrame(self, frame, frame_num):
        """_summary_
        Saving Given Frame
        Args:
            frame (_type_): _description_       Frame required ti be saved
            frame_num (_type_): _description_   Name of the ing file 
        """        
        if path.exists("Saved_Framed"):
            pass
        else: 
            os.makedirs("Saved_Framed")
        currentTime = datetime.now()
        try:
            if frame is not None:
                cv2.imwrite(f"./Saved_Framed/Field_Frame_Number_{frame_num}_Date_{currentTime}.jpg", frame)
            else:
                print("Image is not valid")
        except Exception as e:
            print(f'saveFrame: {e}')

    def calculate_contours_area(self, contours = None):
        return cv2.contourArea(contours)
    
    def calculate_moment(self, contours = None):
        return cv2.moments(contours)
    
    ############################ This Methods are extra and might be needed for later work ############################
    def creat_circle_color_id_mask(self, frame: cv2.VideoCapture.read, color: str = None, cordinate_list :list = None):
        """_summary_
        this method is used for creating circle of given color in a given position 
        Args:
            frame (cv2.VideoCapture.read): _description_
            color (str, optional): _description_. Defaults to None.
            cordinate_list (list, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Center coordinates
        center_coordinates = (int(frame.shape[0]/2), int(frame.shape[1]/2))
        # print(f'center_coordinates: {center_coordinates}')
        # print(frame.shape[0]/2)
        # Radius of circle
        # radius = int(x / 1)         
        # Line thickness of -1 px
        thickness = -1
        if color == "blue":
            # blue color in BGR
            color = (255, 0, 0)
            radius = int(frame.shape[0]/6)
            # print(f'radias is : {radius}')
            frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
            # print("Cicle done!")
            return frame
        
        if color == "red": 
            # red color in BGR
            # center_coordinates = (x + 5 , x - 5)
            center_coordinates = cordinate_list
            color = (0, 0 , 255) # (0, 0, 255)
            radius = int(frame.shape[0]/8)
            # radius = 3
            #for i in cordinate_list:
                #center_coordinates = (i[0], i[1])
            frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
            return frame

        if color == "green": 
            # green color in BGR
            center_coordinates = cordinate_list
            color = (0, 255, 0)
            radius = int(frame.shape[0]/8)
            #for i in cordinate_list:
                #center_coordinates = (i[0], i[1])
            frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
            return frame
    
        return frame
    
    def _crop_robot_rec(self, img : cv2.VideoCapture.read, pos_y , pos_x):
        np_crop_img = np.array(img)
        pos_y   = pos_y - 15
        pos_x   = pos_x - 15
        higth   = pos_y + 30
        width   = pos_x + 30
        # crop rec image from robot
        np_crop_img  = np_crop_img[pos_y:higth, pos_x:width]
        crop_img = np_crop_img
        
        return crop_img
    
    def detect_robot_id_gray_scale(self, frame: cv2.VideoCapture.read):
        # Contants:
        
        robot_num  = 1
        if_is_ball = False
        
        ret, thresh1 = cv2.threshold(frame, 95, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(frame, 115, 255, cv2.THRESH_BINARY)
        blue_thresh = thresh1 - thresh2
        
        # ret, thresh1 = cv2.threshold(frame, 158, 255, cv2.THRESH_BINARY)
        # ret, thresh2 = cv2.threshold(frame, 170, 255, cv2.THRESH_BINARY)
        # green_thresh = thresh1 - thresh2
        
        # ret, thresh1 = cv2.threshold(frame, 115, 255, cv2.THRESH_BINARY)
        # ret, thresh2 = cv2.threshold(frame, 130, 255, cv2.THRESH_BINARY)
        # red_thresh = thresh1 - thresh2
        
        contours_blue, hierarchy = cv2.findContours(image=blue_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        
        area_of_circle_min = frame.shape[0]/180
        area_of_circle_max = frame.shape[0]/175

        area_of_circle_min = pi*area_of_circle_min**2
        area_of_circle_max = pi*area_of_circle_max**2
        
        image_copy = frame.copy()
        
        """ contours for blue area """
        for contours in contours_blue:
            blue_area = cv2.contourArea(contours)
            if blue_area < area_of_circle_max and blue_area > area_of_circle_min:
                moment = cv2.moments(contours) # NOTE: check me again 
                cx_blue = int(moment["m10"]/moment["m00"])
                cy_blue = int(moment["m01"]/moment["m00"])
                cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                crop_img = self._crop_robot_circle(frame, cy_blue, cx_blue, if_is_ball)
                # crop_img = self.creat_circle_color_id_mask(crop_img, color = "blue", cordinate_list=(cx_blue, cy_blue))
        
        '''
        # B, G, R channel splitting
        blue, green, red = cv2.split(frame)
        contours1, hierarchy1 = cv2.findContours(image=blue, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image_contour_blue = blue.copy()
        cv2.drawContours(image=image_contour_blue, contours=contours1, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        '''
        return crop_img

    def drawAxis(self, img, p_, q_, color, scale):
        p = list(p_)
        q = list(q_)
        
        ## [visualization1]
        angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
        
        # Here we lengthen the arrow by a factor of scale
        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
        
        # create the arrow hooks
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
        
        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    def overlay_image_alpha(self, robo_img, field_img):
        a = np.where(robo_img > 0)
        b = np.where(field_img == 129)  # picked one of the channels in your image
        bbox_guy = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        bbox_mask = np.min(b[0]), np.max(b[0]), np.min(b[1]), np.max(b[1])
        guy = robo_img[bbox_guy[0]:bbox_guy[1], bbox_guy[2]:bbox_guy[3],:]
        target = field_img[bbox_mask[0]:bbox_mask[1], bbox_mask[2]:bbox_mask[3],:]
        guy_h, guy_w, _ = guy.shape
        mask_h, mask_w, _ = target.shape
        fy = mask_h / guy_h
        fx = mask_w / guy_w
        scaled_guy = cv2.resize(guy, (0,0), fx=fx,fy=fy)
        for i, row in enumerate(range(bbox_mask[0], bbox_mask[1])):
            for j, col in enumerate(range(bbox_mask[2], bbox_mask[3])):
                field_img[row,col,:] = scaled_guy[i,j,:]

        return field_img

    def detect_ball(self, frame: np.array):
        if_is_ball  = True
        frame_hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        try:
        # try to load the json file if exist
            with open("./src/Config/Robo_Color_Config.json") as color_config:
                color_range = json.load(color_config)
            b_json = True
        # Catch Err in this case might be naming diff in json file and print defined
        except Exception as e:
            b_json = False
            color_range = None
            print(f'detect_ball: {e}')

        if b_json == True:
            try:
                # Color: orange
                low_orange             = np.array(color_range["Low_Orange"], np.uint8)
                upper_orange           = np.array(color_range["Up_Orange"], np.uint8) 
                        
                # define masks
                mask_orange            = cv2.inRange(frame_hsv, low_orange ,upper_orange)

                contours_orange        = cv2.findContours(mask_orange.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                contours_orange        = imutils.grab_contours(contours_orange)
            except Exception as e:
                print(f'detect_ball: Could not open .json file {e}')
                
                
        area_of_circle_min = frame.shape[0]/120
        area_of_circle_max = frame.shape[0]/45

        area_of_circle_min = pi*area_of_circle_min**2
        area_of_circle_max = pi*area_of_circle_max**2
        if Image_Processing.PRINT_DEBUG:
            print(f"Orange_area_of_circle_min {area_of_circle_min}")
            print(f"Orange_area_of_circle_max {area_of_circle_max}")


        """ contours for Orange area """
        for contours in contours_orange:
            orange_area = cv2.contourArea(contours)
            if Image_Processing.PRINT_DEBUG:
                print(f"orange_are {orange_area}")
            if orange_area < area_of_circle_max and orange_area > area_of_circle_min:
                
                if Image_Processing.MASK_COLOR_THRESHOLD:
                    frame[mask_orange > 0] = (66, 161 , 245)
                    
                moment = cv2.moments(contours) # NOTE: check me again 
                cx_orange = int(moment["m10"]/moment["m00"])
                cy_orange = int(moment["m01"]/moment["m00"])
                print(f'cy_orange: {cy_orange}')
                if cy_orange < 400:
                    crop_img  = self._crop_robot_circle(frame, cy_orange, cx_orange, if_is_ball)
                    if_is_ball = False
                    break # FIXME : Not currect way to position ball
                # crop_img = self.creat_circle_color_id_mask(crop_img, color = "blue", cordinate_list=[cx_orange, cy_orange])
            #break
        if if_is_ball != True:
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(crop_img, str(int(fps)), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            cv2.namedWindow(f"BALL \tHit Escape to Exit")
            cv2.imshow(f"BALL \tHit Escape to Exit", crop_img)

        return frame

    def detect_robot_location(self, y: int, x: int, robo_num: int):
        robo_pos_dic = {}

        try: 
            with open('Robo_Pos.json', 'r') as openfile:
                robo_pos_dic = json.load(openfile)
        except Exception as e:
            print(f'detect_robot_location: {e}')

        if isinstance(robo_num, int) and isinstance(y, int) and isinstance(x, int):
            robo_pos = {
                f"Robot{robo_num}" : [y, x]
            }
            if robo_pos_dic is not None:
                if f"Robot{robo_num}" in robo_pos_dic:
                    robo_pos_dic[f"Robot{robo_num}"] = robo_pos[f"Robot{robo_num}"]
                else:
                    robo_pos_dic.update(robo_pos)
            else:
                robo_pos_dic.update(robo_pos)
        else:
            print("Failed to save Robo position")


        json_robo_pos_dic = json.dumps(robo_pos_dic, indent = 6)

        with open("Robo_Pos.json", "w") as outfile:
            outfile.write(json_robo_pos_dic)
