import json
from pickle import FALSE
import cv2
from cv2 import COLOR_BGR2RGB
import numpy as np
import time
import mediapipe as mp
import imutils
from collections import deque
import argparse
from math import atan2, cos, sin, sqrt, pi, acos
from skimage.transform import (hough_line, hough_line_peaks)
import math
import threading
import logging
import multiprocessing
from multiprocessing.context import Process
import time
import random

from sklearn.datasets import load_sample_image
#from UDPComminucation import UDP_COMMINUCATION


#from HSV_Color_Picker import #

# This class receive Images from the ControlCommand
# It processes images to detect the robot ID, origination
# and location. The return the data back as needed
class Image_Processing():
    """ This class get images from ControlCommand """
    # This Parameters are belongs to the CLASS

    # Dimension in cintimeteres
#   0 ################## X ######################
    #                                           #
    #                                           #
    #                                           #
#   Y                                           #
    #                                           #
    #                                           #
    #############################################
    Field_Size_X_Direction  = 277
    Field_Size_Y_Direction  = 188


    # Application Control Parameters #
    SAVE_ROBOT_IMAGE                = False
    SHOW_CIRCLE_LINE_CONNECTION     = False
    MASK_COLOR_THRESHOLD            = True
    CIRCLE_ID_COLOR_BY_CORDINATE    = True
    CENTER_CORDINATE                = False
    ROTATE_ROBOT_IMAGE              = False
    PRINT_DEBUG                     = True
    SHOW_MAIN_FIELD                 = True
    SHOW_ROBOTS_IN_NEW_WINDOW       = True
    CAPTURE_ONE_ROBOT_IMAGE         = False
    FIND_ROBOT_AND_BALL             = True
    LOAD_IMAGE                      = True 
    ANGLE                           = 0
    
    ## NOT DONE
    SHOW_ROBOT_INFO_IN_MAIN_FRAME   = False
    SHOW_BALL_INFO_IN_MAIN_FRAME    = False
    GRAY_SCALE_IMAGE_PROCCESSING    = True
    
    ## Image Path
    IMAGE_PATH                      = "HalfFrame.jpg"


    def __init__(self):

        self.pTime = 0
        self.robot_center_pos = None

        self.xCoef            = 0
        self.yCoef            = 0
        self.xFrameSizePixel  = 0 
        self.yFrameSizePixel  = 0
        self.RoboXRatioFrame  = 0
        self.RoboYRatioFrame  = 0 

        mpPose = mp.solutions.pose
        self.pose = mpPose.Pose()

    def start_capturing(self, camera_config: json):
        cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")
        self.camera_capture = cv2.VideoCapture(0) #
        """ Start capturing get camera config from start_capturing_command """
        self.set_camera_config(camera_config['CameraConfig'], Fps=False, Res=True, Focus=False)

        # Set Pixel Value
        try:
            self.xFrameSizePixel = int(camera_config["resize_frame"][0])
            self.yFrameSizePixel = int(camera_config["resize_frame"][1])
            self.xCoef = Image_Processing.Field_Size_X_Direction/self.xFrameSizePixel
            self.yCoef = Image_Processing.Field_Size_Y_Direction/self.yFrameSizePixel
        except Exception as e: 
            print(e)

        while True:
            cTime = time.time()
            if Image_Processing.GRAY_SCALE_IMAGE_PROCCESSING:
                if Image_Processing.LOAD_IMAGE:
                    frame = cv2.imread(Image_Processing.IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
                else:
                    ret, frame = self.camera_capture.read() # FIXME: Changed to load Image
                    
                    if not ret:
                        print("failed to grab frame")
                        break
                    frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
                if Image_Processing.FIND_ROBOT_AND_BALL:
                    # Detect Robot 
                    frame =  self.detect_robot_id_gray_scale(frame = frame)
                    # frame =  self.detect_ball(frame = frame)
                
            else:
                if Image_Processing.LOAD_IMAGE:
                    frame = cv2.imread(Image_Processing.IMAGE_PATH)  
                else:
                    ret, frame = self.camera_capture.read() # FIXME: Changed to load Image
                    if not ret:
                        print("failed to grab frame")
                        break
                
                if Image_Processing.FIND_ROBOT_AND_BALL:
                    # Detect Robot 
                    frame, SSL_DetectionRobot =  self.detect_robot_id_HSV_scale(frame = frame)
                    frame =  self.detect_ball(frame = frame) 
                    print(SSL_DetectionRobot)
                
            
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(frame, str(abs(int(fps))), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # Creat windows for the frame
            if Image_Processing.SHOW_MAIN_FIELD:
                cv2.imshow("RobotSoccer\tHit Escape or Q to Exit", frame)
            
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            
            if k % 256 == ord("q"):
                # Q pressed
                print("Escape hit, closing...")
                break

    def detect_ball(self, frame: cv2.VideoCapture.read):
        if_is_ball  = True
        frame_hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        try:
        # try to load the json file if exist
            with open("Robo_Color_Config.json") as color_config:
                color_range = json.load(color_config)
            b_json = True
        # Catch Err in this case might be naming diff in json file and print defined
        except Exception as e:
            b_json = False
            color_range = None
            print(e)

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
                print(f'Could not open .json file {e}')
                
                
        are_of_circle_min = frame.shape[0]/120
        are_of_circle_max = frame.shape[0]/45

        are_of_circle_min = pi*are_of_circle_min**2
        are_of_circle_max = pi*are_of_circle_max**2
        #if Image_Processing.PRINT_DEBUG:
        print(f"Orange_are_of_circle_min {are_of_circle_min}")
        print(f"Orange_are_of_circle_max {are_of_circle_max}")


        """ contours for Orange area """
        for contours in contours_orange:
            orange_area = cv2.contourArea(contours)
            #if Image_Processing.PRINT_DEBUG:
            print(f"orange_are {orange_area}")
            if orange_area < are_of_circle_max and orange_area > are_of_circle_min:
                
                if Image_Processing.MASK_COLOR_THRESHOLD:
                    frame[mask_orange > 0] = (66, 161 , 245)
                    
                moment = cv2.moments(contours) # NOTE: check me again 
                cx_orange = int(moment["m10"]/moment["m00"])
                cy_orange = int(moment["m01"]/moment["m00"])
                print(f'cy_orange: {cy_orange}')
                if cy_orange < 400:
                    crop_img  = self.crop_robot_circle(frame, cy_orange, cx_orange, if_is_ball)
                    if_is_ball = False
                    break # FIXME : Not currect way to position ball
                # crop_img = self.creat_circle_id(crop_img, color = "blue", cordinate_list=[cx_orange, cy_orange])
            #break
        if if_is_ball != True:
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(crop_img, str(int(fps)), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            cv2.namedWindow(f"BALL \tHit Escape to Exit")
            cv2.imshow(f"BALL \tHit Escape to Exit", crop_img)

        return frame
    
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
        
        are_of_circle_min = frame.shape[0]/180
        are_of_circle_max = frame.shape[0]/175

        are_of_circle_min = pi*are_of_circle_min**2
        are_of_circle_max = pi*are_of_circle_max**2
        
        image_copy = frame.copy()
        
        """ contours for blue area """
        for contours in contours_blue:
            blue_area = cv2.contourArea(contours)
            if blue_area < are_of_circle_max and blue_area > are_of_circle_min:
                moment = cv2.moments(contours) # NOTE: check me again 
                cx_blue = int(moment["m10"]/moment["m00"])
                cy_blue = int(moment["m01"]/moment["m00"])
                cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                crop_img = self.crop_robot_circle(frame, cy_blue, cx_blue, if_is_ball)
                # crop_img = self.creat_circle_id(crop_img, color = "blue", cordinate_list=[cx_blue, cy_blue])
        
        '''
        # B, G, R channel splitting
        blue, green, red = cv2.split(frame)
        contours1, hierarchy1 = cv2.findContours(image=blue, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        image_contour_blue = blue.copy()
        cv2.drawContours(image=image_contour_blue, contours=contours1, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        '''
        return crop_img
            
    def detect_robot_id_HSV_scale(self, frame: cv2.VideoCapture.read):
        # Contants:
        robot_num  = 1
        b_json     = False
        if_is_ball = False

        frame_hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        try:
        # try to load the json file if exist
            with open("Robo_Color_Config.json") as color_config:
                color_range = json.load(color_config)
            b_json = True
        # Catch Err in this case might be naming diff in json file and print defined
        except Exception as e:
            b_json = False
            color_range = None
            print(e)


        if b_json == True:
            try:
                # Source: https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
                # Color: blue
                low_blue             = np.array(color_range["Low_Blue"], np.uint8)
                upper_blue           = np.array(color_range["Up_Blue"], np.uint8) 
                        
                # define masks
                mask_blue            = cv2.inRange(frame_hsv, low_blue ,upper_blue)

                contours_blue        = cv2.findContours(mask_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                contours_blue        = imutils.grab_contours(contours_blue)
            except Exception as e:
                # Color: blue
                low_blue             = np.array([90, 150, 0], np.uint8)
                upper_blue           = np.array([140, 255, 255], np.uint8)

                # define masks
                mask_blue            = cv2.inRange(frame_hsv, low_blue ,upper_blue)

                contours_blue        = cv2.findContours(mask_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                contours_blue        = imutils.grab_contours(contours_blue)
                b_json = False
                print(e)

        try:     
            cx_blue = 0 
            cy_blue = 0 
            moment = 0
            i = 0
            thread_list = []
            
            are_of_circle_min = frame.shape[0]/200
            are_of_circle_max = frame.shape[0]/100

            are_of_circle_min = pi*are_of_circle_min**2
            are_of_circle_max = pi*are_of_circle_max**2
            
            if Image_Processing.PRINT_DEBUG:
                print(f"blue_are_of_circle_min {are_of_circle_min}")
                print(f"blue_are_of_circle_max {are_of_circle_max}")


            """ contours for blue area """
            for contours in contours_blue:
                blue_area = cv2.contourArea(contours)
                if Image_Processing.PRINT_DEBUG:
                    print(f"blue_are {blue_area}")
                if blue_area < are_of_circle_max and blue_area > are_of_circle_min:
                    
                    if Image_Processing.MASK_COLOR_THRESHOLD:
                        frame[mask_blue > 0] = (255, 0 , 0)
                        
                    moment = cv2.moments(contours) # NOTE: check me again 
                    i += 1
                    cx_blue = int(moment["m10"]/moment["m00"])
                    cy_blue = int(moment["m01"]/moment["m00"])
                    crop_img = self.crop_robot_circle(frame, cy_blue, cx_blue, if_is_ball)
                    crop_img = self.creat_circle_id(crop_img, color = "blue", cordinate_list=[cx_blue, cy_blue])
                    # FIXME: Change for testing orientation

                    if Image_Processing.CENTER_CORDINATE:
                        cv2.line(crop_img, (0 , int(crop_img.shape[0]/2)), (crop_img.shape[0], int(crop_img.shape[0]/2)), (0, 0, 0), thickness=1, lineType=1)
                        cv2.line(crop_img, (int(crop_img.shape[0]/2) , 0), (int(crop_img.shape[0]/2), crop_img.shape[0]), (0, 0, 0), thickness=1, lineType=1)
                    
                    crop_img1 = np.mean(crop_img, axis=2)
                    hspace, angles, distances = hough_line(crop_img1)
                    angle = []
                    for _, a , distances in zip(*hough_line_peaks(hspace, angles, distances)):
                        angle.append(a)

                    angles = [a*180/np.pi for a in angle]
                    angle_difference = np.max(angles) - np.min(angles)
                    # print(f"angle_difference: {angle_difference}")
                    # self.check_if_robot(crop_img, robot_num, frame, cy_blue, cx_blue)
                    # self.detect_robot_location(cy_blue, cx_blue, robot_num) 
                    robot_num += 1
                    
                    # thread_list.append(threads)
                    # thread_list.update(threads)
                    if Image_Processing.CAPTURE_ONE_ROBOT_IMAGE:
                        SSL_DetectionRobot = self.check_if_robot(crop_img, robot_num, frame, cy_blue, cx_blue, color_range)
                        return frame, SSL_DetectionRobot
                    else:
                        SSL_DetectionRobot = self.check_if_robot(crop_img, robot_num, frame, cy_blue, cx_blue, color_range)
        
        except Exception as e:
            print(e)                
        return frame, SSL_DetectionRobot

    def detect_robot_orientation(self, frame: cv2.VideoCapture.read):
        counter = 0
        (dX, dY) = (0, 0)
        direction = ""

        ap = argparse.ArgumentParser()

        ap.add_argument("-v", "--video",
            help="path to the (optional) video file")
        ap.add_argument("-b", "--buffer", type=int, default=32,
            help="max buffer size")

        args = vars(ap.parse_args())
        pts = deque(maxlen=args["buffer"])
        # Color: blue
        low_blue        = (132, 90, 0)
        upper_blue      = (180, 255, 92)

        mask = cv2.inRange(frame, low_blue, upper_blue)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2) 

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)
        # loop over the set of tracked points
        for i in np.arange(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue
            # check to see if enough points have been accumulated in
            # the buffer
            if counter >= 10 and i == 1 and pts[-10] is not None:
                # compute the difference between the x and y
                # coordinates and re-initialize the direction
                # text variables
                dX = pts[-10][0] - pts[i][0]
                dY = pts[-10][1] - pts[i][1]
                (dirX, dirY) = ("", "")
                # ensure there is significant movement in the
                # x-direction
                if np.abs(dX) > 20:
                    dirX = "East" if np.sign(dX) == 1 else "West"
                # ensure there is significant movement in the
                # y-direction
                if np.abs(dY) > 20:
                    dirY = "North" if np.sign(dY) == 1 else "South"
                # handle when both directions are non-empty
                if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)
                # otherwise, only one direction is non-empty
                else:
                    direction = dirX if dirX != "" else dirY
                
                    # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        # show the movement deltas and the direction of movement on
        # the frame
        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 3)
        cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)

        return frame

    def detect_robot_location(self, y: int, x: int, robo_num: int):
        robo_pos_dic = {}

        try: 
            with open('Robo_Pos.json', 'r') as openfile:
                robo_pos_dic = json.load(openfile)
        except Exception as e:
            print(e)

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
        
    def set_camera_config(self, camera_config: json, Fps=False, Res=False, Focus=False):

        if isinstance(Fps, bool) and isinstance(Res, bool) and isinstance(Focus, bool):
            
            if Fps is not False:
                # Change FPS
                if isinstance(camera_config["FPS"], int):
                    self.camera_capture.set(cv2.CAP_PROP_FPS, camera_config["FPS"])  # set camera FPS from Json file
                    print("FPS Set")
                elif isinstance(camera_config["FPS"], float):
                    self.camera_capture.set(cv2.CAP_PROP_FPS, int(camera_config["FPS"]))  # set camera FPS from Json file
                    print("FPS Set")
                else:
                    print("FPS Configuration is incorrect")
            else:
                print("FPS is not set")

            if Res is not False:
                # self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                # self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                # Change Resolution
                if isinstance(camera_config["Resolution"], list):
                    # set camera Resolution from Json file
                    self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config["Resolution"][0]))
                    self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config["Resolution"][1]))
                    print("Resolution Set")
                else:
                    print("Resolution Configuration is incorrect")
            else:
                print("Resolution is not Set")

            if Focus is not False:
                # Change Focus
                if isinstance(camera_config["focus"], int):
                    # set camera Resolution from Json file
                    self.camera_capture.set(cv2.CAP_PROP_FOCUS, camera_config["focus"])
                    print("focus Set")
                elif isinstance(camera_config["focus"], float):
                    # set camera Resolution from Json file
                    # self.camera_capture.set(cv2.CAP_PROP_FOCUS, int(camera_config["focus"])) // This may not work
                    self.camera_capture.set(28, int(camera_config["focus"]))
                    print("focus Set")
                else:
                    print("Focus Configuration is incorrect")
            else:
                print("Focus is not Set")
        else:
            print("Set Boolean value for Camera filter setting ")
            
    def set_image_filter(self, frame  : cv2.VideoCapture.read, filter_frame = None, Blur  : bool = False,GaussianBlur  : bool = False , Segmentation : bool  = False , Res : bool = False):
        if filter_frame != None :
            ''' Blur Image '''
            if Blur is not False:
                frame = cv2.blur(src=frame, ksize=(filter_frame["Blur"][0], filter_frame["Blur"][1]))
                print("Blur is applied")
            

            ''' Blured Image '''
            if GaussianBlur is not False:
                frame = cv2.GaussianBlur(frame, (filter_frame["GaussianBlur"][0], filter_frame["GaussianBlur"][1]), 0)
                print("GaussianBlur is applied")
            
            
            ''' Segmentation '''
            if Segmentation is not False:
                print("Segmentation is applied")
            
                # reshape the image to a 2D array of pixels and 3 color values (RGB) to push it in to the kmeans
                new_frame = np.float32(frame.reshape((-1, 3)))

                # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
                # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

                _, labels, centers = cv2.kmeans(new_frame, filter_frame["Segmentation"], None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
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
            if Res is not False:
                try:
                    frame = cv2.resize(frame, (filter_frame["resize_frame"][0], filter_frame["resize_frame"][1]))
                except Exception as e:
                    print(f'Could not resize image {e}')
        else: 
            print("filter .json file is not loaded or it is corrupted")

        return frame

    def crop_robot_circle(self, img : cv2.VideoCapture.read, pos_y , pos_x, if_is_ball):
        if if_is_ball:
            # print(f"diff {pos_x} and {pos_y}")
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
            # print(f"diff {pos_x} and {pos_y}")
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

    def crop_robot_rec(self, img : cv2.VideoCapture.read, pos_y , pos_x):
        np_crop_img = np.array(img)
        pos_y   = pos_y - 15
        pos_x   = pos_x - 15
        higth   = pos_y + 30
        width   = pos_x + 30
        # crop rec image from robot
        np_crop_img  = np_crop_img[pos_y:higth, pos_x:width]
        crop_img = np_crop_img
        
        return crop_img
    
    def check_if_robot(self, img : cv2.VideoCapture.read, Robo_Num: int, field : cv2.VideoCapture.read, cy: int = None, cx :int = None, color_range = None):
        # constants
        num_of_circle   = 1
        num_of_red      = 0
        num_of_green    = 0

        b_json = False
        
        num_x_cor   = {'green' : [],
                        'red'  : []}

        circle_pack = {"1":     [],
                       "2":     [],
                       "3":     [],
                       "4":     [],
                       "prime": []}
        
        SSL_DetectionRobot  = {"robot_id"   :  0,
                              "x"           :  0,
                              "y"           :  0,
                              "orientation" :  0,
                              "pixel_x"     :  0,
                              "pixel_y"     :  0}
        
        SSL_DetectionBall  =  {"robot_id"   :  0,
                              "x"           :  0,
                              "y"           :  0,
                              "orientation" :  0,
                              "pixel_x"     :  0,
                              "pixel_y"     :  0}


        if color_range == None:
            b_json = False
        else:
            b_json = True
        
        try:
            frame_hsv       = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      
            if b_json == True:
                try:
                    # Color: Red
                    low_red         = np.array(color_range["Low_Red"], np.uint8)
                    upper_red       = np.array(color_range["Up_Red"], np.uint8)
                                    
                    # Color: green
                    low_green       = np.array(color_range["Low_Green"], np.uint8)
                    upper_green     = np.array(color_range["Up_Green"], np.uint8)

                    # Color: black
                    low_black       = np.array([0, 0, 0], np.uint8)
                    upper_black     = np.array([180, 255, 145], np.uint8)
                            
                    # define masks
                    mask_red        = cv2.inRange(frame_hsv, low_red        ,upper_red)
                    mask_green      = cv2.inRange(frame_hsv, low_green      ,upper_green)
                    mask_black      = cv2.inRange(frame_hsv, low_black      ,upper_black)

                    contours_red    = cv2.findContours(mask_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                    contours_red    = imutils.grab_contours(contours_red)

                    contours_green  = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                    contours_green  = imutils.grab_contours(contours_green)

                    contours_black  = cv2.findContours(mask_black.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                    contours_black  = imutils.grab_contours(contours_black)
                except Exception as e:
                    b_json = False
                    print(e)
            
            if b_json == False:
                # Color: Red
                low_red         = np.array([126, 40, 140], np.uint8)
                upper_red       = np.array([180, 255, 255], np.uint8)
                                
                # Color: green
                low_green       = np.array([0,  90 , 150], np.uint8)
                upper_green     = np.array([75, 255, 255], np.uint8)

                # Color: black
                low_black       = np.array([0, 0, 0], np.uint8)
                upper_black     = np.array([180, 255, 145], np.uint8)
                        
                # define masks
                mask_red        = cv2.inRange(frame_hsv, low_red        ,upper_red)
                mask_green      = cv2.inRange(frame_hsv, low_green      ,upper_green)
                mask_black      = cv2.inRange(frame_hsv, low_black      ,upper_black)

                contours_red    = cv2.findContours(mask_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                contours_red    = imutils.grab_contours(contours_red)

                contours_green  = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                contours_green  = imutils.grab_contours(contours_green)

                contours_black  = cv2.findContours(mask_black.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                contours_black  = imutils.grab_contours(contours_black)
            
            
            if Image_Processing.MASK_COLOR_THRESHOLD:
                img[mask_green > 0] = (0  , 255 , 0)
                img[mask_red   > 0] = (0, 0   , 255)

            are_of_circle_min = img.shape[0]/13
            are_of_circle_max = img.shape[0]/6

            are_of_circle_min = pi*are_of_circle_min**2
            are_of_circle_max = pi*are_of_circle_max**2

            if Image_Processing.PRINT_DEBUG:
                print(f'area_min: {are_of_circle_min}')
                print(f'area_max: {are_of_circle_max}')
            list_circle_cordinate = []             
            
            """ contours for red area  """            
            for contours in contours_red:
                red_area = cv2.contourArea(contours)
                # print(f"img.frame {img.shape}")
                if red_area < are_of_circle_max and red_area > are_of_circle_min:
                    if Image_Processing.PRINT_DEBUG:
                        print(f"red_area {red_area}")
                    moment = cv2.moments(contours) # NOTE: check me again 
                    cx_red = int(moment["m10"]/moment["m00"])
                    cy_red = int(moment["m01"]/moment["m00"])
                    list_circle_cordinate.append([cy_red,cx_red])
                    if Image_Processing.PRINT_DEBUG:
                        print(f"cx_red {cx_red}")
                        print(f"cy_red {cy_red}")
                    num_of_red    += 1
                    num_of_circle += 1
                    position = self.check_circle_position(img.shape[0], img.shape[1], cx_red, cy_red)
                    num_x_cor['red'].append([position , cx_red, cy_red])
                    for i in circle_pack:
                        if i == position:
                            circle_pack[i] = [cx_red, cy_red]
                    if Image_Processing.PRINT_DEBUG:
                        print(f'Red Position is: {position}')
                    self.creat_circle_id(img,'red', [cx_red, cy_red])
                    # cv2.circle(img, (cx_red, cy_red), radius=num_of_red, color=(255, 255, 255), thickness=-1)

            list_circle_cordinate.clear()
            """ contours for green area """             
            for contours in contours_green:
                green_area = cv2.contourArea(contours)
                if Image_Processing.PRINT_DEBUG:
                    print(f"Green_are: {green_area}")
                if green_area < are_of_circle_max and green_area > are_of_circle_min:
                    moment = cv2.moments(contours) # NOTE: check me again 
                    # print(f"green_area {green_area}")
                    cx_green = int(moment["m10"]/moment["m00"])
                    cy_green = int(moment["m01"]/moment["m00"])
                    if Image_Processing.PRINT_DEBUG:
                        print(f"cx_green {cx_green}")
                        print(f"cy_green {cy_green}")
                    num_of_green  += 1
                    num_of_circle += 1 
                    list_circle_cordinate.append([cy_green,cx_green])
                    position = self.check_circle_position(img.shape[0], img.shape[1], cx_green, cy_green)
                    for i in circle_pack:
                        if i == position:
                            if len(circle_pack[i]) > 1:
                                circle_pack["prime"] = [cx_green, cy_green]
                            else:
                                circle_pack[i] = [cx_green, cy_green]
                    if Image_Processing.PRINT_DEBUG:
                        print(f'Green Position is: {position}')
                    num_x_cor['green'].append([position , cx_green, cy_green])
                    self.creat_circle_id(img,'green', [cx_green, cy_green])
                    # cv2.circle(img, (cx_green, cy_green), radius=num_of_green, color=(255, 255, 255), thickness=-1)
                    
            a = num_x_cor['red']
            b = num_x_cor['green']
            if Image_Processing.PRINT_DEBUG:
                print(f'List Red is : {len(a)}')
                print(f'List Green is : {len(b)}')
            A = num_x_cor['green']
            B = num_x_cor['red']
            
            if Image_Processing.PRINT_DEBUG:
                print(f'num_x_cor_green : {A}')
                print(f'num_y_cor_red : {B}')
            for x in list(circle_pack.keys()):
                if circle_pack[x] == []:
                    del circle_pack[x]
            
            if Image_Processing.PRINT_DEBUG:
                print(f'circle_pack: {circle_pack}')
            
            if len(A) + len(B) == 4 :
                angle = self.getOrientation(circle_pack)
            else:
                return
            """ contours for black area              
            for contours in contours_black:
                black_area = cv2.contourArea(contours)
                if black_area < 200 and black_area > 1:
                    cv2.drawContours(img, [contours], -1, (255,255,255), 1)
                    moment = cv2.moments(contours) # NOTE: check me again 
                    
                    cx_green = int(moment["m10"]/moment["m00"])
                    cy_green = int(moment["m01"]/moment["m00"])
                    
                    # cv2.circle(frame, (cx_green, cy_green), 1, (255, 255, 255), -1)
                    # cv2.putText(frame, "green", (cx_green, cy_green), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
                    # crop_img[mask_green > 0] = (0, 255 , 0)
            """
            if Image_Processing.ROTATE_ROBOT_IMAGE :
                (h, w) = img.shape[:2]
                M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[1] // 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
                ROBOT_ID = self.match_robot(img)
            else: 
                imgRotate = img 
                (h, w) = imgRotate.shape[:2]
                M = cv2.getRotationMatrix2D((imgRotate.shape[1] // 2, imgRotate.shape[1] // 2), angle, 1.0)
                imgRotate = cv2.warpAffine(imgRotate, M, (w, h))
                ROBOT_ID = self.match_robot(imgRotate)
        
            SSL_DetectionRobot  = {"robot_id"    :  ROBOT_ID,
                                   "x"           :  0,
                                   "y"           :  0,
                                   "orientation" :  angle,
                                   "pixel_x"     :  cx,
                                   "pixel_y"     :  cy}

            self.saveRobotImage(frame= img, robot_num= Robo_Num,cx= cx, cy= cy )
            print(f'ROBOT_ID: {ROBOT_ID}')
        except Exception as e:
            print(e)
            
        if Image_Processing.SHOW_ROBOTS_IN_NEW_WINDOW:
            if ROBOT_ID != None:
                cTime = time.time()
                fps = 1 / (cTime - self.pTime)
                self.pTime = cTime
                cv2.putText(img, str(int(fps)), (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                cv2.namedWindow(f"RobotSoccer Robot{ROBOT_ID}\tHit Escape to Exit")
                cv2.imshow(f"RobotSoccer Robot{ROBOT_ID}\tHit Escape to Exit", img)
            else:
                return ROBOT_ID
       
        return SSL_DetectionRobot     

    def check_circle_position(self,img_shape_x, img_shape_y, x, y):
        if x >= img_shape_x//2:
            if y <= img_shape_y//2:
                return "1"
        
        if x <= img_shape_x//2:
            if y <= img_shape_y//2:
                return "2"
    
        if x <= img_shape_x//2:
            if y >= img_shape_y//2:
                return "3"

        if x >= img_shape_x//2:
            if y >= img_shape_y//2:
                return "4"

    def robot_id_detection(self, num_of_green:int = None, num_of_red:int = None, Robo_Num : int = None, robot_id: list = None):
        if num_of_green == 2 and num_of_red == 2:
            print("Num_Green = 2  Num_Red = 2")
            # return robot_id[1], robot_id[3], robot_id[5], robot_id[10], robot_id[11], robot_id[7]
        
        if num_of_green == 3 and num_of_red == 1:
            print("Num_Green = 3  Num_Red = 1")
            # return robot_id[2], robot_id[6], robot_id[12], robot_id[14]
        
        if num_of_green == 1 and num_of_red == 3:
            print("Num_Green = 1  Num_Red = 3")

    def convert_pixel_to_centimeter(self, xVal = None, yVal = None):  
        x = xVal * self.xCoef
        y = yVal * self.yCoef
            
        return x, y

    def creat_circle_id(self, frame: cv2.VideoCapture.read, color: str = None, cordinate_list :list = None):
        if Image_Processing.CIRCLE_ID_COLOR_BY_CORDINATE:
            # Center coordinates
            center_coordinates = [int(frame.shape[0]/2), int(frame.shape[1]/2)]
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

    def getOrientation(self, circle_pack = None ):
        length_list = {}
        if 'prime' in circle_pack.keys():
            circle_keys = list(circle_pack.keys())
            # circle_keys = 1
            print(circle_pack.keys())
            length_list.update({ f"{circle_keys[0]}-{circle_keys[1]}" : math.sqrt((circle_pack[circle_keys[0]][0] - circle_pack[circle_keys[1]][0])**2 + (circle_pack[circle_keys[0]][1] - circle_pack[circle_keys[1]][1])**2) })
            length_list.update({ f"{circle_keys[0]}-{circle_keys[2]}" : math.sqrt((circle_pack[circle_keys[0]][0] - circle_pack[circle_keys[2]][0])**2 + (circle_pack[circle_keys[0]][1] - circle_pack[circle_keys[2]][1])**2) })
            length_list.update({ f"{circle_keys[0]}-{circle_keys[3]}" : math.sqrt((circle_pack[circle_keys[0]][0] - circle_pack[circle_keys[3]][0])**2 + (circle_pack[circle_keys[0]][1] - circle_pack[circle_keys[3]][1])**2) })
            length_list.update({ f"{circle_keys[1]}-{circle_keys[2]}" : math.sqrt((circle_pack[circle_keys[1]][0] - circle_pack[circle_keys[2]][0])**2 + (circle_pack[circle_keys[1]][1] - circle_pack[circle_keys[2]][1])**2) })
            length_list.update({ f"{circle_keys[1]}-{circle_keys[3]}" : math.sqrt((circle_pack[circle_keys[1]][0] - circle_pack[circle_keys[3]][0])**2 + (circle_pack[circle_keys[1]][1] - circle_pack[circle_keys[3]][1])**2) })
            length_list.update({ f"{circle_keys[2]}-{circle_keys[3]}" : math.sqrt((circle_pack[circle_keys[2]][0] - circle_pack[circle_keys[3]][0])**2 + (circle_pack[circle_keys[2]][1] - circle_pack[circle_keys[3]][1])**2) })
            if Image_Processing.PRINT_DEBUG:
                print("##############################")
                print(f'len_all:   {length_list}')
                print("##############################")
        else:
            length_list.update({ "1-2" : math.sqrt((circle_pack["1"][0] - circle_pack["2"][0])**2 + (circle_pack["1"][1] - circle_pack["2"][1])**2) })
            length_list.update({ "1-3" : math.sqrt((circle_pack["1"][0] - circle_pack["3"][0])**2 + (circle_pack["1"][1] - circle_pack["3"][1])**2) })
            length_list.update({ "1-4" : math.sqrt((circle_pack["1"][0] - circle_pack["4"][0])**2 + (circle_pack["1"][1] - circle_pack["4"][1])**2) })
            length_list.update({ "2-3" : math.sqrt((circle_pack["2"][0] - circle_pack["3"][0])**2 + (circle_pack["2"][1] - circle_pack["3"][1])**2) })
            length_list.update({ "2-4" : math.sqrt((circle_pack["2"][0] - circle_pack["4"][0])**2 + (circle_pack["2"][1] - circle_pack["4"][1])**2) })
            length_list.update({ "3-4" : math.sqrt((circle_pack["3"][0] - circle_pack["4"][0])**2 + (circle_pack["3"][1] - circle_pack["4"][1])**2) })
            if Image_Processing.PRINT_DEBUG:
                print("##############################")
                print(f'len_one:   {length_list["1-2"]}')
                print(f'len_two:   {length_list["1-3"]}')
                print(f'len_three: {length_list["1-4"]}')
                print(f'len_four:  {length_list["2-3"]}')
                print(f'len_five:  {length_list["2-4"]}')
                print(f'len_six:   {length_list["3-4"]}')
                print("##############################")
        
        min_len = min(length_list, key=length_list.get)
        length_list = sorted(length_list)
        if 'prime' in circle_pack.keys():
            min_length_list = [min_len[:1], min_len[-5:]]
        else:
            min_length_list = [min_len[:1], min_len[-1:]]
        # print(f'min_length_list : {min_length_list}')
        # print(f'min_len: {min_len}')
        
        angle = self.angle_between_circle(min_length_list, circle_pack)
        print(f'The final Angle is : {angle}')
        # Image_Processing.ANGLE = angle
        return angle

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

    def saveRobotImage(self, frame, robot_num, cx, cy):
        if Image_Processing.SAVE_ROBOT_IMAGE == True:
            try:
                if frame is not None:
                    cv2.imwrite(f"Robot{robot_num}_x{cx}_y{cy}.jpg", frame)
                else:
                    print("crop image is not valid")
            except Exception as e:
                print(e)
        else:
            pass

    def gradient(self, pt1 ,pt2):
        return(pt2[1] - pt1[1])/ (pt2[0] - pt1[0]) 

    def getAngle(self, PT1 = None, PT2 = None, PT3 = None):
        pt1 = PT1
        pt2 = PT2
        pt3 = PT3
        print(pt1, pt2, pt3)
        n1 = self.gradient(pt1 , pt2)
        n2 = self.gradient(pt1 ,pt3)
        angleR = math.atan((n2 - n1)/(1 + (n2*n1)))
        angleD = round(math.degrees(angleR))
        return angleD

    def angle_between_circle(self, lst_min_line_order = None, lst_min_line =  None):

        print(lst_min_line_order)
        print(lst_min_line)

        ''' Durchmesser ist Ungültig '''
        if lst_min_line_order[0] == '1' and lst_min_line_order[0] == '3':
            return None
        if lst_min_line_order[0] == '3' and lst_min_line_order[0] == '1':
            return None
        if lst_min_line_order[0] == '2' and lst_min_line_order[0] == '4':
            return None
        if lst_min_line_order[0] ==  '4' and lst_min_line_order[0] == '2':
            return None

        ''' first assumption '''
        # # first assumption # #
        if lst_min_line_order[0] == "1" and lst_min_line_order[1] == "2":
            print("first assumption 1")
            if lst_min_line["1"][0] > lst_min_line["2"][0]:
                if lst_min_line["1"][1] > lst_min_line["2"][1]:
                    print("1 is right and down")
                    print("2 is left and up")
                    Angle = self.getAngle(PT1 = [lst_min_line["2"][0], lst_min_line["2"][1]], # 
                                    PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                    PT3 = [lst_min_line["1"][0], lst_min_line["2"][1]])
                    Angle =  ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle
                else:
                    print("1 is right and up")
                    print("2 is left and down")
                    Angle = self.getAngle(PT1 = [lst_min_line["2"][0], lst_min_line["2"][1]], # 
                                    PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                    PT3 = [lst_min_line["1"][0], lst_min_line["2"][1]])
                    Angle =  ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle

            else:
                print("THIS CONFIG IS NOT POSSIBLE !!!")

        if lst_min_line_order[1] == "1" and lst_min_line_order[0] == "2":
            print("first assumption 2")
            if lst_min_line["1"][0] > lst_min_line["2"][0]:
                if lst_min_line["1"][1] > lst_min_line["2"][1]:
                    print("1 is right and down")
                    print("2 is left and up")
                    Angle = self.getAngle(PT1 = [lst_min_line["2"][0], lst_min_line["2"][1]], # 
                                    PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                    PT3 = [lst_min_line["1"][0], lst_min_line["2"][1]])
                    Angle =  ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle
                else:
                    print("1 is right and up")
                    print("2 is left and down")
                    Angle = self.getAngle(PT1 = [lst_min_line["2"][0], lst_min_line["2"][1]], # 
                                    PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                    PT3 = [lst_min_line["1"][0], lst_min_line["2"][1]])
                    Angle =  ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle

            else:
                print("THIS CONFIG IS NOT POSSIBLE !!!")


        ''' second assumption '''
        # # second assumption # #
        if lst_min_line_order[0] == "3" and lst_min_line_order[1] == "2":
            print("second assumption 1")
            if lst_min_line["2"][0] < lst_min_line["3"][0]:
                if lst_min_line["2"][1] < lst_min_line["3"][1]:
                    print("2 is up and back")
                    print("3 is down and frot")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["3"][0], lst_min_line["3"][1]], # 
                                    PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                    PT3 = [lst_min_line["2"][0], lst_min_line["3"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                    
                    
                else:
                    print("2 is down and back")
                    print("3 is up and front")
                    print("THIS CONFIG IS NOT POSSIBLE !!!")
                    
            else:
                if lst_min_line["2"][1] < lst_min_line["3"][1]:
                    print("2 is up and front")
                    print("3 is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["3"][0], lst_min_line["3"][1]], # 
                                    PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                    PT3 = [lst_min_line["2"][0], lst_min_line["3"][1]])
                    Angle = 90 - Angle + 90
                    print(Angle)
                    return Angle
                

        if lst_min_line_order[1] == "3" and lst_min_line_order[0] == "2":
            print("second assumption 2")
            if lst_min_line["2"][0] < lst_min_line["3"][0]:
                if lst_min_line["2"][1] < lst_min_line["3"][1]:
                    print("2 is up and back")
                    print("3 is down and frot")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["3"][0], lst_min_line["3"][1]], # 
                                    PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                    PT3 = [lst_min_line["2"][0], lst_min_line["3"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                    
                    
                else:
                    print("2 is down and back")
                    print("3 is up and front")
                    print("THIS CONFIG IS NOT POSSIBLE !!!")
                    
            else:
                if lst_min_line["2"][1] < lst_min_line["3"][1]:
                    print("2 is up and front")
                    print("3 is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["3"][0], lst_min_line["3"][1]], # 
                                    PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                    PT3 = [lst_min_line["2"][0], lst_min_line["3"][1]])
                    Angle = 90 - Angle + 90
                    print(Angle)
                    return Angle
                
        ''' third assumption '''
        # # third assumption # #
        if lst_min_line_order[0] == "3" and lst_min_line_order[1] == "4":
            print("third assumption 1")
            if lst_min_line["3"][0] < lst_min_line["4"][0]:
                if lst_min_line["3"][1] < lst_min_line["4"][1]:
                    print("3 is up and back")
                    print("4 is down and front")
                    # TODO: Check the 10 degree error
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                    PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                    PT3 = [lst_min_line["3"][0], lst_min_line["4"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                else:
                    print("3 is down and back")
                    print("4 is up and front")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                    PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                    PT3 = [lst_min_line["3"][0], lst_min_line["4"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
            else:
                print("THIS IS NOT POSSIBLE!!")

        if lst_min_line_order[1] == "3" and lst_min_line_order[0] == "4":
            print("third assumption 2")
            if lst_min_line["3"][0] < lst_min_line["4"][0]:
                if lst_min_line["3"][1] < lst_min_line["4"][1]:
                    print("3 is up and back")
                    print("4 is down and front")
                    # TODO: Check the 10 degree error
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                    PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                    PT3 = [lst_min_line["3"][0], lst_min_line["4"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                else:
                    print("3 is down and back")
                    print("4 is up and front")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                    PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                    PT3 = [lst_min_line["3"][0], lst_min_line["4"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
            else:
                print("THIS IS NOT POSSIBLE!!")
                    
        ''' fourth assumption '''
        # # fourth assumption # #
        if lst_min_line_order[0] == "4"  and lst_min_line_order[1] == "1":
            print("fourth assumption 1")
            if lst_min_line["1"][0] > lst_min_line["4"][0]:
                if lst_min_line["1"][1] < lst_min_line["4"][1]:
                    print("1 is up and front")
                    print("4 is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                    PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                    PT3 = [lst_min_line["1"][0], lst_min_line["4"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("1 is up and back")
                print("4 is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                    PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                    PT3 = [lst_min_line["1"][0], lst_min_line["4"][1]])
                Angle = ( Angle + 90 + 90 ) * -1
                print(Angle)
                return Angle
        
        if lst_min_line_order[1] == "4" and lst_min_line_order[0] == "1":
            print("fourth assumption 2")
            if lst_min_line["1"][0] > lst_min_line["4"][0]:
                if lst_min_line["1"][1] < lst_min_line["4"][1]:
                    print("1 is up and front")
                    print("4 is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                    PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                    PT3 = [lst_min_line["1"][0], lst_min_line["4"][1]])
                    Angle = Angle * -1
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("1 is up and back")
                print("4 is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [lst_min_line["4"][0], lst_min_line["4"][1]], # 
                                    PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                    PT3 = [lst_min_line["1"][0], lst_min_line["4"][1]])
                Angle = ( Angle + 90 + 90 ) * -1
                print(Angle)
                return Angle
            
        ''' fifth assumption '''
        # # fifth assumption # #
        if lst_min_line_order[0] == "1"  and lst_min_line_order[1] == "prime":
            print("fifth assumption 1")
            if lst_min_line["1"][0] > lst_min_line["prime"][0]:
                if lst_min_line["1"][1] > lst_min_line["prime"][1]:
                    print("1 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                          PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                          PT3 = [lst_min_line["1"][0], lst_min_line["prime"][1]])
                    Angle = ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("1 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                      PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                      PT3 = [lst_min_line["1"][0], lst_min_line["prime"][1]])
                Angle = ( 180 + Angle ) * -1
                print(Angle)
                return Angle
            
        if lst_min_line_order[0] == "prime"  and lst_min_line_order[1] == "1":
            print("fifth assumption 2")
            if lst_min_line["1"][0] > lst_min_line["prime"][0]:
                if lst_min_line["1"][1] > lst_min_line["prime"][1]:
                    print("1 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                          PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                          PT3 = [lst_min_line["1"][0], lst_min_line["prime"][1]])
                    Angle = ( 180 + Angle ) * -1
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("1 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                      PT2 = [lst_min_line["1"][0], lst_min_line["1"][1]], #
                                      PT3 = [lst_min_line["1"][0], lst_min_line["prime"][1]])
                Angle = ( 180 + Angle ) * -1
                print(Angle)
                return Angle
            
        ''' seventh assumption '''
        # # seventh assumption # #
        if lst_min_line_order[0] == "2"  and lst_min_line_order[1] == "prime":
            print("seventh assumption 1")
            if lst_min_line["2"][0] > lst_min_line["prime"][0]:
                if lst_min_line["2"][1] < lst_min_line["prime"][1]:
                    print("2 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                          PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                          PT3 = [lst_min_line["2"][0], lst_min_line["prime"][1]])
                    Angle = ( 90 - Angle ) + 90
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("2 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                      PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                      PT3 = [lst_min_line["2"][0], lst_min_line["prime"][1]])
                Angle = ( 90 - Angle ) + 90
                print(Angle)
                return Angle
            
        if lst_min_line_order[0] == "prime"  and lst_min_line_order[1] == "2":
            print("seventh assumption 2")
            if lst_min_line["2"][0] > lst_min_line["prime"][0]:
                if lst_min_line["2"][1] < lst_min_line["prime"][1]:
                    print("2 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                          PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                          PT3 = [lst_min_line["2"][0], lst_min_line["prime"][1]])
                    Angle = ( 90 - Angle ) + 90
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("2 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                      PT2 = [lst_min_line["2"][0], lst_min_line["2"][1]], #
                                      PT3 = [lst_min_line["2"][0], lst_min_line["prime"][1]])
                Angle = ( 90 - Angle ) + 90
                print(Angle)
                return Angle
            
        ''' Eighth assumption '''
        # # Eighth assumption # #
        if lst_min_line_order[0] == "3"  and lst_min_line_order[1] == "prime":
            print("Eighth assumption 1")
            if lst_min_line["3"][0] > lst_min_line["prime"][0]:
                if lst_min_line["3"][1] < lst_min_line["prime"][1]:
                    print("3 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                          PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                          PT3 = [lst_min_line["prime"][0], lst_min_line["3"][1]])
                    Angle = Angle * -1 
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("3 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                      PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                      PT3 = [lst_min_line["3"][0], lst_min_line["prime"][1]])
                # Angle = ( 90 + Angle ) + 90
                Angle = Angle * -1 
                print(Angle)
                return Angle
            
        if lst_min_line_order[0] == "prime"  and lst_min_line_order[1] == "3":
            print("Eighth assumption 2")
            if lst_min_line["3"][0] > lst_min_line["prime"][0]:
                if lst_min_line["3"][1] < lst_min_line["prime"][1]:
                    print("3 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                          PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                          PT3 = [lst_min_line["3"][0], lst_min_line["prime"][1]])
                    Angle = Angle * -1 
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("3 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                      PT2 = [lst_min_line["3"][0], lst_min_line["3"][1]], #
                                      PT3 = [lst_min_line["3"][0], lst_min_line["prime"][1]])
                Angle = Angle * -1 
                print(Angle)
                return Angle
            
        ''' Ninth assumption '''
        # # Ninth assumption # #
        if lst_min_line_order[0] == "4"  and lst_min_line_order[1] == "prime":
            print("Ninth assumption 1")
            if lst_min_line["4"][0] > lst_min_line["prime"][0]:
                if lst_min_line["4"][1] < lst_min_line["prime"][1]:
                    print("4 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                          PT2 = [lst_min_line["4"][0], lst_min_line["4"][1]], #
                                          PT3 = [lst_min_line["4"][0], lst_min_line["prime"][1]])
                    Angle = Angle * -1 
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("4 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                      PT2 = [lst_min_line["4"][0], lst_min_line["4"][1]], #
                                      PT3 = [lst_min_line["4"][0], lst_min_line["prime"][1]])
                # Angle = ( 90 + Angle ) + 90
                Angle = Angle * -1 
                print(Angle)
                return Angle
            
        if lst_min_line_order[0] == "prime"  and lst_min_line_order[1] == "4":
            print("Ninth assumption 2")
            if lst_min_line["4"][0] > lst_min_line["prime"][0]:
                if lst_min_line["4"][1] < lst_min_line["prime"][1]:
                    print("4 is up and front")
                    print("prime is down and back")
                    # FIXED
                    Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                          PT2 = [lst_min_line["4"][0], lst_min_line["4"][1]], #
                                          PT3 = [lst_min_line["4"][0], lst_min_line["prime"][1]])
                    Angle = Angle * -1 
                    print(Angle)
                    return Angle
                else:
                    print("THIS IS NOT POSSIBLE!!")
            else:
                print("4 is up and back")
                print("prime is down and front")
                # FIXED
                Angle = self.getAngle(PT1 = [lst_min_line["prime"][0], lst_min_line["prime"][1]], # 
                                      PT2 = [lst_min_line["4"][0], lst_min_line["4"][1]], #
                                      PT3 = [lst_min_line["4"][0], lst_min_line["prime"][1]])
                Angle = Angle * -1 
                print(Angle)
                return Angle
            
        return None
    
    def match_robot(self, img):
        # constants
        num_of_circle   = 0
        num_of_red      = 0
        num_of_green    = 0
        
        num_x_cor   = {'green' : [],
                        'red'  : []}
        
        circle_pack = {"1":     [],
                       "2":     [],
                       "3":     [],
                       "4":     []}
        
        frame_hsv       = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        low_red         = np.array([0, 26, 104], np.uint8)
        upper_red       = np.array([11, 255, 255], np.uint8)
                        
        # Color: green
        low_green       = np.array([30,  175 , 140], np.uint8)
        upper_green     = np.array([105, 255, 255], np.uint8)

                
        # define masks
        mask_red        = cv2.inRange(frame_hsv, low_red        ,upper_red)
        mask_green      = cv2.inRange(frame_hsv, low_green      ,upper_green)

        contours_red    = cv2.findContours(mask_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_red    = imutils.grab_contours(contours_red)

        contours_green  = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_green  = imutils.grab_contours(contours_green)

        are_of_circle_min = img.shape[0]/13
        are_of_circle_max = img.shape[0]/6

        are_of_circle_min = pi*are_of_circle_min**2
        are_of_circle_max = pi*are_of_circle_max**2

        print(f'area_min: {are_of_circle_min}')
        print(f'area_max: {are_of_circle_max}')
        list_circle_cordinate = []             
        
        """ contours for red area  """            
        for contours in contours_red:
            red_area = cv2.contourArea(contours)
            # print(f"img.frame {img.shape}")
            if red_area < are_of_circle_max and red_area > are_of_circle_min:
                print(f"red_area {red_area}")
                moment = cv2.moments(contours) # NOTE: check me again 
                cx_red = int(moment["m10"]/moment["m00"])
                cy_red = int(moment["m01"]/moment["m00"])
                if cx_red > img.shape[0]/2 and cy_red < img.shape[1]/2:
                    circle_pack['1'] = "red"
                if cx_red < img.shape[0]/2 and cy_red < img.shape[1]/2:
                    circle_pack['2'] = "red"
                if cx_red < img.shape[0]/2 and cy_red > img.shape[1]/2:
                    circle_pack['3'] = "red"
                if cx_red > img.shape[0]/2 and cy_red > img.shape[1]/2:
                    circle_pack['4'] = "red"

        list_circle_cordinate.clear()
        """ contours for green area """             
        for contours in contours_green:
            green_area = cv2.contourArea(contours)
            if green_area < are_of_circle_max and green_area > are_of_circle_min:
                moment = cv2.moments(contours) # NOTE: check me again 
                print(f"green_area {green_area}")
                cx_green = int(moment["m10"]/moment["m00"])
                cy_green = int(moment["m01"]/moment["m00"])
                if cx_green > img.shape[0]/2 and cy_green < img.shape[1]/2:
                    circle_pack['1'] = "green"
                if cx_green < img.shape[0]/2 and cy_green < img.shape[1]/2:
                    circle_pack['2'] = "green"
                if cx_green < img.shape[0]/2 and cy_green > img.shape[1]/2:
                    circle_pack['3'] = "green"
                if cx_green > img.shape[0]/2 and cy_green > img.shape[1]/2:
                    circle_pack['4'] = "green"

        return self.loop_robot_id_list(color_pattern_list = circle_pack)
        
    def loop_robot_id_list(self, color_pattern_list: dict = None):
        Robot_Pattern_Dict= {
        "Robo1"  : {'1': 'red'  , '2': 'red'  , '3': 'green', '4': 'red'},
        "Robo2"  : {'1': 'red'  , '2': 'green', '3': 'green', '4': 'red'},
        "Robo3"  : {'1': 'green', '2': 'green', '3': 'green', '4': 'red'},
        "Robo4"  : {'1': 'green', '2': 'red'  , '3': 'green', '4': 'red'},
        "Robo5"  : {'1': 'red'  , '2': 'red'  , '3': 'red'  , '4': 'green'},
        "Robo6"  : {'1': 'red'  , '2': 'green', '3': 'red'  , '4': 'green'},
        "Robo7"  : {'1': 'green', '2': 'green', '3': 'red'  , '4': 'green'},
        "Robo8"  : {'1': 'green', '2': 'red'  , '3': 'red'  , '4': 'green'},
        "Robo9"  : {'1': 'green', '2': 'green', '3': 'green', '4': 'green'},
        "Robo10" : {'1': 'red'  , '2': 'red'  , '3': 'red'  , '4': 'red'  },
        "Robo11" : {'1': 'red'  , '2': 'red'  , '3': 'green', '4': 'green'},
        "Robo12" : {'1': 'green', '2': 'green', '3': 'red'  , '4': 'red'  },
        "Robo13" : {'1': 'red'  , '2': 'green', '3': 'green', '4': 'green'},
        "Robo14" : {'1': 'red'  , '2': 'green', '3': 'red'  , '4': 'red'  },
        "Robo15" : {'1': 'green', '2': 'red'  , '3': 'green', '4': 'green'},
        "Robo16" : {'1': 'green', '2': 'red'  , '3': 'red'  , '4': 'red'  }}
        
        for id in Robot_Pattern_Dict:
            if Robot_Pattern_Dict[id] == color_pattern_list:
                return id
            
        return None
        
    
    def finish_capturing(self):
        cv2.destroyAllWindows()

# imgProc = Image_Processing()
