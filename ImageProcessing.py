import json
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
from threading import Thread

#from HSV_Color_Picker import *


# This class receive Images from the ControlCommand
# It processes images to detect the robot ID, origination
# and location. The return the data back as needed
class ImageProcessing():
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
    MASK_COLOR_THRESHOLD            = False
    CIRCLE_ID_COLOR_BY_CORDINATE    = True
    CENTER_CORDINATE                = False

    def __init__(self):
        
        self.camera_capture = cv2.VideoCapture(0)

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
        """ Start capturing get camera config from start_capturing_command """
        self.set_camera_config(camera_config, Fps=False, Res=False, Focus=False)

        # Set Pixel Value
        try:
            self.xFrameSizePixel = int(camera_config["resize_frame"][0])
            self.yFrameSizePixel = int(camera_config["resize_frame"][1])
            self.xCoef = ImageProcessing.Field_Size_X_Direction/self.xFrameSizePixel
            self.yCoef = ImageProcessing.Field_Size_Y_Direction/self.yFrameSizePixel
        except Exception as e: 
            print(e)

        # Creat windows for the frame
        cv2.namedWindow("RobotSoccer\tHit Escape to Exit")
        
        while True:
        
            # ret, frame = self.camera_capture.read() # FIXME: Changed to load Image
            # frame = cv2.imread("FieldTest_Left_Light_On_Daylight(hight).jpg")
            frame = cv2.imread("FieldTest_Left_Light_On_Daylight(hight).jpg")
            # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            # frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            # self.detect_robot_orientation(frame=frame)
            # Aplied Filter GaussianBlur and Segmentation
            frame = self.set_image_filter(frame = frame, Blur= False,  GaussianBlur = False, Segmentation = False, Res = False)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
            
            # Detect Robot 
            frame =  self.detect_robot_id(frame = frame)
            
            # FIXME changed for working on images
            # if not ret:
            #     print("failed to grab frame")
            #     break
            
            
            # print(frame.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            # print(frame.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
            # Show frame rate
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(frame, str(int(fps)), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            # cv2.putText(blurred_frame, str(int(fps)), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # cv2.imshow("B&W RobotSoccer\tHit Escape to Exit", mask)  # BLack and White Image
            # cv2.imshow("RobotSoccer\tHit Escape to Exit", np.vstack(result))  # Normal Images with contours
            # cv2.imshow("RobotSoccer\tHit Escape to Exit", reduced)

            # cv2.imshow("RobotSoccer\tHit Escape to Exit", frame)
            # cv2.waitKey(1)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

    def detect_robot_id(self, frame: cv2.VideoCapture.read):
        # Contants:
        robot_num = 1
        try:
            frame_hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Source: https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
            
            # Color: blue
            low_blue    = np.array([90, 150, 0], np.uint8)
            upper_blue  = np.array([140, 255, 255], np.uint8)        
            
            # Color: Red
            # low_red     = np.array([110, 50, 120], np.uint8)
            # upper_red   = np.array([250, 255, 255], np.uint8)
                    
            # Color: yellow
            # low_yellow      = np.array([18, 70, 120], np.uint8)
            # upper_yellow    = np.array([15, 255, 255], np.uint8)
            
            # Color: green
            # low_green      = np.array([60, 0, 150], np.uint8)
            # upper_green    = np.array([80, 230, 255], np.uint8)
                    
            # Color: Orange
            # low_orange     = np.array([10, 100, 20], np.uint8)
            # upper_orange   = np.array([25, 255, 255], np.uint8)

            # Color: black
            # low_black       = np.array([0, 0, 0], np.uint8)
            # upper_black     = np.array([180, 255, 145], np.uint8)
                    
            
            
            # define masks
            mask_blue   = cv2.inRange(frame_hsv, low_blue       ,upper_blue)
            # mask_red    = cv2.inRange(frame_hsv, low_red        ,upper_red)
            # mask_yellow = cv2.inRange(frame_hsv, low_yellow     ,upper_yellow)
            # mask_green  = cv2.inRange(frame_hsv, low_green      ,upper_green)
            # mask_orange = cv2.inRange(frame_hsv, low_orange     ,upper_orange)
            # mask_black  = cv2.inRange(frame_hsv, low_black     ,upper_black)


            
            # CHAIN_APPROX_NONE gibe all points
            # contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            # cv2.drawContours(mask, contours, -1, (0, 0, 0), 1)  # -1 means all the counters
            contours_blue       = cv2.findContours(mask_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            contours_blue       = imutils.grab_contours(contours_blue)

            # contours_red        = cv2.findContours(mask_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            # contours_red        = imutils.grab_contours(contours_red)

            # contours_yellow     = cv2.findContours(mask_yellow.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            # contours_yellow     = imutils.grab_contours(contours_yellow)
            # frame[mask_yellow > 0] = (0, 30 , 255)

            # contours_green      = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            # contours_green      = imutils.grab_contours(contours_green)

            # contours_black      = cv2.findContours(mask_black.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            # contours_black      = imutils.grab_contours(contours_black)
            # #frame[mask_black > 0] = (0, 0 , 0)

            # contours_orange     = cv2.findContours(mask_orange.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
            # contours_orange     = imutils.grab_contours(contours_orange)
            # frame[mask_orange > 0] = (20, 150 , 255)

            # frame = cv2.bitwise_and(result, result, mask=full_mask)
            cx_blue = 0 
            cy_blue = 0 
            moment = 0
            
            r = 100
            i = 0

            # print(f"sss : {self.xFrameSizePixel/15}")
            # print(f"sss : {self.xFrameSizePixel/22}")
            # print(f"frame shape : {frame.shape}")
            # print(f"frame shape : {frame.shape[1]/25}")
            
            """ contours for blue area """
            for contours in contours_blue:
                blue_area = cv2.contourArea(contours)
                #  / 
                if blue_area < frame.shape[1]/9 and blue_area > frame.shape[1]/37:
                    # cv2.drawContours(frame, [contours], -1, (255,255,255), 1)
                    if ImageProcessing.MASK_COLOR_THRESHOLD:
                        frame[mask_blue > 0] = (255, 0 , 0)
                    moment = cv2.moments(contours) # NOTE: check me again 
                    # cv2.drawContours(frame, contours, i, (0, 0, 255), 2)
                    i += 1
                    # Find the orientation of each shape
                    # self.getOrientation(contours, frame)

                    cx_blue = int(moment["m10"]/moment["m00"])
                    cy_blue = int(moment["m01"]/moment["m00"])
                    # print(self.convert_pixel_to_centimeter(xVal= cx_blue, yVal=cy_blue))
                    crop_img = self.crop_robot_circle(frame, cy_blue, cx_blue)
                    crop_img = self.creat_circle_id(crop_img, color = "blue", cordinate_list=[cx_blue, cy_blue])
                    # FIXME: Change for testing orientation
 
                    # self.check_if_robot(crop_img) # cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV) 
                    # color_picker(cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV))
                    # print(crop_img.shape)
                    if ImageProcessing.CENTER_CORDINATE:
                        cv2.line(crop_img, (0 , int(crop_img.shape[0]/2)), (crop_img.shape[0], int(crop_img.shape[0]/2)), (0, 0, 0), thickness=1, lineType=1)
                        cv2.line(crop_img, (int(crop_img.shape[0]/2) , 0), (int(crop_img.shape[0]/2), crop_img.shape[0]), (0, 0, 0), thickness=1, lineType=1)
                    
                    crop_img1 = np.mean(crop_img, axis=2)
                    hspace, angles, distances = hough_line(crop_img1)
                    angle = []
                    for _, a , distances in zip(*hough_line_peaks(hspace, angles, distances)):
                        angle.append(a)

                    angles = [a*180/np.pi for a in angle]
                    angle_difference = np.max(angles) - np.min(angles)
                    print(f"angle_difference: {angle_difference}")

                    if self.check_if_robot(crop_img, robot_num, frame, cy_blue, cx_blue):
                        self.detect_robot_location(cy_blue, cx_blue, robot_num) 
                        print("It is a Robot")
                    else:
                        print("It is not a Robot")
                    
                    robot_num += 1
                    break
                    # cv2.circle(frame, (cx_blue, cy_blue), 1, (255, 255, 255), -1)
                    # cv2.putText(frame, "Blue", (cx_blue, cy_blue), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
        except Exception as e:
            print(e)
                        
        return frame

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
                print(".json file config FPS is not set")

            if Res is not False:
                self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                # # Change Resolution
                # if isinstance(camera_config["Resolution"], list):
                #     # set camera Resolution from Json file
                #     self.camera_capture.set(3, int(camera_config["Resolution"][0]))
                #     self.camera_capture.set(4, int(camera_config["Resolution"][1]))
                #     print("Resolution Set")
                # else:
                #     print("Resolution Configuration is incorrect")
            else:
                print(".json file config Resolution is not Set")

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
                print(".json file config focus is not Set")
        else:
            print("Set Boolean value for Camera filter setting ")
            
    def set_image_filter(self, frame  : cv2.VideoCapture.read, Blur  : bool = False,GaussianBlur  : bool = False , Segmentation : bool  = False , Res : bool = False):

        ''' Blur Image '''
        if Blur is not False:
            frame = cv2.blur(src=frame, ksize=(5, 5))
            print("Blur is applied")
        

        ''' Blured Image '''
        if GaussianBlur is not False:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            print("GaussianBlur is applied")
         
        
        ''' Segmentation '''
        if Segmentation is not False:
            print("Segmentation is applied")
          
            # reshape the image to a 2D array of pixels and 3 color values (RGB) to push it in to the kmeans
            new_frame = np.float32(frame.reshape((-1, 3)))

            # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
            # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            _, labels, centers = cv2.kmeans(new_frame, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
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
        if Res:
            try:
                frame = cv2.resize(frame, (self.xFrameSizePixel ,self.yFrameSizePixel))
            except Exception as e:
                print(e)

        return frame

    def crop_robot_circle(self, img : cv2.VideoCapture.read, pos_y , pos_x):
        print(f"diff {pos_x} and {pos_y}")
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
    
    def check_if_robot(self, img : cv2.VideoCapture.read, Robo_Num: int, field : cv2.VideoCapture.read, cy: int = None, cx :int = None):
        # constants
        num_of_circle   = 1
        num_of_red      = 0
        num_of_green    = 0
        list_vector_len = []

        b_json = False

        self.saveRobotImage(frame= img, robot_num= Robo_Num,cx= cx, cy= cy )
        
        num_x_cor   = {'green' : [],
                        'red'  : []}

        num_y_cor   = {'green' : [],
                        'red'  : []}

        circle_pack = {"1": [],
                       "2": [],
                       "3": [],
                       "4": []}

        b_if_robot      = False
        robot_id = 0

        try:
        # try to load the json file if exist
            with open("Robo_Color_Config.json") as color_config:
                color_range = json.load(color_config)
            b_json = True
        # Catch Err in this case might be naming diff in json file and print defined
        except Exception as e:
            b_json = False
            print(e)
        
        try:
            frame_hsv       = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # color_picker(img)

            # Color: blue
            # low_blue        = np.array([70, 250, 175], np.uint8)
            # upper_blue      = np.array([165, 255, 255], np.uint8)        
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
                    # mask_blue       = cv2.inRange(frame_hsv, low_blue       ,upper_blue)
                    mask_red        = cv2.inRange(frame_hsv, low_red        ,upper_red)
                    mask_green      = cv2.inRange(frame_hsv, low_green      ,upper_green)
                    mask_black      = cv2.inRange(frame_hsv, low_black      ,upper_black)

                    # contours_blue   = cv2.findContours(mask_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                    # contours_blue   = imutils.grab_contours(contours_blue)
                    
                    contours_red    = cv2.findContours(mask_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                    contours_red    = imutils.grab_contours(contours_red)
                    # img[mask_red > 0] = (255, 0 , 255)

                    contours_green  = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                    contours_green  = imutils.grab_contours(contours_green)
                    # img[mask_green > 0] = (0, 255 , 0)

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
                # mask_blue       = cv2.inRange(frame_hsv, low_blue       ,upper_blue)
                mask_red        = cv2.inRange(frame_hsv, low_red        ,upper_red)
                mask_green      = cv2.inRange(frame_hsv, low_green      ,upper_green)
                mask_black      = cv2.inRange(frame_hsv, low_black      ,upper_black)

                # contours_blue   = cv2.findContours(mask_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                # contours_blue   = imutils.grab_contours(contours_blue)
                
                contours_red    = cv2.findContours(mask_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                contours_red    = imutils.grab_contours(contours_red)
                # img[mask_red > 0] = (255, 0 , 255)

                contours_green  = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                contours_green  = imutils.grab_contours(contours_green)
                # img[mask_green > 0] = (0, 255 , 0)

                contours_black  = cv2.findContours(mask_black.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
                contours_black  = imutils.grab_contours(contours_black)
            
            
            if ImageProcessing.MASK_COLOR_THRESHOLD:
                img[mask_green > 0] = (0  , 255 , 0)
                img[mask_red   > 0] = (0, 0   , 255)
            # img[mask_black > 0] = (0, 0 , 0)

            # for contours in contours_blue:
            #     blue_area = cv2.contourArea(contours)
            #     if blue_area < 60 and blue_area > 5:
            #         #cv2.drawContours(img, [contours], -1, (255,255,255), 1)
            #         moment = cv2.moments(contours) # NOTE: check me again 
                    
            #         cx_blue = int(moment["m10"]/moment["m00"])
            #         cy_blue = int(moment["m01"]/moment["m00"])
            #         self.creat_circle_id(frame = img, color = "blue")
            #         # cv2.circle(frame, (cx_blue, cy_blue), 1, (255, 255, 255), -1)
            #         # cv2.putText(frame, "Blue", (cx_blue, cy_blue), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
            are_of_circle_min = img.shape[0]/15
            are_of_circle_max = img.shape[0]/6

            are_of_circle_min = pi*are_of_circle_min**2
            are_of_circle_max = pi*are_of_circle_max**2

            print(f'area_min: {are_of_circle_min}')
            print(f'area_max: {are_of_circle_max}')
            list_circle_cordinate = []             
            """ contours for red area  """            
            for contours in contours_red:
                red_area = cv2.contourArea(contours)
                print(f"img.frame {img.shape}")
                print(f"red_area {red_area}")
                if red_area < are_of_circle_max and red_area > are_of_circle_min:
                    # cv2.drawContours(img, [contours], -1, (255,255,255), 1)
                    moment = cv2.moments(contours) # NOTE: check me again 
                    cx_red = int(moment["m10"]/moment["m00"])
                    cy_red = int(moment["m01"]/moment["m00"])
                    list_circle_cordinate.append([cy_red,cx_red])
                    print(f"cx_red {cx_red}")
                    print(f"cy_red {cy_red}")
                    # print(f"cx_red: {cx_red} and cy_green: {cy_red}")
                    # img[mask_red > 0] = (255, 0 , 255)
                    num_of_red    += 1
                    num_of_circle += 1
                    position = self.check_circle_position(img.shape[0], img.shape[1], cx_red, cy_red)
                    num_x_cor['red'].append([position , cx_red, cy_red])
                    # circle_pack.append([position , cx_red, cy_red])
                    for i in circle_pack:
                        if i == position:
                            circle_pack[i] = [cx_red, cy_red]
                    # num_y_cor['red'].append(cy_red)
                    print(f'Red Position is: {position}')
                    myradians = math.atan2(int(img.shape[1]/2)-cx_red, int(img.shape[0]/2)-cy_red)
                    mydegrees = math.degrees(myradians)
                    self.creat_circle_id(img,'red', [cx_red, cy_red])
                    cv2.circle(img, (cx_red, cy_red), radius=num_of_red, color=(255, 255, 255), thickness=-1)

            
            # self.creat_circle_id(frame = img, color = "red", cordinate_list = list_circle_cordinate) 
            # if ImageProcessing.SHOW_CIRCLE_LINE_CONNECTION:
            #     cv2.line(img, (num_x_cor['red'][0], num_y_cor['red'][0]), (num_x_cor['red'][1], num_y_cor['red'][1]), (0, 0, 0), thickness=1, lineType=1)
            #     cv2.line(img, (num_x_cor['red'][0], num_y_cor['red'][0]), (num_x_cor['red'][2], num_y_cor['red'][2]), (0, 0, 0), thickness=1, lineType=1)
            #     cv2.line(img, (num_x_cor['red'][2], num_y_cor['red'][2]), (num_x_cor['red'][1], num_y_cor['red'][1]), (0, 0, 0), thickness=1, lineType=1)
            # cv2.line(img, (num_x_cor['red'][2], num_y_cor['red'][2]), (num_x_cor['red'][3], num_y_cor['red'][3]), (0, 0, 0), thickness=1, lineType=1)
            # cv2.line(img, (num_x_cor['red'][3], num_y_cor['red'][3]), (num_x_cor['red'][0], num_y_cor['red'][0]), (0, 0, 0), thickness=1, lineType=1)
            # cv2.line(img, (num_x_cor['red'][2], num_y_cor['red'][2]), (num_x_cor['red'][3], num_y_cor['red'][3]), (0, 0, 0), thickness=1, lineType=1)
            # cv2.line(img, (num_x_cor['red'][2], num_y_cor['red'][2]), (num_x_cor['red'][3], num_y_cor['red'][3]), (0, 0, 0), thickness=1, lineType=1)

            # len1 = math.sqrt((num_x_cor['red'][0] - num_x_cor['red'][1])**2 + (num_y_cor['red'][0] - num_y_cor['red'][1])**2)
            # len2 = math.sqrt((num_x_cor['red'][0] - num_x_cor['red'][2])**2 + (num_y_cor['red'][0] - num_y_cor['red'][2])**2)
            # len3 = math.sqrt((num_x_cor['red'][2] - num_x_cor['red'][1])**2 + (num_y_cor['red'][2] - num_y_cor['red'][1])**2)
            # print(f'length1: {len1}')
            # print(f'length2: {len2}')
            # print(f'length3: {len3}')

            # Check if the the red color rech the limit
            # if num_of_red == 4:
            #     b_if_robot = True
            #     return b_if_robot, robot_id[9]


            list_circle_cordinate.clear()
            """ contours for green area """             
            for contours in contours_green:
                green_area = cv2.contourArea(contours)
                print(f"green_area {green_area}")
                if green_area < are_of_circle_max and green_area > are_of_circle_min:
                    # cv2.drawContours(img, [contours], -1, (255,255,255), 1)
                    moment = cv2.moments(contours) # NOTE: check me again 
                    
                    cx_green = int(moment["m10"]/moment["m00"])
                    cy_green = int(moment["m01"]/moment["m00"])
                    print(f"cx_green {cx_green}")
                    print(f"cy_green {cy_green}")
                    # print(f"cx_green: {cx_green} and cy_green: {cy_green}")
                    num_of_green  += 1
                    num_of_circle += 1 
                    list_circle_cordinate.append([cy_green,cx_green])
                    # img[mask_green > 0] = (0, 255 , 0)
                    # cv2.circle(frame, (cx_green, cy_green), 1, (255, 255, 255), -1)
                    # cv2.putText(frame, "green", (cx_green, cy_green), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
                    # crop_img[mask_green > 0] = (0, 255 , 0)
                    position = self.check_circle_position(img.shape[0], img.shape[1], cx_green, cy_green)
                    # circle_pack.append([position , cx_red, cy_red])
                    for i in circle_pack:
                        if i == position:
                            circle_pack[i] = [cx_green, cy_green]
                    print(f'Green Position is: {position}')
                    num_x_cor['green'].append([position , cx_green, cy_green])
                    # num_x_cor['green'].append(cx_green)
                    # num_y_cor['green'].append(cy_green)
                    self.creat_circle_id(img,'green', [cx_green, cy_green])
                    cv2.circle(img, (cx_green, cy_green), radius=num_of_green, color=(255, 255, 255), thickness=-1)
                    
            # self.creat_circle_id(frame = img, color = "green", cordinate_list = list_circle_cordinate) 
            # print(list_circle_cordinate)
            a = num_x_cor['red']
            b = num_x_cor['green']
            print(f'List Red is : {len(a)}')
            print(f'List Green is : {len(b)}')
            A = num_x_cor['green']
            B = num_x_cor['red']
            print(f'num_x_cor_green : {A}')
            print(f'num_y_cor_red : {B}')
            print(f'circle_pack: {circle_pack}')
            self.getOrientation(circle_pack)
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

            # Check if the the red color reach the limit return the and finish
            # print(num_of_green)
            # if num_of_green == 4:
            #     b_if_robot = True
            #     return b_if_robot, robot_id[8]

            # # Check if the number of spot are match to robot Config
            # if num_of_circle == 4:
            #     b_if_robot = True
            #     #self.saveRobotImage(frame= img, robot_num= Robo_Num,cx= cx, cy= cy )
            # else:
            #     b_if_robot = False
            
            # if ImageProcessing.SHOW_CIRCLE_LINE_CONNECTION:
            #     cv2.line(img, (num_x_cor['green'][0], num_y_cor['green'][0]), (num_x_cor['red'][0], num_y_cor['red'][0]), (0, 0, 0), thickness=1, lineType=1)
            #     cv2.line(img, (num_x_cor['green'][0], num_y_cor['green'][0]), (num_x_cor['red'][2], num_y_cor['red'][2]), (0, 0, 0), thickness=1, lineType=1)
            #     cv2.line(img, (num_x_cor['green'][0], num_y_cor['green'][0]), (num_x_cor['red'][1], num_y_cor['red'][1]), (0, 0, 0), thickness=1, lineType=1)
            
            # a = self.angle_clockwise([num_x_cor['red'][0], num_y_cor['red'][0]], [11, 11])
            # print(f'The Rotation of the image is: a {a}')
            
            # b = self.inner_angle([num_x_cor['red'][0], num_y_cor['red'][0]], [11, 11])
            # print(f'The Rotation of the image is: b {b}')
            
            '''
            height, width = img.shape[:2]
            center        = (width/2, height/2)
            rotate_matrix = cv2.getRotationMatrix2D(center=center, angle= 360 - b, scale=1)
            img           = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))
            '''
             
            img = cv2.circle(img, (11,11), radius=0, color=(0, 0, 255), thickness=-1)

            # print(f'num_x_cor green: {num_x_cor["green"]}')
            # print(f'num_y_cor green: {num_y_cor["green"]}')
            
            # print(f'num_x_cor red:   {num_x_cor["red"]}')
            # print(f'num_y_cor red:   {num_y_cor["red"]}')
            
            # point_one     = (num_x_cor['green'][0] - num_x_cor['red'][0])**2 + (num_y_cor['green'][0] - num_y_cor['red'][0])**2
            # # values_green = [int(num_x_cor['green'][0]), int(num_y_cor['green'][0])]
            # # values_red = [int(num_x_cor['red'][0]), int(num_y_cor['red'][0])]

            # values_green = [int(num_x_cor['green'][0]), int(num_y_cor['green'][0])]
            # values_red = [int(num_x_cor['red'][0]), int(num_y_cor['red'][0])]
            # print(f'values_green : {values_green} and values_red : {values_red}')
            
            # img = cv2.line(img, values_red, values_green, (0, 0, 0), thickness=1, lineType=1)
            # dx = values_green[1] - values_red[1] # img.shape[1] #values_red[1]# 
            # dy = values_green[0] - img.shape[0] // 2 #values_red[0]# 
            # point_one   = math.atan2(dy, dx)
            # point_one   = math.degrees(point_one)
            
            # # point_one = 80 - point_one
            # # point_one = -1* point_one
            # print(f"My degree is : {point_one}")
            # point_one = 10
            # (h, w) = img.shape[:2]
            # # rotate our image by 45 degrees around the center of the image
            # # rotate our image by -90 degrees around the image
            # # M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[1] // 2), point_one, 1.0)
            # # img = cv2.warpAffine(img, M, (w, h))
            # # cv2.line(img, (0 , int(img.shape[0]/2)), (img.shape[0], int(img.shape[0]/2)), (0, 0, 0), thickness=1, lineType=1)
            # # cv2.line(img, (int(img.shape[0]/2) , 0), (int(img.shape[0]/2), img.shape[0]), (0, 0, 0), thickness=1, lineType=1)
            # # cv2.imshow("Rotated by -90 Degrees", rotated)
            
            # point_two     = (num_x_cor['green'][0] - num_x_cor['red'][1])**2 + (num_y_cor['green'][0] - num_y_cor['red'][1])**2
            # point_three   = (num_x_cor['green'][0] - num_x_cor['red'][2])**2 + (num_y_cor['green'][0] - num_y_cor['red'][2])**2

            # point_four    = (num_x_cor['red'][0] - num_x_cor['red'][1])**2 + (num_y_cor['red'][0] - num_y_cor['red'][1])**2
            # point_five    = (num_x_cor['red'][0] - num_x_cor['red'][2])**2 + (num_y_cor['red'][0] - num_y_cor['red'][2])**2
            # point_six     = (num_x_cor['red'][1] - num_x_cor['red'][2])**2 + (num_y_cor['red'][1] - num_y_cor['red'][2])**2


            

            # #if point_one   < 0 : point_one   = point_one   * -1
            # if point_two   < 0 : point_two   = point_two   * -1
            # if point_three < 0 : point_three = point_three * -1
            # if point_four  < 0 : point_four  = point_four  * -1
            # if point_five  < 0 : point_five  = point_five  * -1
            # if point_six   < 0 : point_six   = point_six   * -1

            # point_one   = math.sqrt(point_one)
            # point_two   = math.sqrt(point_two)
            # point_three = math.sqrt(point_three)
            # point_four  = math.sqrt(point_four)
            # point_five  = math.sqrt(point_five)
            # point_six   = math.sqrt(point_six)
            
            # list_me = [point_one, point_two, point_three, point_four, point_five, point_six]
            # list_me = sorted(list_me)
            # print(list_me)
            '''
            point_one   = math.atan2((num_x_cor['red'][0] - 11) , (num_y_cor['red'][0] - 11))
            point_one   = math.degrees(point_one)
            
            point_two   = math.atan2((num_x_cor['red'][1] - 11) , (num_y_cor['red'][1] - 11))
            point_two   = math.degrees(point_two)

            point_three   = math.atan2((num_x_cor['green'][0] - 11) , (num_y_cor['green'][0] - 11))
            point_three   = math.degrees(point_three)

            point_four   = math.atan2((num_x_cor['green'][1] - 11) , (num_y_cor['green'][1] - 11))
            point_four   = math.degrees(point_four)
            '''
            # print("##############################")
            # print(f'green: {point_one}')
            # print(f'green: {point_two}')
            # print(f'green: {point_three}')
            # print(f'{point_four}')
            # print(f'{point_five}')
            # print(f'{point_six}')
            # print("##############################")
            # point_one   = math.atan2((num_x_cor['red'][0] - num_x_cor['green'][0]) , (num_y_cor['red'][0] - num_x_cor['green'][0]))
            # point_one   = math.degrees(-1*point_one)
            # print(f'Degree is: {point_one}')
            # print(f'num_x_cor {num_x_cor}')    
            # print(f'num_y_cor {num_y_cor}')
            # img = self.overlay_image_alpha(img, field)  
            '''
            if b_if_robot:
                self.robot_id_detection(num_of_green,num_of_red, robot_num, robot_id)
            else:
                self.robot_id_detection(num_of_green,num_of_red, robot_num, robot_id)
                print(f"ROBOT Num.{robot_num} IT IS NOT ROBOT {num_of_circle}!!") 
            '''
        except Exception as e:
            print(e)

        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, str(int(fps)), (30, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        cv2.namedWindow("RobotSoccer1\tHit Escape to Exit")
        cv2.imshow("RobotSoccer1\tHit Escape to Exit", img)
        return b_if_robot, robot_id      

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
        if ImageProcessing.CIRCLE_ID_COLOR_BY_CORDINATE:
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
        length_list = []

        len_one   = math.sqrt((circle_pack["1"][0] - circle_pack["2"][0])**2 + (circle_pack["1"][1] - circle_pack["2"][1])**2)
        len_two   = math.sqrt((circle_pack["1"][0] - circle_pack["3"][0])**2 + (circle_pack["1"][1] - circle_pack["3"][1])**2)
        len_three = math.sqrt((circle_pack["1"][0] - circle_pack["4"][0])**2 + (circle_pack["1"][1] - circle_pack["4"][1])**2)
        len_four  = math.sqrt((circle_pack["2"][0] - circle_pack["3"][0])**2 + (circle_pack["2"][1] - circle_pack["3"][1])**2)
        len_five  = math.sqrt((circle_pack["2"][0] - circle_pack["4"][0])**2 + (circle_pack["2"][1] - circle_pack["4"][1])**2)
        len_six   = math.sqrt((circle_pack["3"][0] - circle_pack["4"][0])**2 + (circle_pack["3"][1] - circle_pack["4"][1])**2)

        print("##############################")
        print(f'len_one:   {len_one}'  )
        print(f'len_two:   {len_two}'  )
        print(f'len_three: {len_three}')
        print(f'len_four:  {len_four}' )
        print(f'len_five:  {len_five}' )
        print(f'len_six:   {len_six}'  )
        print("##############################")
        length_list = [len_one, len_two, len_three, len_four, len_five, len_six]
        length_list = sorted(length_list)
        
        angle = 0
        return angle

    def resize_frame(self, frame):
        return frame

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
        if ImageProcessing.SAVE_ROBOT_IMAGE == True:
            try:
                if frame is not None:
                    cv2.imwrite(f"Robot{robot_num}_x{cx}_y{cy}.jpg", frame)
                else:
                    print("crop image is not valid")
            except Exception as e:
                print(e)
        else:
            pass

    def length(self, v):
        return sqrt(v[0]**2+v[1]**2)
    def dot_product(self, v,w):
        return v[0]*w[0]+v[1]*w[1]
    def determinant(self, v,w):
        return v[0]*w[1]-v[1]*w[0]
    def inner_angle(self, v,w):
        cosx= self.dot_product(v,w)/(self.length(v)*self.length(w))
        rad=acos(cosx) # in radians
        return rad*180/pi # returns degrees
    def angle_clockwise(self, A, B):
        inner = self.inner_angle(A,B)
        det   = self.determinant(A,B)
        if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else: # if the det > 0 then A is immediately clockwise of B
            return 360-inner

    def finish_capturing(self):
        cv2.destroyAllWindows()

imgProc = ImageProcessing()
