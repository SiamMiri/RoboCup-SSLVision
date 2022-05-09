import json
import cv2
from cv2 import COLOR_BGR2RGB
import numpy as np
import time
import mediapipe as mp
import imutils
from collections import deque
import argparse
from math import atan2, cos, sin, sqrt, pi

#from HSV_Color_Picker import *


# This class receive Images from the ControlCommand
# It processes images to detect the robot ID, origination
# and location. The return the data back as needed
class ImageProcessing():
    """ This class get images from ControlCommand """

    def __init__(self):
        self.camera_capture = cv2.VideoCapture(0)
        self.pTime = 0
        self.robot_center_pos = None

        mpPose = mp.solutions.pose
        self.pose = mpPose.Pose()

    def start_capturing(self, camera_config: json):
        """ Start capturing get camera config from start_capturing_command """
        self.set_camera_config(camera_config, Fps=False, Res=False, Focus=False)

        # Creat windows for the frame
        cv2.namedWindow("RobotSoccer\tHit Escape to Exit")
        
        while True:
        
            # capture a image from the camera
            # ret, frame = self.camera_capture.read() # FIXME: Changed to load Image

            # print(frame.shape)
            frame = cv2.imread("FieldTest_AllLight_Off_Daylight(hight).jpg")
            # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            # frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            # self.detect_robot_orientation(frame=frame)
            # Aplied Filter GaussianBlur and Segmentation
            frame = self.set_image_filter(frame = frame, Blur= False,  GaussianBlur = False, Segmentation = False)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)

            """"""
            # Change Frame Size
            # if isinstance(camera_config["resize_frame"], list):
            #     frame = cv2.resize(frame, (int(camera_config["resize_frame"][0]),
            #                             int(camera_config["resize_frame"][1])))
            # else:
            #    print("Frame Size Configuration is incorrect")
            
            # Detect Robot 
            frame =  self.detect_robot_id(frame = frame)
            ''' FIXME changed for working on images
            if not ret:
                print("failed to grab frame")
                break
            '''
            
            # Show frame rate
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(frame, str(int(fps)), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            # cv2.putText(blurred_frame, str(int(fps)), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # cv2.imshow("B&W RobotSoccer\tHit Escape to Exit", mask)  # BLack and White Image
            # cv2.imshow("RobotSoccer\tHit Escape to Exit", np.vstack(result))  # Normal Images with contours
            # cv2.imshow("RobotSoccer\tHit Escape to Exit", reduced)

            cv2.imshow("RobotSoccer\tHit Escape to Exit", frame)
            # cv2.waitKey(1)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

    def detect_robot_id(self, frame: cv2.VideoCapture.read):
        # Contants:
        robot_num = 1

        frame_hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.convert_pixel_to_centimeter(frame_hsv)

        # Source: https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
        
        # Color: blue
        low_blue    = np.array([90, 150, 0], np.uint8)
        upper_blue  = np.array([140, 255, 255], np.uint8)        
        
        # Color: Red
        low_red     = np.array([110, 50, 120], np.uint8)
        upper_red   = np.array([250, 255, 255], np.uint8)
                
        # Color: yellow
        low_yellow      = np.array([18, 70, 120], np.uint8)
        upper_yellow    = np.array([15, 255, 255], np.uint8)
        
        # Color: green
        low_green      = np.array([60, 0, 150], np.uint8)
        upper_green    = np.array([80, 230, 255], np.uint8)
                
        # Color: Orange
        low_orange     = np.array([10, 100, 20], np.uint8)
        upper_orange   = np.array([25, 255, 255], np.uint8)

        # Color: black
        low_black       = np.array([0, 0, 0], np.uint8)
        upper_black     = np.array([180, 255, 145], np.uint8)
                
        
        
        # define masks
        mask_blue   = cv2.inRange(frame_hsv, low_blue       ,upper_blue)
        mask_red    = cv2.inRange(frame_hsv, low_red        ,upper_red)
        mask_yellow = cv2.inRange(frame_hsv, low_yellow     ,upper_yellow)
        mask_green  = cv2.inRange(frame_hsv, low_green      ,upper_green)
        mask_orange = cv2.inRange(frame_hsv, low_orange     ,upper_orange)
        mask_black  = cv2.inRange(frame_hsv, low_black     ,upper_black)


        
        # CHAIN_APPROX_NONE gibe all points
        # contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # cv2.drawContours(mask, contours, -1, (0, 0, 0), 1)  # -1 means all the counters
        contours_blue       = cv2.findContours(mask_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_blue       = imutils.grab_contours(contours_blue)
        frame[mask_blue > 0] = (255, 0 , 97)

        contours_red        = cv2.findContours(mask_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_red        = imutils.grab_contours(contours_red)

        contours_yellow     = cv2.findContours(mask_yellow.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_yellow     = imutils.grab_contours(contours_yellow)
        #frame[mask_yellow > 0] = (0, 30 , 255)

        contours_green      = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_green      = imutils.grab_contours(contours_green)

        contours_black      = cv2.findContours(mask_black.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_black      = imutils.grab_contours(contours_black)
        #frame[mask_black > 0] = (0, 0 , 0)

        contours_orange     = cv2.findContours(mask_orange.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_orange     = imutils.grab_contours(contours_orange)
        #frame[mask_orange > 0] = (20, 150 , 255)

        # frame = cv2.bitwise_and(result, result, mask=full_mask)
        cx_blue = 0 
        cy_blue = 0 
        moment = 0
        r = 100
        i = 0
        
        """ contours for blue area """
        for contours in contours_blue:
            blue_area = cv2.contourArea(contours)
            width = frame.shape[1]
            height = frame.shape[0]
            print(f"width: {width}\nheight: {height}")
            if blue_area < 6000 and blue_area > 15:
                #cv2.drawContours(frame, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                #cv2.drawContours(frame, contours, i, (0, 0, 255), 2)
                i += 1
                # Find the orientation of each shape
                # self.getOrientation(contours, frame)

                cx_blue = int(moment["m10"]/moment["m00"])
                cy_blue = int(moment["m01"]/moment["m00"])
                crop_img = self.crop_robot_circle(frame, cy_blue, cx_blue)
                 # FIXME: Change for testing orientation
                try:
                    if crop_img is not None:
                        cv2.imwrite(f"Robot{robot_num}_x{cx_blue}_y{cy_blue}.jpg", crop_img)
                    else:
                        print("crop image is not valid")
                except Exception as e:
                    print(e)
                # self.check_if_robot(crop_img) # cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV) 
                # color_picker(cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV))
                if self.check_if_robot(crop_img, robot_num):
                    self.detect_robot_location(cy_blue, cx_blue, robot_num) 
                    print("It is a Robot")
                else:
                    print("It is not a Robot")
                
                robot_num += 1
                # cv2.circle(frame, (cx_blue, cy_blue), 1, (255, 255, 255), -1)
                # cv2.putText(frame, "Blue", (cx_blue, cy_blue), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
        
        cx_red = 0 
        cy_red = 0 
        moment = 0
        """ contours for red area              
        for contours in contours_red:
            red_area = cv2.contourArea(contours)
            if red_area < 90 and red_area > 15:
                cv2.drawContours(frame, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_red = int(moment["m10"]/moment["m00"])
                cy_red = int(moment["m01"]/moment["m00"])
                
                # cv2.circle(frame, (cx_red, cy_red), 7, (255, 255, 255), -1)
                # cv2.putText(frame, "red", (cx_red, cy_red), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
        """       
        cx_yellow = 0 
        cy_yellow = 0 
        moment = 0 
        """ contours for yellow area              
        for contours in contours_yellow:
            yellow_area = cv2.contourArea(contours)
            if yellow_area < 90 and yellow_area > 15:
                cv2.drawContours(frame, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_yellow = int(moment["m10"]/moment["m00"])
                cy_yellow = int(moment["m01"]/moment["m00"])
                
                cv2.circle(frame, (cx_yellow, cy_yellow), 1, (255, 255, 255), -1)
                cv2.putText(frame, "yellow", (cx_yellow, cy_yellow), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
                #
        """
        cx_green = 0 
        cy_green = 0 
        moment = 0  
        """ contours for green area              
        for contours in contours_green:
            green_area = cv2.contourArea(contours)
            if green_area < 60 and green_area > 15:
                # cv2.drawContours(frame, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_green = int(moment["m10"]/moment["m00"])
                cy_green = int(moment["m01"]/moment["m00"])
                
                # cv2.circle(frame, (cx_green, cy_green), 1, (255, 255, 255), -1)
                # cv2.putText(frame, "green", (cx_green, cy_green), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
                # crop_img[mask_green > 0] = (0, 255 , 0)
        """
        cx_orange = 0 
        cy_orange = 0 
        moment = 0  
        """ contours for orange area              
        for contours in contours_orange:
            orange_area = cv2.contourArea(contours)
            if orange_area < 60 and orange_area > 40:
                cv2.drawContours(frame, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_orange = int(moment["m10"]/moment["m00"])
                cy_orange = int(moment["m01"]/moment["m00"])
                
                # cv2.circle(frame, (cx_orange, cy_orange), 1, (255, 255, 255), -1)
                # cv2.putText(frame, "orange", (cx_orange, cy_orange), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
        """
        #cx  = cx_red - cx_orange
        #cy  = cy_red - cy_orange
        #print(abs(cx))
        #print(abs(cy))
        '''
        if abs(cx) < 700 and abs(cx) > 200 and abs(cy) < 700 and abs(cy) < 200:
            print("********************")
            print(abs(cx))
            print(abs(cy))
            print("********************")
        '''
        
       
        '''
        # print(contours[0]) if contours[0] is not None else print("Pass")
        if len(contours_blue) > 40:
            # for contour in contours:
            # position = cv2.contourArea(contour)
            blue_area = max(contours_blue, key=cv2.contourArea)
            (xg, yg, wg, hg) = cv2.boundingRect(blue_area)
            cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)
        '''
                        
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
                # Change Resolution
                if isinstance(camera_config["Resolution"], list):
                    # set camera Resolution from Json file
                    self.camera_capture.set(3, int(camera_config["Resolution"][0]))
                    self.camera_capture.set(4, int(camera_config["Resolution"][1]))
                    print("Resolution Set")
                else:
                    print("Resolution Configuration is incorrect")
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
            
    def set_image_filter(self, frame  : cv2.VideoCapture.read, Blur  : bool = False,GaussianBlur  : bool = False , Segmentation : bool  = False ):

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
                mask = cv2.inRange(labels, i, i)
                mask = np.dstack([mask] * 3)  # Make it 3 channel
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
                             
        return frame

    def crop_robot_circle(self, img : cv2.VideoCapture.read, pos_y , pos_x):

        pos_y   = pos_y - 10
        pos_x   = pos_x - 10
        radius  = 10 
                
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
         #crop rec image from robot
        np_crop_img  = np_crop_img[pos_y:higth, pos_x:width]
        crop_img = np_crop_img
        
        return crop_img
    
    def check_if_robot(self, img : cv2.VideoCapture.read, Robo_Num: int):
        # contants
        num_of_circle   = 0
        num_of_red      = 0
        num_of_green    = 0
        robot_id        = [0, 1, 2, 3, 4, 5, 6,
                           7, 8, 9, 10, 11, 12, 
                           13, 14, 15]

        b_if_robot      = False

        if isinstance(Robo_Num, int):
            robot_num = Robo_Num
        else:
            robot_num = "Unknown"

        frame_hsv       = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # color_picker(img)

        # Color: blue
        low_blue        = np.array([70, 250, 175], np.uint8)
        upper_blue      = np.array([165, 255, 255], np.uint8)        
        
        # Color: Red
        low_red         = np.array([5, 0, 0], np.uint8)
        upper_red       = np.array([165, 255, 255], np.uint8)
                        
        # Color: green
        low_green       = np.array([30, 0, 254], np.uint8)
        upper_green     = np.array([100, 255, 255], np.uint8)

        # Color: black
        low_black       = np.array([0, 0, 0], np.uint8)
        upper_black     = np.array([180, 255, 145], np.uint8)
                
        # define masks
        mask_blue       = cv2.inRange(frame_hsv, low_blue       ,upper_blue)
        mask_red        = cv2.inRange(frame_hsv, low_red        ,upper_red)
        mask_green      = cv2.inRange(frame_hsv, low_green      ,upper_green)
        mask_black      = cv2.inRange(frame_hsv, low_black      ,upper_black)

        contours_blue   = cv2.findContours(mask_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_blue   = imutils.grab_contours(contours_blue)
        
        contours_red    = cv2.findContours(mask_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_red    = imutils.grab_contours(contours_red)
        #img[mask_red > 0] = (255, 0 , 255)

        contours_green  = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_green  = imutils.grab_contours(contours_green)
        #img[mask_green > 0] = (0, 255 , 0)

        contours_black  = cv2.findContours(mask_black.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_black  = imutils.grab_contours(contours_black)
        img[mask_black > 0] = (0, 0 , 0)

        for contours in contours_blue:
            blue_area = cv2.contourArea(contours)
            if blue_area < 60 and blue_area > 5:
                #cv2.drawContours(img, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_blue = int(moment["m10"]/moment["m00"])
                cy_blue = int(moment["m01"]/moment["m00"])
                self.creat_circle_id(frame = img, color = "blue")
                # cv2.circle(frame, (cx_blue, cy_blue), 1, (255, 255, 255), -1)
                # cv2.putText(frame, "Blue", (cx_blue, cy_blue), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
        
        cx_red = 0 
        cy_red = 0
        list_circle_cordinate = [] 
        moment = 0
        """ contours for red area  """            
        for contours in contours_red:
            red_area = cv2.contourArea(contours)
            if red_area < 60 and red_area > 5:
                cv2.drawContours(img, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_red = int(moment["m10"]/moment["m00"])
                cy_red = int(moment["m01"]/moment["m00"])
                list_circle_cordinate.append([cy_red,cx_red])
                print(f"cx_red: {cx_red} and cy_green: {cy_red}")
                num_of_red    += 1
                num_of_circle += 1
                # cv2.circle(frame, (cx_red, cy_red), 7, (255, 255, 255), -1)
                # cv2.putText(frame, "red", (cx_red, cy_red), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
        
        self.creat_circle_id(frame = img, color = "red", cordinate_list = list_circle_cordinate) 

        # Check if the the red color rech the limit
        if num_of_red == 4:
            b_if_robot = True
            return b_if_robot, robot_id[9]

        cx_green    = 0 
        cy_green    = 0 
        list_circle_cordinate.clear()
        moment      = 0  
        """ contours for green area """             
        for contours in contours_green:
            green_area = cv2.contourArea(contours)
            if green_area < 60 and green_area > 5:
                #cv2.drawContours(img, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_green = int(moment["m10"]/moment["m00"])
                cy_green = int(moment["m01"]/moment["m00"])
                print(f"cx_green: {cx_green} and cy_green: {cy_green}")
                num_of_green  += 1
                num_of_circle += 1 
                list_circle_cordinate.append([cy_green,cx_green])
                # cv2.circle(frame, (cx_green, cy_green), 1, (255, 255, 255), -1)
                # cv2.putText(frame, "green", (cx_green, cy_green), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
                # crop_img[mask_green > 0] = (0, 255 , 0)
        
        self.creat_circle_id(frame = img, color = "green", cordinate_list = list_circle_cordinate) 
        #print(list_circle_cordinate)

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
        if num_of_green == 4:
            b_if_robot = True
            return b_if_robot, robot_id[8]

        # Check if the number of spot are match to robot Config
        if num_of_circle == 4:
            b_if_robot = True
        else:
            b_if_robot = False

        cv2.namedWindow("RobotSoccer1\tHit Escape to Exit")
        cv2.imshow("RobotSoccer1\tHit Escape to Exit", img) 
        '''
        if b_if_robot:
            self.robot_id_detection(num_of_green,num_of_red, robot_num, robot_id)
        else:
            self.robot_id_detection(num_of_green,num_of_red, robot_num, robot_id)
            print(f"ROBOT Num.{robot_num} IT IS NOT ROBOT {num_of_circle}!!") 
        '''
        return b_if_robot, robot_id
        

    def robot_id_detection(self, num_of_green:int = None, num_of_red:int = None, Robo_Num : int = None, robot_id: list = None):
        if num_of_green == 2 and num_of_red == 2:
            print("Num_Green = 2  Num_Red = 2")
            #return robot_id[1], robot_id[3], robot_id[5], robot_id[10], robot_id[11], robot_id[7]
        
        if num_of_green == 3 and num_of_red == 1:
            print("Num_Green = 3  Num_Red = 1")
            #return robot_id[2], robot_id[6], robot_id[12], robot_id[14]
        
        if num_of_green == 1 and num_of_red == 3:
            print("Num_Green = 1  Num_Red = 3")
            #return robot_id[0], robot_id[4], robot_id[13], robot_id[15]

        #print(f"Robot{Robo_Num}: num_of_green: {num_of_green}, num_of_red: {num_of_red}")

    
    def convert_pixel_to_centimeter(self, frame: cv2.VideoCapture.read):
        try:
            height, width = frame.shape[:2]
            #print(f"height: {height} and width: {width}")

        except Exception as e :
            print(e) 

    def creat_circle_id(self, frame: cv2.VideoCapture.read, color: str = None, cordinate_list :list = None):
        # Center coordinates
        center_coordinates = (10, 10)
        
        # Radius of circle
        radius = 3
        # Line thickness of -1 px
        thickness = -1

        if color == "blue":
            # blue color in BGR
            color = (255, 0, 0)
            frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
            return frame

        if color == "red": 
            # red color in BGR
            color = (255, 0 , 255) # (0, 0, 255)
            radius = 3
            for i in cordinate_list:
                center_coordinates = (i[0], i[1])
                frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
            return frame

        if color == "green": 
            # green color in BGR
            color = (0, 255, 0)
            radius = 3
            for i in cordinate_list:
                center_coordinates = (i[0], i[1])
                frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
            return frame

        return frame

    def finish_capturing(self):
        cv2.destroyAllWindows()

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

    def getOrientation(self, pts, img):
        ## [pca]
        # Construct a buffer used by the pca analysis
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i,0] = pts[i,0,0]
            data_pts[i,1] = pts[i,0,1]
        
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        
        # Store the center of the object
        cntr = (int(mean[0,0]), int(mean[0,1]))
        ## [pca]
        
        ## [visualization]
        # Draw the principal components
        cv2.circle(img, cntr, 3, (255, 0, 255), 2)
        p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
        self.drawAxis(img, cntr, p1, (255, 255, 0), 1)
        self.drawAxis(img, cntr, p2, (0, 0, 255), 5)
        
        angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
        ## [visualization]

        # Label with the rotation angle
        label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
        textbox = cv2.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
        cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        return angle

imgProc = ImageProcessing()
