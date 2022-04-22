import json
import cv2
from cv2 import COLOR_BGR2RGB
import numpy as np
import time
import mediapipe as mp
import imutils
from collections import deque
import argparse
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

            frame = cv2.imread("FieldTest_Left_Light_On_Daylight(hight).jpg")
            # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            # frame = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            # self.detect_robot_orientation(frame=frame)
            # Aplied Filter GaussianBlur and Segmentation
            frame = self.set_image_filter(frame = frame, GaussianBlur = False, Segmentation = False)

            # Change Frame Size
            if isinstance(camera_config["resize_frame"], list):
                frame = cv2.resize(frame, (int(camera_config["resize_frame"][0]),
                                           int(camera_config["resize_frame"][1])))
            else:
                print("Frame Size Configuration is incorrect")
            
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
        
        frame_hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

        
        
        # define masks
        mask_blue   = cv2.inRange(frame_hsv, low_blue       ,upper_blue)
        mask_red    = cv2.inRange(frame_hsv, low_red        ,upper_red)
        mask_yellow = cv2.inRange(frame_hsv, low_yellow     ,upper_yellow)
        mask_green  = cv2.inRange(frame_hsv, low_green      ,upper_green)
        mask_orange = cv2.inRange(frame_hsv, low_orange     ,upper_orange)


        
        # CHAIN_APPROX_NONE gibe all points
        # contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # cv2.drawContours(mask, contours, -1, (0, 0, 0), 1)  # -1 means all the counters
        contours_blue       = cv2.findContours(mask_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_blue       = imutils.grab_contours(contours_blue)
        frame[mask_blue > 0] = (255, 0 , 0)
        
        contours_red        = cv2.findContours(mask_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_red        = imutils.grab_contours(contours_red)
        frame[mask_red > 0] = (0, 0 , 255)

        contours_yellow     = cv2.findContours(mask_yellow.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_yellow     = imutils.grab_contours(contours_yellow)
        #frame[mask_yellow > 0] = (0, 30 , 255)

        contours_green      = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_green      = imutils.grab_contours(contours_green)
        frame[mask_green > 0] = (0, 255 , 0)

        contours_orange     = cv2.findContours(mask_orange.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_orange     = imutils.grab_contours(contours_orange)
        #frame[mask_orange > 0] = (20, 150 , 255)

        # frame = cv2.bitwise_and(result, result, mask=full_mask)
        cx_blue = 0 
        cy_blue = 0 
        moment = 0
        """ contours for blue area """
        robot_num = 1
        for contours in contours_blue:
            blue_area = cv2.contourArea(contours)
            if blue_area < 60 and blue_area > 15:
                #cv2.drawContours(frame, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_blue = int(moment["m10"]/moment["m00"])
                cy_blue = int(moment["m01"]/moment["m00"])
                crop_img = self.crop_robot_rec(frame, cy_blue, cx_blue)
                 # FIXME: Change for testing orientation
                try:
                    if crop_img is not None:
                        cv2.imwrite(f"Robot{robot_num}_x{cx_blue}_y{cy_blue}.jpg", crop_img)
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
            
    def set_image_filter(self, frame  : cv2.VideoCapture.read, GaussianBlur  : bool = False , Segmentation : bool  = False ):
        bGaussianBlur = False
        bSegmentation = False

        ''' Blured Image '''
        if GaussianBlur is not False:
            frame = cv2.GaussianBlur(frame, (37, 37), 0)
            
            if bGaussianBlur is not True:
                print("GaussianBlur is applied")
                bGaussianBlur = True
         
        
        ''' Segmentation '''
        if Segmentation is not False:
            
            if bSegmentation is not True:
                print("Segmentation is applied")
                bSegmentation = True
          
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

    def crop_robot_rec(self, img : cv2.VideoCapture.read, pos_y , pos_x):
        np_crop_img = np.array(img)
        pos_y = pos_y - 15
        pos_x = pos_x - 15
        higth = pos_y + 30
        width = pos_x + 30
        np_crop_img  = np_crop_img[pos_y:higth, pos_x:width]
        crop_img = np_crop_img
        return crop_img
    
    def check_if_robot(self, img : cv2.VideoCapture.read, Robo_Num: int):
        num_of_circle   = 0
        num_of_red      = 0
        num_of_green    = 0
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
        low_red         = np.array([15, 0, 0], np.uint8)
        upper_red       = np.array([155, 255, 255], np.uint8)
                        
        # Color: green
        low_green       = np.array([30, 0, 254], np.uint8)
        upper_green     = np.array([100, 255, 255], np.uint8)
                
        # define masks
        mask_blue       = cv2.inRange(frame_hsv, low_blue       ,upper_blue)
        mask_red        = cv2.inRange(frame_hsv, low_red        ,upper_red)
        mask_green      = cv2.inRange(frame_hsv, low_green      ,upper_green)

        contours_blue   = cv2.findContours(mask_blue.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_blue   = imutils.grab_contours(contours_blue)
        
        contours_red    = cv2.findContours(mask_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_red    = imutils.grab_contours(contours_red)

        contours_green  = cv2.findContours(mask_green.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#[-2]
        contours_green  = imutils.grab_contours(contours_green)

        for contours in contours_blue:
            blue_area = cv2.contourArea(contours)
            if blue_area < 60 and blue_area > 5:
                cv2.drawContours(img, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_blue = int(moment["m10"]/moment["m00"])
                cy_blue = int(moment["m01"]/moment["m00"])
                # cv2.circle(frame, (cx_blue, cy_blue), 1, (255, 255, 255), -1)
                # cv2.putText(frame, "Blue", (cx_blue, cy_blue), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
        
        cx_red = 0 
        cy_red = 0 
        moment = 0
        """ contours for red area """             
        for contours in contours_red:
            red_area = cv2.contourArea(contours)
            if red_area < 60 and red_area > 20:
                cv2.drawContours(img, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_red = int(moment["m10"]/moment["m00"])
                cy_red = int(moment["m01"]/moment["m00"])
                
                num_of_red    += 1
                num_of_circle += 1 
                # cv2.circle(frame, (cx_red, cy_red), 7, (255, 255, 255), -1)
                # cv2.putText(frame, "red", (cx_red, cy_red), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
                        
        cx_green    = 0 
        cy_green    = 0 
        moment      = 0  
        """ contours for green area """             
        for contours in contours_green:
            green_area = cv2.contourArea(contours)
            if green_area < 60 and green_area > 5:
                cv2.drawContours(img, [contours], -1, (255,255,255), 1)
                moment = cv2.moments(contours) # NOTE: check me again 
                
                cx_green = int(moment["m10"]/moment["m00"])
                cy_green = int(moment["m01"]/moment["m00"])

                num_of_green    += 1
                num_of_circle += 1 
                # cv2.circle(frame, (cx_green, cy_green), 1, (255, 255, 255), -1)
                # cv2.putText(frame, "green", (cx_green, cy_green), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 1)
                # crop_img[mask_green > 0] = (0, 255 , 0)
        
        # Check if the number of spot are match to robot Config
        if num_of_circle == 4:
            b_if_robot = True
        else:
            b_if_robot = False

        cv2.namedWindow("RobotSoccer1\tHit Escape to Exit")
        cv2.imshow("RobotSoccer1\tHit Escape to Exit", img) 

        if b_if_robot:
            self.robot_id_detection(num_of_green,num_of_red, robot_num)
        else:
            self.robot_id_detection(num_of_green,num_of_red, robot_num)
            print(f"ROBOT Num.{robot_num} IT IS NOT ROBOT {num_of_circle}!!") 
        return b_if_robot
        

    def robot_id_detection(self, num_of_green:int, num_of_red:int, Robo_Num : int):
        print(f"Robot{Robo_Num}: num_of_green: {num_of_green}, num_of_red: {num_of_red}")

    def finish_capturing(self):
        cv2.destroyAllWindows()
""" Extra: 

# Find the rotation and translation vectors
ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
    
# Project 3D points to images plane
imgpts, jac = cv.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)    
    
"""

imgProc = ImageProcessing()
