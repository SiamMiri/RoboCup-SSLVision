import json
import cv2
from cv2 import COLOR_BGR2RGB
import numpy as np
import time
import mediapipe as mp

################ deffines available robots colors ####################
robot_teams = {"read_team":  (255,0,0),
               "blue_team":  (0,255,0),
               "green_team": (0,0,255)}

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

    
    def start_capturing(self, cameraConfig: json):
        """ Start capturing get camera config from start_capturing_cammand """        
        # Change FPS
        # FIXME : FPS does not applied
        if isinstance(cameraConfig["FPS"], int):
            self.camera_capture.set(cv2.CAP_PROP_FPS, cameraConfig["FPS"])       # set camera FPS from Json file
        elif isinstance(cameraConfig["FPS"], float):
            self.camera_capture.set(cv2.CAP_PROP_FPS, int(cameraConfig["FPS"]))  # set camera FPS from Json file
        else:
            print(".json file config FPS is not correct")

        # Change Resolution
        if isinstance(cameraConfig["Resolution"], list):
            # set camera Resolution from Json file
            self.camera_capture.set(3, int(cameraConfig["Resolution"][0]))
            self.camera_capture.set(4, int(cameraConfig["Resolution"][1]))
        else:
            print(".json file config Resolution is not correct")

        # Change Focus
        if isinstance(cameraConfig["focus"], int):
            # set camera Resolution from Json file
            self.camera_capture.set(cv2.CAP_PROP_FOCUS, cameraConfig["focus"])
        elif isinstance(cameraConfig["focus"], float):
            # set camera Resolution from Json file
            self.camera_capture.set(cv2.CAP_PROP_FOCUS, int(cameraConfig["focus"]))
        else:
            self.camera_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        cv2.namedWindow("RobotSoccer\tHit Escape to Exit")

        while True:
            ret, frame = self.camera_capture.read()

            imgRGB = cv2.cvtColor(frame,  cv2.COLOR_BGR2RGB)
            res = self.pose.process(imgRGB)

            print(res.pose_landmarks)

            if not ret:
                print("failed to grab frame")
                break
            
            # Show frame rate
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(frame, str(int(fps)), (30,40), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
            cv2.imshow("RobotSoccer\tHit Escape to Exit", frame)
            
            # cv2.waitKey(1)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
                    
    def detect_robot_id(self, img):
        image = cv2.imread(self.args[img])

    def detect_robot_orientation(self):
        return None

    def detect_robot_location(self):
        return None

    def finish_capturing(self):
        cv2.destroyAllWindows()


""" Extra: 

# Find the rotation and translation vectors
ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
    
# Project 3D points to images plane
imgpts, jac = cv.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)    
    
"""

imgProc = ImageProcessing()
