import json
import cv2
from cv2 import COLOR_BGR2RGB
import numpy as np
import time
import mediapipe as mp

################ deffines available robots colors ####################
robot_teams = {"read_team": (255, 0, 0),
               "blue_team": (0, 255, 0),
               "green_team": (0, 0, 255)}


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
        # Change FPS
        # FIXME : FPS does not applied
        if isinstance(camera_config["FPS"], int):
            self.camera_capture.set(cv2.CAP_PROP_FPS, camera_config["FPS"])  # set camera FPS from Json file
        elif isinstance(camera_config["FPS"], float):
            self.camera_capture.set(cv2.CAP_PROP_FPS, int(camera_config["FPS"]))  # set camera FPS from Json file
        else:
            print(".json file config FPS is not correct")

        # Change Resolution
        if isinstance(camera_config["Resolution"], list):
            # set camera Resolution from Json file
            self.camera_capture.set(3, int(camera_config["Resolution"][0]))
            self.camera_capture.set(4, int(camera_config["Resolution"][1]))
        else:
            print(".json file config Resolution is not correct")

        # Change Focus
        if isinstance(camera_config["focus"], int):
            # set camera Resolution from Json file
            self.camera_capture.set(cv2.CAP_PROP_FOCUS, camera_config["focus"])
        elif isinstance(camera_config["focus"], float):
            # set camera Resolution from Json file
            self.camera_capture.set(cv2.CAP_PROP_FOCUS, int(camera_config["focus"]))
        else:
            self.camera_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        cv2.namedWindow("RobotSoccer\tHit Escape to Exit")

        while True:
            ret, frame = self.camera_capture.read()
            blurred_frame = cv2.GaussianBlur(frame, (37, 37), 0)


            Z = np.float32(frame.reshape((-1, 3)))

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 4
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            labels = labels.reshape((frame.shape[:-1]))
            reduced = np.uint8(centers)[labels]

            result = [np.hstack([frame, reduced])]
            for i, c in enumerate(centers):
                mask = cv2.inRange(labels, i, i)
                mask = np.dstack([mask] * 3)  # Make it 3 channel
                ex_img = cv2.bitwise_and(frame, mask)
                ex_reduced = cv2.bitwise_and(reduced, mask)
                result.append(np.hstack([ex_img, ex_reduced]))

            # cv2.imwrite('watermelon_out.jpg', np.vstack(result))


            hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

            # CHANGE : This is just sample and it should be changed
            low_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])

            # define mask
            mask = cv2.inRange(hsv, low_blue, upper_blue)

            # CHAIN_APPROX_NONE gibe all points
            contours = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cv2.drawContours(frame, contours, -1, (0, 0, 0), 1)  # -1 means all the counters

            # print(contours[0]) if contours[0] is not None else print("Pass")
            if len(contours) > 40:
                # for contour in contours:
                # position = cv2.contourArea(contour)
                blue_area = max(contours, key=cv2.contourArea)
                (xg, yg, wg, hg) = cv2.boundingRect(blue_area)
                cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

            if not ret:
                print("failed to grab frame")
                break

            # Show frame rate
            cTime = time.time()
            fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            cv2.putText(mask, str(int(fps)), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.putText(blurred_frame, str(int(fps)), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            cv2.imshow("B&W RobotSoccer\tHit Escape to Exit", mask)  # BLack and White Image
            # cv2.imshow("RobotSoccer\tHit Escape to Exit", np.vstack(result))  # Normal Images with contours
            cv2.imshow("RobotSoccer\tHit Escape to Exit", result[0])

            # cv2.waitKey(1)
            k = cv2.waitKey(1)  # TODO: You can define fps here as well each 1 is 1ms
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
