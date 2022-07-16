# Author : Modified by Siamak Mirifar
# Sample available on following link:
# https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html

import cv2 as cv
import numpy as np
import json
import time

class HSV_COLOR_PICKER():
    def __init__(self, image_path = None):
        """_summary_

        Args:
            image_path (_type_, optional): _description_. Defaults to None. If the img path
            is defined the color picker will be set for thr img
        """
        
        # __init__ Variable of the Object 
        self.max_value = 255
        self.max_value_Hue = 360//2
        self.low_Hue = 0
        self.low_Saturation = 0
        self.low_Value = 0
        self.high_Hue = self.max_value_Hue
        self.high_Saturation = self.max_value
        self.high_Value = self.max_value
        self.window_capture_name    = 'Video Capture'
        self.window_detection_name  = 'Object Detection'
        self.window_res_name        = 'Result Image'
        self.low_Hue_name           = 'Low Hue'
        self.low_saturation_name    = 'Low Saturation'
        self.low_value_name         = 'Low Value'
        self.high_hue_name            = 'High Hue'
        self.high_saturation_name   = 'High Saturation'
        self.high_value_name        = 'High Value'
        self.cap = cv.VideoCapture(1)
        self.imgPath = image_path
    
    def on_low_H_thresh_trackbar(self, val):
        self.low_Hue = val
        self.low_Hue = min(self.high_Hue-1, self.low_Hue)
        cv.setTrackbarPos(self.low_Hue_name, self.window_detection_name, self.low_Hue)
    
    def on_high_H_thresh_trackbar(self, val):
        
        self.high_Hue = val
        self.high_Hue = max(self.high_Hue, self.low_Hue+1)
        cv.setTrackbarPos(self.high_hue_name, self.window_detection_name, self.high_Hue)
    
    def on_low_S_thresh_trackbar(self, val):
        
        self.low_Saturation = val
        self.low_Saturation = min(self.high_Saturation-1, self.low_Saturation)
        cv.setTrackbarPos(self.low_saturation_name, self.window_detection_name, self.low_Saturation)
    
    def on_high_S_thresh_trackbar(self, val):
        
        self.high_Saturation = val
        self.high_Saturation = max(self.high_Saturation, self.low_Saturation+1)
        cv.setTrackbarPos(self.high_saturation_name, self.window_detection_name, self.high_Saturation)
    
    def on_low_V_thresh_trackbar(self, val):
        
        self.low_Value = val
        self.low_Value = min(self.high_Value-1, self.low_Value)
        cv.setTrackbarPos(self.low_value_name, self.window_detection_name, self.low_Value)
    
    def on_high_V_thresh_trackbar(self, val):
        
        self.high_Value = val
        self.high_Value = max(self.high_Value, self.low_Value+1)
        cv.setTrackbarPos(self.high_value_name, self.window_detection_name, self.high_Value)

    def mouse_clicked(self, event, x, y, flags, params):
        """_summary_

        Args:
            event (_type_): _description_ mouse click event 
            x (_type_): _description_ x     position of the mouse clicked position
            y (_type_): _description_ y     position of the mouse clicked position
            flags (_type_): _description_   mouse event flags  
            params (_type_): _description_  mouse event parameters
        """        
        if event == cv.EVENT_LBUTTONDOWN:
            Red_area = np.array([[(20, 20), (270, 20), (270, 50), (20, 50)]])
            inside_red = cv.pointPolygonTest(Red_area, (x, y), False)
            
            Green_area = np.array([[(20, 70), (270, 70), (270, 100), (20, 100)]])
            inside_green = cv.pointPolygonTest(Green_area, (x, y), False)
            
            Orange_area = np.array([[(20, 120), (270, 120), (270, 150), (20, 150)]])
            inside_orange = cv.pointPolygonTest(Orange_area, (x, y), False)
            # (20, 170), (270, 200),
            Blue_area = np.array([[(20, 170), (270, 170), (270, 200), (20, 200)]])
            inside_blue = cv.pointPolygonTest(Blue_area, (x, y), False)

            if inside_red > 0 :
                try:
                    with open('./src/Config/Robo_Color_Config.json', 'r') as file:
                        data = json.load(file)
                        data['Low_Red'] = [self.low_Hue, self.low_Saturation, self.low_Value]
                        data['Up_Red']  = [self.high_Hue, self.high_Saturation, self.high_Value]
                    
                    with open('./src/Config/Robo_Color_Config.json', 'w') as file:
                        json.dump(data, file, indent=2)
                        print("Red HSV Color Range Set")
                
                except Exception as err:
                    print(f'Could not find .json file {err}')
                
            if inside_green > 0 :
                try:
                    with open('./src/Config/Robo_Color_Config.json', 'r') as file:
                        data = json.load(file)
                        data['Low_Green'] = [self.low_Hue, self.low_Saturation, self.low_Value]
                        data['Up_Green']  = [self.high_Hue, self.high_Saturation, self.high_Value]
                    
                    with open('./src/Config/Robo_Color_Config.json', 'w') as file:
                        json.dump(data, file, indent=2)
                        print("Green HSV Color Range Set")
                
                except Exception as err:
                    print(f'Could not find .json file {err}')
                
            if inside_orange > 0 :
                try:
                    with open('./src/Config/Robo_Color_Config.json', 'r') as file:
                        data = json.load(file)
                        data['Low_Orange'] = [self.low_Hue, self.low_Saturation, self.low_Value]
                        data['Up_Orange']  = [self.high_Hue, self.high_Saturation, self.high_Value]
                    
                    with open('./src/Config/Robo_Color_Config.json', 'w') as file:
                        json.dump(data, file, indent=2)
                        print("Orange HSV Color Range Set")
                
                except Exception as err:
                    print(f'Could not find .json file {err}')
                
            if inside_blue > 0 :
                try:
                    with open('./src/Config/Robo_Color_Config.json', 'r') as file:
                        data = json.load(file)
                        data['Low_Blue'] = [self.low_Hue, self.low_Saturation, self.low_Value]
                        data['Up_Blue']  = [self.high_Hue, self.high_Saturation, self.high_Value]
                    
                    with open('./src/Config/Robo_Color_Config.json', 'w') as file:
                        json.dump(data, file, indent=2)
                        print("Blue HSV Color Range Set")
                
                except Exception as err:
                    print(f'Could not find .json file {err}')

    def mouse_call_back(self, frame_name):
        return cv.setMouseCallback(frame_name, self.mouse_clicked)
    
    def gaussian_blur(self, img):
        img = cv.GaussianBlur(img, (15, 15), 0)    
        return img

    def median_blur(self, img):
        img = cv.medianBlur(img, 15)
        return img

    def color_picker(self):
        cv.namedWindow(self.window_capture_name)
        cv.namedWindow(self.window_detection_name)

        cv.createTrackbar(self.low_Hue_name  , self.window_detection_name , self.low_Hue  , self.max_value_Hue , self.on_low_H_thresh_trackbar )
        cv.createTrackbar(self.high_hue_name , self.window_detection_name , self.high_Hue , self.max_value_Hue , self.on_high_H_thresh_trackbar)
        cv.createTrackbar(self.low_saturation_name  , self.window_detection_name , self.low_Saturation  , self.max_value   , self.on_low_S_thresh_trackbar )

        cv.createTrackbar(self.high_saturation_name , self.window_detection_name , self.high_Saturation , self.max_value   , self.on_high_S_thresh_trackbar)
        cv.createTrackbar(self.low_value_name  , self.window_detection_name , self.low_Value  , self.max_value   , self.on_low_V_thresh_trackbar )
        cv.createTrackbar(self.high_value_name , self.window_detection_name , self.high_Value , self.max_value   , self.on_high_V_thresh_trackbar)
        time.sleep(2)

        while True:
            try:
                # ret, frame = self.cap.read()
                if self.imgPath != None: 
                    frame = cv.imread(self.imgPath)
                    frame = cv.resize(frame, (740,480))
                else:
                    ret, frame = self.cap.read()
                    
                frame_HSV       = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                frame_threshold = cv.inRange(frame_HSV, (self.low_Hue, self.low_Saturation, self.low_Value), (self.high_Hue, self.high_Saturation, self.high_Value))
                
                frame_res       = cv.bitwise_and(frame, frame, mask = frame_threshold)
                
                # Create Button for saving Color
                cv.rectangle(frame, (20, 20), (270, 50), (0, 0, 0), -1)
                cv.putText(frame, "Save Red HSV Range Color", (30, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                
                cv.rectangle(frame, (20, 70), (270, 100), (0, 0, 0), -1)
                cv.putText(frame, "Save Green HSV Range Color", (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                
                cv.rectangle(frame, (20, 120), (270, 150), (0, 0, 0), -1)
                cv.putText(frame, "Save Orange HSV Range Color", (30, 140), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                cv.rectangle(frame, (20, 170), (270, 200), (0, 0, 0), -1)
                cv.putText(frame, "Save Blue HSV Range Color", (30, 190), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                
                cv.imshow(self.window_capture_name, frame)
                cv.imshow(self.window_detection_name, frame_threshold)
                cv.imshow(self.window_res_name, frame_res)

                self.mouse_call_back(self.window_capture_name)
                
                key = cv.waitKey(30)
                if key == ord('q') or key == 27:
                    cv.destroyAllWindows()
                    self.cap.release()
                    return True
            except Exception as err:
                print(err)
                cv.destroyWindow(self.window_capture_name)
                cv.destroyWindow(self.window_detection_name)
                self.cap.release()
                return False