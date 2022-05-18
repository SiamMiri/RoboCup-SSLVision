from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import json
# pip install opencv-contrib-python

class HSV_COLOR_PICKER():
    
    def __init__(self) -> None:
        
        self.max_value = 255
        self.max_value_H = 360//2
        self.low_H = 0
        self.low_S = 0
        self.low_V = 0
        self.high_H = self.max_value_H
        self.high_S = self.max_value
        self.high_V = self.max_value
        self.window_capture_name = 'Video Capture'
        self.window_detection_name = 'Object Detection'
        self.low_H_name  = 'Low H'
        self.low_S_name  = 'Low S'
        self.low_V_name  = 'Low V'
        self.high_H_name = 'High H'
        self.high_S_name = 'High S'
        self.high_V_name = 'High V'
        
        
        parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
        parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)

        args = parser.parse_args()

        self.capture = cv.VideoCapture(args.camera)

        cv.namedWindow(self.window_capture_name)
        cv.namedWindow(self.window_detection_name)

        cv.createTrackbar(self.low_H_name  , self.window_detection_name , self.low_H  , self.max_value_H , self.on_low_H_thresh_trackbar )
        cv.createTrackbar(self.high_H_name , self.window_detection_name , self.high_H , self.max_value_H , self.on_high_H_thresh_trackbar)
        cv.createTrackbar(self.low_S_name  , self.window_detection_name , self.low_S  , self.max_value   , self.on_low_S_thresh_trackbar )

        cv.createTrackbar(self.high_S_name , self.window_detection_name , self.high_S , self.max_value   , self.on_high_S_thresh_trackbar)
        cv.createTrackbar(self.low_V_name  , self.window_detection_name , self.low_V  , self.max_value   , self.on_low_V_thresh_trackbar )
        cv.createTrackbar(self.high_V_name , self.window_detection_name , self.high_V , self.max_value   , self.on_high_V_thresh_trackbar)

        # cv.createButton(self.window_detection_name, self.back,None,cv.QT_PUSH_BUTTON,1)
        
    def back(*args):
        print("HI")
    
    def on_low_H_thresh_trackbar(self, val):
        self.low_H = val
        self.low_H = min(self.high_H-1, self.low_H)
        cv.setTrackbarPos(self.low_H_name, self.window_detection_name, self.low_H)
    
    def on_high_H_thresh_trackbar(self, val):
        
        self.high_H = val
        self.high_H = max(self.high_H, self.low_H+1)
        cv.setTrackbarPos(self.high_H_name, self.window_detection_name, self.high_H)
    
    def on_low_S_thresh_trackbar(self, val):
        
        self.low_S = val
        self.low_S = min(self.high_S-1, self.low_S)
        cv.setTrackbarPos(self.low_S_name, self.window_detection_name, self.low_S)
    
    def on_high_S_thresh_trackbar(self, val):
        
        self.high_S = val
        self.high_S = max(self.high_S, self.low_S+1)
        cv.setTrackbarPos(self.high_S_name, self.window_detection_name, self.high_S)
    
    def on_low_V_thresh_trackbar(self, val):
        
        self.low_V = val
        self.low_V = min(self.high_V-1, self.low_V)
        cv.setTrackbarPos(self.low_V_name, self.window_detection_name, self.low_V)
    
    def on_high_V_thresh_trackbar(self, val):
        
        self.high_V = val
        self.high_V = max(self.high_V, self.low_V+1)
        cv.setTrackbarPos(self.high_V_name, self.window_detection_name, self.high_V)

    def mouse_clicked(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            
            Red_area = np.array([[(20, 20), (270, 20), (270, 50), (20, 50)]])
            inside_red = cv.pointPolygonTest(Red_area, (x, y), False)
            
            Green_area = np.array([[(20, 70), (270, 70), (270, 100), (20, 100)]])
            inside_green = cv.pointPolygonTest(Green_area, (x, y), False)
            
            Blue_area = np.array([[(20, 120), (270, 120), (270, 150), (20, 150)]])
            inside_blue = cv.pointPolygonTest(Blue_area, (x, y), False)
            
            if inside_red > 0 :
                try:
                    with open('Robo_Color_Config.json', 'r') as file:
                        data = json.load(file)
                        data['Low_Red'] = [self.low_H, self.low_S, self.low_V]
                        data['Up_Red']  = [self.high_H, self.high_S, self.high_V]
                    
                    with open('Robo_Color_Config.json', 'w') as file:
                        json.dump(data, file, indent=2)
                
                except Exception as err:
                    print(f'Could not find .json file {err}')
                
            if inside_green > 0 :
                print("green")
                
            if inside_blue > 0 :
                print("blue")
                
    def save_hsv_color_range():
        pass
            
    def mouse_call_back(self, fram_name, are_red):
        return cv.setMouseCallback(fram_name, self.mouse_clicked)
    
    def color_picker(self, frame):
        
        while True:
            
            ret, frame = self.capture.read()

            frame = cv.imread('Robot5_x841_y356.jpg')
            frame = cv.resize(frame, (740,480))

            #if frame is None:
                #break
            frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, (self.low_H, self.low_S, self.low_V), (self.high_H, self.high_S, self.high_V))
            
            # Creat Button
            cv.rectangle(frame_threshold, (20, 20), (270, 50), (0, 0, 0), -1)
            click_area_red = np.array([[(20, 20), (270, 20), (270, 50), (20, 50)]])  
            cv.putText(frame_threshold, "Save Red HSV Range Color", (30, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            
            cv.rectangle(frame_threshold, (20, 70), (270, 100), (0, 0, 0), -1)
            cv.putText(frame_threshold, "Save Green HSV Range Color", (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            
            cv.rectangle(frame_threshold, (20, 120), (270, 150), (0, 0, 0), -1)
            cv.putText(frame_threshold, "Save Blue HSV Range Color", (30, 140), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            
            cv.imshow(self.window_capture_name, frame)
            cv.imshow(self.window_detection_name, frame_threshold)
            
            self.mouse_call_back(self.window_detection_name, are_red = click_area_red)
            
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                break