import cv2 as cv
import numpy as np
import json
import time
# pip install opencv-contrib-python

class GRAY_COLOR_PICKER():
    def __init__(self):
        self.max_value_H = 255
        self.low_H = 0
        self.high_H = self.max_value_H
        self.window_capture_name = 'Video Capture'
        self.window_detection_name = 'Object Detection'
        self.low_H_name   = 'Low Color'
        self.high_H_name  = 'High Color'
        self.cap = cv.VideoCapture(0)
        
    
    def on_low_H_thresh_trackbar(self, val):
        self.low_H = val
        self.low_H = min(self.high_H-1, self.low_H)
        cv.setTrackbarPos(self.low_H_name, self.window_detection_name, self.low_H)
    
    def on_high_H_thresh_trackbar(self, val):
        
        self.high_H = val
        self.high_H = max(self.high_H, self.low_H+1)
        cv.setTrackbarPos(self.high_H_name, self.window_detection_name, self.high_H)
    
    
    def mouse_clicked(self, event, x, y, flags, params):
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
                    with open('Robo_Color_Config.json', 'r') as file:
                        data = json.load(file)
                        data['Low_Red'] = [self.low_H]
                        data['Up_Red']  = [self.high_H]
                    
                    with open('Robo_Color_Config.json', 'w') as file:
                        json.dump(data, file, indent=2)
                        print("Red HSV Color Range Set")
                
                except Exception as err:
                    print(f'Could not find .json file {err}')
                
            if inside_green > 0 :
                try:
                    with open('Robo_Color_Config.json', 'r') as file:
                        data = json.load(file)
                        data['Low_Green'] = [self.low_H]
                        data['Up_Green']  = [self.high_H]
                    
                    with open('Robo_Color_Config.json', 'w') as file:
                        json.dump(data, file, indent=2)
                        print("Green HSV Color Range Set")
                
                except Exception as err:
                    print(f'Could not find .json file {err}')
                
            if inside_orange > 0 :
                try:
                    with open('Robo_Color_Config.json', 'r') as file:
                        data = json.load(file)
                        data['Low_Orange'] = [self.low_H]
                        data['Up_Orange']  = [self.high_H]
                    
                    with open('Robo_Color_Config.json', 'w') as file:
                        json.dump(data, file, indent=2)
                        print("Orange HSV Color Range Set")
                
                except Exception as err:
                    print(f'Could not find .json file {err}')
                
            if inside_blue > 0 :
                try:
                    with open('Robo_Color_Config.json', 'r') as file:
                        data = json.load(file)
                        data['Low_Blue'] = [self.low_H]
                        data['Up_Blue']  = [self.high_H]
                    
                    with open('Robo_Color_Config.json', 'w') as file:
                        json.dump(data, file, indent=2)
                        print("Blue HSV Color Range Set")
                
                except Exception as err:
                    print(f'Could not find .json file {err}')

    def mouse_call_back(self, fram_name):
        return cv.setMouseCallback(fram_name, self.mouse_clicked)
    
    def color_picker(self):
        cv.namedWindow(self.window_capture_name)
        cv.namedWindow(self.window_detection_name)

        cv.createTrackbar(self.low_H_name  , self.window_detection_name , self.low_H  , self.max_value_H , self.on_low_H_thresh_trackbar )
        cv.createTrackbar(self.high_H_name , self.window_detection_name , self.high_H , self.max_value_H , self.on_high_H_thresh_trackbar)

        time.sleep(2)

        while True:
            try:
                # ret, frame = self.cap.read() 
                frame = cv.imread('FieldTest_AllLight_Off_Daylight(hight).jpg', cv.IMREAD_GRAYSCALE)
                frame = cv.resize(frame, (740,480))

                # frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                # frame_threshold = cv.inRange(frame, (self.low_H, self.low_S, self.low_V), (self.high_H, self.high_S, self.high_V))
                frame_threshold = cv.inRange(frame, (self.low_H), (self.high_H))
                
                # Creat Button
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
                
                self.mouse_call_back(self.window_capture_name)
                
                key = cv.waitKey(30)
                if key == ord('q') or key == 27:
                    # cv.destroyAllWindows()
                    cv.destroyWindow(self.window_capture_name)
                    cv.destroyWindow(self.window_detection_name)
                    self.cap.release()
                    return True
            except Exception as err:
                print(err)
                # cv.destroyAllWindows()
                cv.destroyWindow(self.window_capture_name)
                cv.destroyWindow(self.window_detection_name)
                self.cap.release()
                return False