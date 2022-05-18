from __future__ import print_function
import cv2 as cv
import argparse

class HSV_COLOR_PICKER():
    max_value = 255
    max_value_H = 360//2
    low_H = 0
    low_S = 0
    low_V = 0
    high_H = max_value_H
    high_S = max_value
    high_V = max_value
    window_capture_name = 'Video Capture'
    window_detection_name = 'Object Detection'
    low_H_name  = 'Low H'
    low_S_name  = 'Low S'
    low_V_name  = 'Low V'
    high_H_name = 'High H'
    high_S_name = 'High S'
    high_V_name = 'High V'
    
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
        parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)

        args = parser.parse_args()

        self.capture = cv.VideoCapture(args.camera)

        cv.namedWindow(HSV_COLOR_PICKER.window_capture_name)
        cv.namedWindow(HSV_COLOR_PICKER.window_detection_name)

        cv.createTrackbar(HSV_COLOR_PICKER.low_H_name  , HSV_COLOR_PICKER.window_detection_name , HSV_COLOR_PICKER.low_H  , HSV_COLOR_PICKER.max_value_H , HSV_COLOR_PICKER.on_low_H_thresh_trackbar )
        cv.createTrackbar(HSV_COLOR_PICKER.high_H_name , HSV_COLOR_PICKER.window_detection_name , HSV_COLOR_PICKER.high_H , HSV_COLOR_PICKER.max_value_H , HSV_COLOR_PICKER.on_high_H_thresh_trackbar)
        cv.createTrackbar(HSV_COLOR_PICKER.low_S_name  , HSV_COLOR_PICKER.window_detection_name , HSV_COLOR_PICKER.low_S  , HSV_COLOR_PICKER.max_value   , HSV_COLOR_PICKER.on_low_S_thresh_trackbar )

        cv.createTrackbar(HSV_COLOR_PICKER.high_S_name , HSV_COLOR_PICKER.window_detection_name , HSV_COLOR_PICKER.high_S , HSV_COLOR_PICKER.max_value   , HSV_COLOR_PICKER.on_high_S_thresh_trackbar)
        cv.createTrackbar(HSV_COLOR_PICKER.low_V_name  , HSV_COLOR_PICKER.window_detection_name , HSV_COLOR_PICKER.low_V  , HSV_COLOR_PICKER.max_value   , HSV_COLOR_PICKER.on_low_V_thresh_trackbar )
        cv.createTrackbar(HSV_COLOR_PICKER.high_V_name , HSV_COLOR_PICKER.window_detection_name , HSV_COLOR_PICKER.high_V , HSV_COLOR_PICKER.max_value   , HSV_COLOR_PICKER.on_high_V_thresh_trackbar)
    
    def on_low_H_thresh_trackbar(self, val):
        global low_H
        global high_H
        low_H = val
        low_H = min(high_H-1, low_H)
        cv.setTrackbarPos(HSV_COLOR_PICKER.low_H_name, HSV_COLOR_PICKER.window_detection_name, low_H)
    def on_high_H_thresh_trackbar(self, val):
        global low_H
        global high_H
        high_H = val
        high_H = max(high_H, low_H+1)
        cv.setTrackbarPos(HSV_COLOR_PICKER.high_H_name, HSV_COLOR_PICKER.window_detection_name, high_H)
    def on_low_S_thresh_trackbar(self, val):
        global low_S
        global high_S
        low_S = val
        low_S = min(high_S-1, low_S)
        cv.setTrackbarPos(HSV_COLOR_PICKER.low_S_name, HSV_COLOR_PICKER.window_detection_name, low_S)
    def on_high_S_thresh_trackbar(self, val):
        global low_S
        global high_S
        high_S = val
        high_S = max(high_S, low_S+1)
        cv.setTrackbarPos(HSV_COLOR_PICKER.high_S_name, HSV_COLOR_PICKER.window_detection_name, high_S)
    def on_low_V_thresh_trackbar(self, val):
        global low_V
        global high_V
        low_V = val
        low_V = min(high_V-1, low_V)
        cv.setTrackbarPos(HSV_COLOR_PICKER.low_V_name, HSV_COLOR_PICKER.window_detection_name, low_V)
    def on_high_V_thresh_trackbar(self, val):
        global low_V
        global high_V
        high_V = val
        high_V = max(high_V, low_V+1)
        cv.setTrackbarPos(HSV_COLOR_PICKER.high_V_name, HSV_COLOR_PICKER.window_detection_name, high_V)

    def color_picker(self, frame):
        while True:
            
            ret, frame = self.capture.read()

            frame = cv.imread('Robot5_x841_y356.jpg')
            frame = cv.resize(frame, (740,480))

            #if frame is None:
                #break
            frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            frame_threshold = cv.inRange(frame_HSV, (HSV_COLOR_PICKER.low_H, HSV_COLOR_PICKER.low_S, HSV_COLOR_PICKER.low_V), (HSV_COLOR_PICKER.high_H, HSV_COLOR_PICKER.high_S, HSV_COLOR_PICKER.high_V))
            
            
            cv.imshow(HSV_COLOR_PICKER.window_capture_name, frame)
            cv.imshow(HSV_COLOR_PICKER.window_detection_name, frame_threshold)
            
            key = cv.waitKey(30)
            if key == ord('q') or key == 27:
                break