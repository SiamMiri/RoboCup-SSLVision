# import ImageProcessing
import cv2
import logging
import time

class Capture_Image():
    def __init__(self, image_path = None) -> None:
        # Define in __init__ of object
        self.image_path  = image_path

        # Object variable which needed
        self.frame = None

    def __del__(self):
        pass
    
    def load_image(self, image_path = None):
        
        startTime = time.time()
       
        if image_path is not None:
            self.frame = cv2.imread(image_path)
        else:
            self.frame = cv2.imread(self.image_path)
        
        endTime = time.time()
        
        logging.info(f'Time takes read Image: {endTime - startTime}')
        logging.info(f'FPS Read Image       : {1/(endTime - startTime)}\n\n')
        
        return self.frame
    
    def show_image(self, input_image = None):
        
        if input_image is None:
            if self.frame is None:
                self.load_image() 
            return self.frame
        else: 
            return input_image 
