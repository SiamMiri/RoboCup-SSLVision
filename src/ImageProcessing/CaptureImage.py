# Author : Siamak Mirifar

import cv2
import logging
import time

class Capture_Image():
    def __init__(self, image_path = None) -> None:
        """_summary_

        Args:
            image_path (_type_, optional): _description_. Defaults to None.
        """        
        
        # Object Variable 
        self.image_path  = image_path
        self.frame = None
    
    def load_image(self, image_path = None):
        """_summary_

        Args:
            image_path (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
            return loaded image from path
        """        
               
        if image_path is not None:
            self.frame = cv2.imread(image_path)
        else:
            self.frame = cv2.imread(self.image_path)
        
        return self.frame
