# Autor : Siamak Mirifar
# Liscence : Student project ( Open source Liscence)


# NOTE:
# TODO: 1- Find Robot Id base on the robot color
# TODO: 2- Find Robot location in the field
# TODO: 3- Find Robot oriantation in the field
# TODO: 4- Find Robot ball position (similar to robot position)
# TODO: 5- Make DOCUMETS from where the Idea of the method has been taken !! IMPORTANT
from ControlCommand import *
import time
from ImageProcessing import Image_Processing

# Define Color of the Frame/Image
Image_Processing.GRAY_SCALE_IMAGE_PROCCESSING = False

# Load Class of Color Picker 
if Image_Processing.GRAY_SCALE_IMAGE_PROCCESSING == False:
    from HSVColorPicker import HSV_COLOR_PICKER as ColorPicker    
else:
    from GrayColorPicker import GRAY_COLOR_PICKER as ColorPicker
 


# Definition of slots in this class are for futures if there were need to use GUI
class CaptureImage:

    def __init__(self)  :
        self.img_counter        = 0
        self.cameraConfig       = load_json_config_file()  # Loading Camera Configuration from Json File
        self.robot_id           = None                         # Robot Id base on Color
        self.robot_location     = None                   # Robot location on the field
        self.robot_orientation  = None                # Robot orientation
        self.imgProc            = Image_Processing()

    def __del__(self):
        """ Destroy All the class objects """
        # finish_capturing_command(self)
        pass
    def slot_start_capturing(self):
        """ start capturing """
        start_capturing_command(self)

    def slot_finish_capturing(self):
        """ Delete Cv2 class to empty the buffer """
        finish_capturing_command(self)
        pass
 
if __name__ == "__main__":   
    # cam = CaptureImage()
    # cam.slot_start_capturing()
    colorpicker = ColorPicker()
    Color_Selection = colorpicker.color_picker()
    time.sleep(1)
    if Color_Selection:
        cam = CaptureImage()
        cam.slot_start_capturing()
