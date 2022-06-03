# Autor : Siamak Mirifar
# Liscence : Student project ( Open source Liscence)


# NOTE:
# TODO: 1- Find Robot Id base on the robot color
# TODO: 2- Find Robot location in the field
# TODO: 3- Find Robot oriantation in the field
# TODO: 4- Find Robot ball position (similar to robot position)
# TODO: 5- Make DOCUMETS from where the Idea of the method has been taken !! IMPORTANT
import time
# from ImageProcessing import Image_Processing
from ImageProcessing.CaptureImage import Capture_Image
# from ImageProcessing import Image_Processing
from RobotClassificaion.DetectRobotBall import Detect_Robot_Ball as DetectRobot
from ImageProcessing.HSVColorPicker import HSV_COLOR_PICKER as ColorPicker  
import cv2
DetectRobot.ROTATE_ROBAT_SINGLE_IMAGE = True
# # # Define Color of the Frame/Image
# # Image_Processing.GRAY_SCALE_IMAGE_PROCCESSING = False

# # # Load Class of Color Picker 
# # if Image_Processing.GRAY_SCALE_IMAGE_PROCCESSING == False:
# #     from HSVColorPicker import HSV_COLOR_PICKER as ColorPicker    
# # else:
# #     from GrayColorPicker import GRAY_COLOR_PICKER as ColorPicker
 


# Definition of slots in this class are for futures if there were need to use GUI
class Main:

    def __init__(self)  :
        self.img_counter        = 0
        self.robot_id           = None                # Robot Id base on Color
        self.robot_location     = None                # Robot location on the field
        self.robot_orientation  = None                # Robot orientation

    def __del__(self):
        """ Destroy all class methods """
        # finish_capturing_command(self)
        pass
    def slot_video_capturing(self):
        """ start capturing video from camera """
        start_capturing_command(self)
        
    def slot_image_capturing(self):
        """ start capturing image from path"""
        capturingImage  = Capture_Image(image_path= "FieldTest_AllLight_Off_Daylight(hight).jpg")
        # imageProcessing = Image_Processing()
        detectRobot     = DetectRobot()
        
        # Creat windows for the frame
        cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")

        # try:
        while True:
            filed_frame = capturingImage.show_image(input_image=None)
            # imageProcessing.start_process(frame= filed_frame)
            angle = detectRobot.detect_robot(frame=filed_frame)
            cv2.imshow("RobotSoccer\tHit Escape or Q to Exit", filed_frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            
            if k % 256 == ord("q"):
                # Q pressed
                print("Escape hit, closing...")
                break
        # except Exception as e:
        #     print(f'Error in Capture_Image::show_frame {e}')

    def slot_finish_capturing(self):
        """ Delete Cv2 class to empty the buffer """
        finish_capturing_command(self)
        
 
if __name__ == "__main__":   
    # cam = Main()
    # cam.slot_video_capturing()
    colorpicker = ColorPicker()
    Color_Selection = colorpicker.color_picker()
    time.sleep(1)
    if Color_Selection:
        main = Main()
        main.slot_image_capturing()
