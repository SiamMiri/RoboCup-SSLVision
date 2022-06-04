# Autor : Siamak Mirifar
# Liscence : Student project ( Open source Liscence)


# NOTE:
# TODO: 1- Find Robot Id base on the robot color
# TODO: 2- Find Robot location in the field
# TODO: 3- Find Robot oriantation in the field
# TODO: 4- Find Robot ball position (similar to robot position)
# TODO: 5- Make DOCUMETS from where the Idea of the method has been taken !! IMPORTANT
import time

from cv2 import VideoCapture
from ImageProcessing.CaptureImage import Capture_Image
from ImageProcessing.CaptureVideo import Capture_Video
from RobotClassificaion.DetectRobotBall import Detect_Robot_Ball as DetectRobot
from ImageProcessing.HSVColorPicker import HSV_COLOR_PICKER as ColorPicker
from MainGui.MainWindow import Ui_MainWindow  
import cv2
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5 import QtCore, QtWidgets
import sys

DetectRobot.ROTATE_ROBAT_SINGLE_IMAGE = True
 


# Definition of slots in this class are for futures if there were need to use GUI
class Main(QMainWindow, Ui_MainWindow):

    def __init__(self)  :
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.img_counter        = 0
        self.robot_id           = None                # Robot Id base on Color
        self.robot_location     = None                # Robot location on the field
        self.robot_orientation  = None                # Robot orientation
        self.connectMe()
        
    def __del__(self):
        """ Destroy all class methods """
        self.slot_finish_capturing()
    
    # Define All Signals Here
    def connectMe(self):
        self.btn_LoadImageFile.clicked.connect(self.slotLoadImageFile)
        self.btn_StartImageCapturing.clicked.connect(self.slotImageCapturing)
        self.btn_StartVideoCapturing.clicked.connect(self.slotVideoCapturing)
        self.btn_ColorConfiguration.clicked.connect(self.slotColorConfiguration)
    
    # Define All Slots Here
    def slotLoadImageFile(self):
        filepath = ""
        filepath = QFileDialog.getOpenFileName(self, "Choose The Image File", "", "Image File (*.jpg .*JPG)")
        print(filepath)
        if filepath != "":
            self.txt_FilePath.setText(str(filepath[0]))
    
    def slotImageCapturing(self):
        self.image_capturing(path= self.txt_FilePath.toPlainText())
        
    def slotVideoCapturing(self):
        self.video_capturing()
        
    def slotColorConfiguration(self):
        self.Color_Configuration()
        
    def video_capturing(self):
        """ start capturing video from camera """
        capturingVideo  = Capture_Video()
        # imageProcessing = Image_Processing()
        detectRobot     = DetectRobot()
        
        # Creat windows for the frame
        cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")


        try:
            while True:
                filed_frame = capturingVideo.load_image(image_path="./ImageSample/FieldTest_AllLight_Off_Daylight(hight).jpg")
                # imageProcessing.start_process(frame= filed_frame)
                DetectRobot.SEND_DATA_TO_SERVER = True
                detectRobot.detect_robot(frame=filed_frame)
                
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
        except Exception as e:
            print(f'Error in Capture_Image::show_frame {e}')
        
    def image_capturing(self, path:str=None):
        """ start capturing image from path"""
        capturingImage  = Capture_Image(image_path= None)
        # imageProcessing = Image_Processing()
        detectRobot     = DetectRobot()
        
        # Creat windows for the frame
        cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")

        # try:
        while True:
            filed_frame = capturingImage.load_image(image_path=path)
            # imageProcessing.start_process(frame= filed_frame)
            DetectRobot.SEND_DATA_TO_SERVER = True
            detectRobot.detect_robot(frame=filed_frame)
            
            cv2.imshow("RobotSoccer\tHit Escape or Q to Exit", filed_frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                break
            
            if k % 256 == ord("q"):
                # Q pressed
                print("Escape hit, closing...")
                cv2.destroyAllWindows()
                break
        # except Exception as e:
        #     print(f'Error in Capture_Image::show_frame {e}')

    def Color_Configuration(self):
        colorpicker = ColorPicker()
        Color_Selection = colorpicker.color_picker()

    def slot_finish_capturing(self):
        """ Delete Cv2 class to empty the buffer """
        cv2.destroyAllWindows()
        
 
if __name__ == "__main__":   
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
