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
import os.path
from os import path
from ImageProcessing.CaptureImage import Capture_Image
from ImageProcessing.CaptureVideo import Capture_Video
from RobotClassificaion.DetectRobotBall import Detect_Robot_Ball as DetectRobot
from ImageProcessing.HSVColorPicker import HSV_COLOR_PICKER as ColorPicker
from MainGui.MainWindow import Ui_MainWindow  
import cv2
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox
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
        self.msg                = QMessageBox()
        
    def __del__(self):
        """ Destroy all class methods """
        self.slot_finish_capturing()
    
    # Define All Signals Here
    def connectMe(self):
        self.btn_LoadImageFile.clicked.connect(self.slotLoadImageFile)
        self.btn_StartImageCapturing.clicked.connect(self.slotImageCapturing)
        self.btn_StartVideoCapturing.clicked.connect(self.slotVideoCapturing)
        self.btn_ImageColorConfiguration.clicked.connect(self.slotImageColorConfiguration)
        self.btn_VideoColorConfiguration.clicked.connect(self.slotVideoColorConfiguration)
    
    # Define All Slots Here
    def slotLoadImageFile(self):
        filepath = ""
        filepath = QFileDialog.getOpenFileName(self, "Choose The Image File", "", "Image File (*.jpg .*JPG)")
        if filepath != "":
            self.txt_FilePath.setText(str(filepath[0]))
    
    def slotImageCapturing(self):
        self.image_capturing(img_path= self.txt_FilePath.toPlainText())
        
    def slotVideoCapturing(self):
        self.video_capturing()
        
    def slotImageColorConfiguration(self):
        self.image_color_configuration()
        
    def slotVideoColorConfiguration(self):
        self.video_color_configuration()
        
    # Methods calls by slots
    def video_capturing(self):
        """ start capturing video from camera """
        capturingVideo  = Capture_Video()
        capturingVideo.load_json_config_file()
        capturingVideo.set_camera_config(Fps=False, Res=True, Focus=False)
        detectRobot     = DetectRobot()
        
        # Creat windows for the frame
        cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")

        while True:
            field_frame = capturingVideo.start_video_capturing()
            if field_frame is None:
                break
            # imageProcessing.start_process(frame= field_frame)
            DetectRobot.SEND_DATA_TO_SERVER = True
            detectRobot.detect_robot(frame=field_frame)
            
            cv2.imshow("RobotSoccer\tHit Escape or Q to Exit", field_frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            
            if k % 256 == ord("q"):
                # Q pressed
                print("Escape hit, closing...")
                break
        
    def image_capturing(self, img_path:str=None):
        """ start capturing image from path"""
        if len(img_path) != 0:
            if path.exists(img_path):
                capturingImage  = Capture_Image(image_path= None)
                # imageProcessing = Image_Processing()
                detectRobot     = DetectRobot()
                # Creat windows for the frame
                cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")

                # try:
                while True:
                    field_frame = capturingImage.load_image(image_path=img_path)
                    # imageProcessing.start_process(frame= field_frame)
                    DetectRobot.SEND_DATA_TO_SERVER = True
                    detectRobot.detect_robot(frame=field_frame)
                    
                    cv2.imshow("RobotSoccer\tHit Escape or Q to Exit", field_frame)
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
            else:
                self.msg.setIcon(QMessageBox.Information)
                self.msg.setText("The Image Path does not exist, PLEASE REENTER YOUR FILE PATH !!")
                self.msg.setWindowTitle("Error")
                self.msg.setStandardButtons(QMessageBox.Ok)
                self.msg.show()
        else:
            self.msg.setIcon(QMessageBox.Information)
            self.msg.setText("The Image Path is Empty, PLEASE LOAD YOUR FILE !!")
            self.msg.setWindowTitle("Error")
            self.msg.setStandardButtons(QMessageBox.Ok)
            self.msg.show()
            
    def video_color_configuration(self):
        colorpicker = ColorPicker(image_path= None)
        colorpicker.color_picker()
                
    def image_color_configuration(self):
        imgPath = self.txt_FilePath.toPlainText()
        if len(imgPath) != 0:
            if path.exists(imgPath):
                colorpicker = ColorPicker(image_path= imgPath)
                colorpicker.color_picker()
            else:
                    self.msg.setIcon(QMessageBox.Information)
                    self.msg.setText("The Image Path does not exist, PLEASE REENTER YOUR FILE PATH !!")
                    self.msg.setWindowTitle("Error")
                    self.msg.setStandardButtons(QMessageBox.Ok)
                    self.msg.show()
        else:
            self.msg.setIcon(QMessageBox.Information)
            self.msg.setText("The Image Path is Empty, PLEASE LOAD YOUR FILE !!")
            self.msg.setWindowTitle("Error")
            self.msg.setStandardButtons(QMessageBox.Ok)
            self.msg.show()

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
