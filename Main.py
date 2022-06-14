# Autor : Siamak Mirifar
# Liscence : Student project ( Open source Liscence)


# NOTE:
# TODO: 1- Find Robot Id base on the robot color
# TODO: 2- Find Robot location in the field
# TODO: 3- Find Robot oriantation in the field
# TODO: 4- Find Robot ball position (similar to robot position)
# TODO: 5- Make DOCUMETS from where the Idea of the method has been taken !! IMPORTANT

# TODO: 6- Record Video (e.x for one min), save image with frame number 

import time
import logging
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
import multiprocessing as mp
import concurrent.futures
import sys

logging.basicConfig(filename="RoboCupLoggingFile", level=logging.DEBUG) #encoding="utf-8",

DetectRobot.ROTATE_ROBAT_SINGLE_IMAGE = True


# Definition of slots in this class are for futures if there were need to use GUI
class Main(QMainWindow, Ui_MainWindow):
    
    DICT_SSL_LIST   = mp.Queue()
    CROP_FRAME_DICT = mp.Queue()
    SHOW_CROP_IMAGE = True

    def __init__(self)  :
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.img_counter        = 0
        self.robot_id           = None                # Robot Id base on Color
        self.robot_location     = None                # Robot location on the field
        self.robot_orientation  = None                # Robot orientation
        self.mainFieldWindow    = None
        
        self.txt_FilePath.setPlainText("/home/siamakmirifar/Documents/Rosenheim/ThirdSemmester/MasterProjekt/server_robot_vision/ImageSample/FieldTest_AllLight_Off_Daylight(hight).jpg")


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
    
    def slot_finish_capturing(self):
        """ Delete Cv2 class to empty the buffer """
        # cv2.destroyAllWindows()
        # cv2.destroyWindow(self.mainFieldWindow)    
        # self.mainFieldWindow = None
        pass 

    def deactive(self):
        self.btn_ImageColorConfiguration.setEnabled(False)
        self.btn_LoadImageFile.setEnabled(False)
        self.btn_StartImageCapturing.setEnabled(False)
        self.btn_StartVideoCapturing.setEnabled(False)
        self.btn_VideoColorConfiguration.setEnabled(False)
        self.checkBox_ShowRobotindividualWindow.setEnabled(False)
        self.txt_FilePath.setEnabled(False)
        self.txt_NumFrameToSave.setEnabled(False)
        self.txt_NumRoboImageToSave.setEnabled(False)

    def active(self):
        self.btn_ImageColorConfiguration.setEnabled(True)
        self.btn_LoadImageFile.setEnabled(True)
        self.btn_StartImageCapturing.setEnabled(True)
        self.btn_StartVideoCapturing.setEnabled(True)
        self.btn_VideoColorConfiguration.setEnabled(True)
        self.checkBox_ShowRobotindividualWindow.setEnabled(True)
        self.txt_FilePath.setEnabled(True)
        self.txt_NumFrameToSave.setEnabled(True)
        self.txt_NumRoboImageToSave.setEnabled(True)
        
    # Methods calls by slots
    def video_capturing(self):
        frameIdx = 0
        self.deactive()
        """ start capturing video from camera """
        try:
            capturingVideo  = Capture_Video()
            capturingVideo.load_json_config_file()
            capturingVideo.set_camera_config(Fps=True, Res=True, Focus=False)
            
            DetectRobot.SHOW_CROPED_IMAGE = Main.SHOW_CROP_IMAGE     


            # Creat windows for the frame
            self.mainFieldWindow = cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")
            while True:
                
                timeStartProcess = time.time()
                    
                # if Main.DICT_SSL_LIST.empty():
                #     print("PASSED_5")
                #     continue
                # print("PASSED_6")
                # field_frame = Main.DICT_SSL_LIST.get()
                field_frame = capturingVideo.start_video_capturing()# DICT_SSL_LIST
                
                if field_frame is None:
                    pass
                else:
                    startTime = time.time()
                    detectRobot     = DetectRobot(SslListQueue=Main.DICT_SSL_LIST, CropImageQueue = Main.CROP_FRAME_DICT, frameIdx = frameIdx)
                    detectRobot.detect_robot(frame=field_frame)
                    detectRobot.join()
                    endTime = time.time()
                    
                    # if Main.DICT_SSL_LIST.empty():
                    #     continue
                    # Main.DICT_SSL_LIST.get()
                    # cropImagList = Main.CROP_FRAME_DICT.get()
                    # for key in cropImagList:
                    #     cv2.namedWindow(f"RobotSoccer_Robot{key}\tHit Escape or Q to Exit")
                    #     cv2.imshow(f"RobotSoccer_Robot{key}\tHit Escape or Q to Exit", cropImagList[key])
                    
                    logging.info(f'Time takes for Image Processing: {endTime - startTime}')
                    logging.info(f'FPS Image Processing:            {1/(endTime - startTime)}\n\n')
                    
                    
                    cv2.imshow(self.mainFieldWindow, field_frame)
                
                timeEndProcess = time.time()
                logging.info(f'Time takes for whole Process: {timeEndProcess - timeStartProcess}')
                logging.info(f'FPS Process:                  {1/(timeEndProcess - timeStartProcess)}\n\n')
                
                k = cv2.waitKey(1)
                if k % 256 == 27:
                    # ESC pressed
                    self.slot_finish_capturing()
                    print("Escape hit, closing...")
                    break
                
                if k % 256 == ord("q"):
                    # Q pressed
                    print("Escape hit, closing...")
                    self.slot_finish_capturing()
                    break
            self.active()
        except Exception as e:
            print(e)

    def image_capturing(self, img_path:str=None):
        self.deactive()
        DetectRobot.SHOW_CROPED_IMAGE = Main.SHOW_CROP_IMAGE
        """ start capturing image from path"""
        if len(img_path) != 0:
            if path.exists(img_path):
                frameIdx = 0
                capturingImage  = Capture_Image(image_path= None)
                
                # Creat windows for the frame
                cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")

                while True:

                    
                    field_frame = capturingImage.load_image(image_path=img_path)
                    
                    startTime = time.time()
                    detectRobot     = DetectRobot(SslListQueue=Main.DICT_SSL_LIST, CropImageQueue = Main.CROP_FRAME_DICT, frameIdx = frameIdx)
                    detectRobot.detect_robot(frame=field_frame)
                    detectRobot.join()
                    endTime = time.time()
                    
                    logging.info(f'Time takes for Image Processing: {endTime - startTime}')
                    logging.info(f'FPS Image Processing:            {1/(endTime - startTime)}\n\n')
                    
                    while Main.DICT_SSL_LIST.empty():
                        pass
                    Main.DICT_SSL_LIST.get()
                    cropImagList = Main.CROP_FRAME_DICT.get()
                    # for key in cropImagList:
                    #     cv2.namedWindow(f"RobotSoccer_Robot{key}\tHit Escape or Q to Exit")
                    #     cv2.imshow(f"RobotSoccer_Robot{key}\tHit Escape or Q to Exit", cropImagList[key])
                    
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
                self.active()
            else:
                self.msg.setIcon(QMessageBox.Information)
                self.msg.setText("The Image Path does not exist, PLEASE REENTER YOUR FILE PATH !!")
                self.msg.setWindowTitle("Error")
                self.msg.setStandardButtons(QMessageBox.Ok)
                self.msg.show()
                self.active()
        else:
            self.msg.setIcon(QMessageBox.Information)
            self.msg.setText("The Image Path is Empty, PLEASE LOAD YOUR FILE !!")
            self.msg.setWindowTitle("Error")
            self.msg.setStandardButtons(QMessageBox.Ok)
            self.msg.show()
            self.active()
                    
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
        
 
if __name__ == "__main__":   
    
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
