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
from ImageProcessing.ImageProcessing import Image_Processing
from MainGui.MainWindow import Ui_MainWindow  
import cv2
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from PyQt5 import QtCore, QtWidgets
import multiprocessing as mp
import concurrent.futures
import sys
from server_robot_client.UDPSockets_Connection.UDPSend import UDP_Send
from server_robot_client.UDPSockets_Connection.UDPConnection import UDP_Connection

logging.basicConfig(filename="RoboCupLoggingFile", level=logging.DEBUG) #encoding="utf-8",

DetectRobot.ROTATE_ROBAT_SINGLE_IMAGE = True


# Definition of slots in this class are for futures if there were need to use GUI
class Main(QMainWindow, Ui_MainWindow):
    
    DICT_SSL_LIST            = mp.Queue()
    LOG_FPS                  = False 
    SHOW_CROP_IMAGE          = True

    def __init__(self)  :
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.img_counter        = 0
        self.robot_id           = None                # Robot Id base on Color
        self.robot_location     = None                # Robot location on the field
        self.robot_orientation  = None                # Robot orientation
        self.mainFieldWindow    = None
        
        self.txt_FilePath.setPlainText("/home/robosoccer/Desktop/server_robot_vision/ImageSample/FieldTest_AllLight_Off_Daylight(hight).jpg")
        self.lbl_Port.setText(f"PORT: {str(UDP_Connection.PORT)}")
        self.lbl_Group.setText(f"Group: {UDP_Connection.GROUP}")

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
        
    # Methods calls by slots
    def video_capturing(self):
        frameIdx = 0
        self.deactive()
        DetectRobot.SHOW_CROPED_IMAGE = Main.SHOW_CROP_IMAGE
        
        """ start capturing video from camera """
        try:
            
            """ Read Camera Objects """
            capturingVideo  = Capture_Video()
            capturingVideo.load_json_config_file()
            capturingVideo.set_camera_config(Fps=True, Res=True, Focus=False)
            
            """ Creat Objects """
            processImage    = Image_Processing()
            detectRobot     = DetectRobot(FuncRotateImage = processImage.rotate_image_by_degree, FuncfindContours = processImage.find_contours_mask,
                                            CircleArea = [processImage.area_of_circle_min, processImage.area_of_circle_max],
                                            FuncCalContoursArea = processImage.calculate_contours_area, FuncCalMoment = processImage.calculate_moment)# , CropImageQueue = Main.CROP_FRAME_DICT 


            # Creat windows for the frame
            self.mainFieldWindow = cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")
            
            # Frame Index End on 1
            # Frame Index End on 1
            frameIdx, startCap = self.check_num_frame_to_save()
            
            while startCap:                    
                tsRead = time.time()
                field_frame = capturingVideo.start_video_capturing()# DICT_SSL_LIST
                
                if field_frame == None:
                    break
                else:
                    teRead = time.time()
                    print(f'\ntime for read image {teRead-tsRead}')
                    print(f'fps for read image {1 // (teRead-tsRead)}\n')
                    
                    # tsProccess = time.time()
                    # Main.QUEUE_FRAME.put(field_frame)
                    field_frame = processImage.set_image_filter(frame = field_frame , filterJsonFile = processImage.ConfigFrame["FrameConfig"],
                                                                    Blur  = False,GaussianBlur = False , Segmentation = False,
                                                                    Res   = True)
                    Frame_Data = processImage._start_process(field_frame = field_frame)
                    # teProccess = time.time()
                    # print(f'\ntime for Img Proccess {teProccess-tsProccess}')
                    # print(f'fps for Img Proccess {1 // (teProccess-tsProccess)}\n')
                    
                    # Main.QUEUE_FRAME_DATA.put(Frame_Data)
                    # startTime = time.time()
                    DetectedRobot = detectRobot.detect_robot(frame_data= Frame_Data, CircleArea = [processImage.area_of_circle_min, processImage.area_of_circle_max])
                    # detectRobot.join()
                    # endTime = time.time()
                    # print(f'\ntime for Detect {endTime-startTime}')
                    # print(f'fps for Detect {1 // (endTime-startTime)}\n')
                    
                    # logging.info(f'Time takes for Image Processing: {endTime - startTime}')
                    # logging.info(f'FPS Image Processing:            {1/(endTime - startTime)}\n\n')
                    
                    # while Main.DICT_SSL_LIST.empty():
                    #     pass
                    # a = Main.DICT_SSL_LIST.get()
                    # # cropImagList = Main.CROP_FRAME_DICT.get()
                    # for key in cropImagList:
                    #     cv2.namedWindow(f"RobotSoccer_Robot{key}\tHit Escape or Q to Exit")
                    #     cv2.imshow(f"RobotSoccer_Robot{key}\tHit Escape or Q to Exit", cropImagList[key])
                    if self.checkBox_ShowRobotsInfo.isChecked():
                        for i in DetectedRobot:
                            # print(a)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(field_frame, f'Id: {DetectedRobot[i][0]} ', (DetectedRobot[i][2],DetectedRobot[i][3]-40), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(field_frame, f'Degree {DetectedRobot[i][1]}', (DetectedRobot[i][2],DetectedRobot[i][3]-30), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    if frameIdx != 0: 
                        processImage.saveFrame(field_frame, frameIdx)
                        frameIdx -= 1
                
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
        frameIdx = 0
        DetectRobot.SHOW_CROPED_IMAGE = Main.SHOW_CROP_IMAGE
        
        """ Creat Objects """
        processImage        = Image_Processing()
        capturingImage      = Capture_Image(image_path= None)
        detectRobot         = DetectRobot(FuncRotateImage = processImage.rotate_image_by_degree, FuncfindContours = processImage.find_contours_mask,
                                            CircleArea = [processImage.area_of_circle_min, processImage.area_of_circle_max],
                                            FuncCalContoursArea = processImage.calculate_contours_area, FuncCalMoment = processImage.calculate_moment)# , CropImageQueue = Main.CROP_FRAME_DICT 
        
        if self.checkBox_ConnectServer.isChecked():
            sendDataToServer    = UDP_Send(Main.DICT_SSL_LIST)
            sendDataToServer.start()

        """ start capturing image from path"""
        if len(img_path) != 0:
            if path.exists(img_path):                       
                # Creat windows for the frame
                cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")
                
                # Frame Index End on 1
                frameIdx, startCap = self.check_num_frame_to_save()
                
                while startCap:       
                        
                    t1 = time.time()
                    # tsRead = time.time()
                    field_frame = capturingImage.load_image(image_path=img_path)
                    
                    # teRead = time.time()
                    # print(f'\ntime for read image {teRead-tsRead}')
                    # print(f'fps for read image {1 // (teRead-tsRead)}\n')
                    
                    # tsProccess = time.time()
                    # Main.QUEUE_FRAME.put(field_frame)
                    field_frame = processImage.set_image_filter(frame = field_frame , filterJsonFile = processImage.ConfigFrame["FrameConfig"],
                                                                    Blur  = False,GaussianBlur = False , Segmentation = False,
                                                                    Res   = True)
                    Frame_Data = processImage._start_process(field_frame = field_frame)
                    # teProccess = time.time()
                    # print(f'\ntime for Img Proccess {teProccess-tsProccess}')
                    # print(f'fps for Img Proccess {1 // (teProccess-tsProccess)}\n')
                    
                    # Main.QUEUE_FRAME_DATA.put(Frame_Data)
                    # startTime = time.time()
                    DetectedRobot = detectRobot.detect_robot(frame_data= Frame_Data, CircleArea = [processImage.area_of_circle_min, processImage.area_of_circle_max])
                    # detectRobot.join()
                    # endTime = time.time()
                    # print(f'\ntime for Detect {endTime-startTime}')
                    # print(f'fps for Detect {1 // (endTime-startTime)}\n')
                    
                    # logging.info(f'Time takes for Image Processing: {endTime - startTime}')
                    # logging.info(f'FPS Image Processing:            {1/(endTime - startTime)}\n\n')
                    
                    # while Main.DICT_SSL_LIST.empty():
                    #     pass
                    # a = Main.DICT_SSL_LIST.get()
                    # # cropImagList = Main.CROP_FRAME_DICT.get()
                    # for key in cropImagList:
                    #     cv2.namedWindow(f"RobotSoccer_Robot{key}\tHit Escape or Q to Exit")
                    #     cv2.imshow(f"RobotSoccer_Robot{key}\tHit Escape or Q to Exit", cropImagList[key])
                    if self.checkBox_ShowRobotsInfo.isChecked():
                        for i in DetectedRobot:
                            # print(a)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(field_frame, f'Id: {DetectedRobot[i][0]} ', (DetectedRobot[i][2],DetectedRobot[i][3]-40), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(field_frame, f'Degree {DetectedRobot[i][1]}', (DetectedRobot[i][2],DetectedRobot[i][3]-30), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    if frameIdx != 0: 
                        processImage.saveFrame(field_frame, frameIdx)
                        frameIdx -= 1
                    if self.checkBox_ConnectServer.isChecked():
                        Main.DICT_SSL_LIST.put(DetectedRobot)
                    # sendDataToServer.send(DetectedRobot)
                    cv2.imshow("RobotSoccer\tHit Escape or Q to Exit", field_frame)
                    k = cv2.waitKey(1)
                    t2 = time.time()
                    print(f'\ntime for all proccess takes {t2-t1}')
                    print(f'fps for all proccess takes {1 // (t2-t1)}\n')
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
        
    # General Methods              
    def deactive(self):
        self.btn_ImageColorConfiguration.setEnabled(False)
        self.btn_LoadImageFile.setEnabled(False)
        self.btn_StartImageCapturing.setEnabled(False)
        self.btn_StartVideoCapturing.setEnabled(False)
        self.btn_VideoColorConfiguration.setEnabled(False)
        # self.checkBox_ShowRobotsInfo.setEnabled(False)
        self.txt_FilePath.setEnabled(False)
        self.txt_NumFrameToSave.setEnabled(False)
        self.checkBox_SaveFrame.setEnabled(False)
        # self.checkBox_ConnectServer.setEnabled(False)

        # self.txt_NumRoboImageToSave.setEnabled(False)

    def active(self):
        self.btn_ImageColorConfiguration.setEnabled(True)
        self.btn_LoadImageFile.setEnabled(True)
        self.btn_StartImageCapturing.setEnabled(True)
        self.btn_StartVideoCapturing.setEnabled(True)
        self.btn_VideoColorConfiguration.setEnabled(True)
        # self.checkBox_ShowRobotsInfo.setEnabled(True)
        self.txt_FilePath.setEnabled(True)
        self.txt_NumFrameToSave.setEnabled(True)
        self.checkBox_SaveFrame.setEnabled(True)
        # self.checkBox_ConnectServer.setEnabled(True)
        # self.txt_NumRoboImageToSave.setEnabled(True)
    
    def check_num_frame_to_save(self):
        i = self.txt_NumFrameToSave.toPlainText()
        if i != None and self.checkBox_SaveFrame.isChecked():
            try:
                frameIdx = int(self.txt_NumFrameToSave.toPlainText())
                startCap = True
            except Exception as e:
                print("Error Input is not Integer {e}")
                frameIdx = 0
                startCap = False
                cv2.destroyAllWindows()
        else: 
            frameIdx = 0
            startCap = True
        
        return frameIdx, startCap
 
if __name__ == "__main__":   
    
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
