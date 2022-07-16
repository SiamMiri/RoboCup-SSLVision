# Author : Siamak Mirifar
# License : Student project ( Open source License)


# NOTE: Project Milestone
# TODO: Task 1- Find Robot Id base on the robot color
# TODO: Task 2- Find Robot location in the field
# TODO: Task 3- Find Robot orientation in the field
# TODO: Task 4- Find Robot ball position (similar to robot position)
# TODO: Task 5- Save image with frame number 


##### required libraries
from telnetlib import SE
import time
import logging
import os.path
from os import path
import multiprocessing as mp
import sys
import cv2 # Image Processing Lib
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox # GUI Lib
from PyQt5 import QtCore, QtWidgets # GUI Lib

# Import Classes and functions
from src.ImageProcessing.CaptureImage import Capture_Image
from src.ImageProcessing.CaptureVideo import Capture_Video
from src.RobotClassification.DetectRobotBall import Detect_Robot_Ball as DetectRobot
from src.ImageProcessing.HSVColorPicker import HSV_COLOR_PICKER as ColorPicker
from src.ImageProcessing.ImageProcessing import Image_Processing
from src.MainGui.MainWindow import Ui_MainWindow

# required submodules
try: 
    # Import Server Class
    from server_robot_client.UDPSockets_Connection.UDPSend import UDP_Send
    from server_robot_client.UDPSockets_Connection.UDPConnection import UDP_Connection

    # Status of loading server submodules 
    SERVER_STATUS = True
    
except Exception as e:
    print(f"The submodules for server doesn't exist {e}")
    # Could not import submodules, set server status to False
    SERVER_STATUS = False
    

# Logging Configuration
logging.basicConfig(filename="RoboCupLoggingFile", level=logging.DEBUG) #encoding="utf-8",


# Definition of slots in this class are for futures if there were need to use GUI
class Main(QMainWindow, Ui_MainWindow):
    
    DICT_SSL_LIST            = mp.Queue()
    LOG_FPS                  = False 
    SHOW_CROP_IMAGE          = True

    def __init__(self)  :
        QMainWindow.__init__(self)
        
        # Load Graphical User Interface
        self.setupUi(self)
        
        # Create windows for the showing frames
        self.FieldWindowName = None
        
        # Default Image File Path
        self.txt_FilePath.setPlainText("/home/siamakmirifar/Documents/Rosenheim/ThirdSemmester/MasterProjekt/server_robot_vision/ImageSample/FieldTest_Left_Light_On_Daylight(hight).jpg")
        
        # Set server information on Gui
        if SERVER_STATUS:
            self.lbl_Port.setText(f"PORT: {str(UDP_Connection.PORT)}")
            self.lbl_Group.setText(f"Group: {UDP_Connection.GROUP}")

        # Connect Signals to Slots
        self.connectMe()
        
        # Create Object of QMessageBox for handling errors
        self.msg                = QMessageBox()
        
    def __del__(self):
        """ Destroy all class methods """
        self.slot_finish_capturing()
    
    # Define All Signals Here
    def connectMe(self):
        """_summary_
        This Methods connect all Signals to the Slots
        """        
        self.btn_LoadImageFile.clicked.connect(self.slotLoadImageFile)
        self.btn_StartImageCapturing.clicked.connect(self.slotImageCapturing)
        self.btn_StartVideoCapturing.clicked.connect(self.slotVideoCapturing)
        self.btn_ImageColorConfiguration.clicked.connect(self.slotImageColorConfiguration)
        self.btn_VideoColorConfiguration.clicked.connect(self.slotVideoColorConfiguration)
    
################################# All Slots are Defined Here #################################
    def slotLoadImageFile(self):
        """_summary_
        Loading Image Directory
        """        
        filepath = ""
        filepath = QFileDialog.getOpenFileName(self, "Choose The Image File", "", "Image File (*.jpg .*JPG)")
        if filepath != "":
            self.txt_FilePath.setText(str(filepath[0]))
    
    def slotImageCapturing(self):
        """_summary_
        Connect slot slotImageCapturing to method image_capturing 
        """        
        self.image_capturing(img_path= self.txt_FilePath.toPlainText())
        
    def slotVideoCapturing(self):
        """_summary_
        Connect slot slotVideoCapturing to method video_capturing 
        """
        self.video_capturing()
        
    def slotImageColorConfiguration(self):
        """_summary_
        Connect slot slotImageColorConfiguration to method image_color_configuration 
        """
        self.image_color_configuration()
        
    def slotVideoColorConfiguration(self):
        """_summary_
        Connect slot slotVideoColorConfiguration to method video_color_configuration 
        """
        self.video_color_configuration()
   
################################# All Methods of Signals are Defined Here #################################
    def video_capturing(self):
        frameIdx = 0
        self.deactivate()
        
        """ start capturing video from camera """
        try:
            
            """ Read Camera Objects """
            capturingVideo  = Capture_Video()
            capturingVideo.load_json_config_file()
            capturingVideo.set_camera_config(Fps=True, Res=True, Focus=False)
            
            """ Create Objects """
            processImage    = Image_Processing()
            detectRobot     = DetectRobot(FuncRotateImage = processImage.rotate_image_by_degree, FuncfindContours = processImage.find_contours_mask,
                                            CircleArea = [processImage.area_of_circle_min, processImage.area_of_circle_max],
                                            FuncCalContoursArea = processImage.calculate_contours_area, FuncCalMoment = processImage.calculate_moment)# , CropImageQueue = Main.CROP_FRAME_DICT 


            # Create windows for the frame
            self.FieldWindowName = cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")
            
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
                    
                    # tsProcess = time.time()
                    # Main.QUEUE_FRAME.put(field_frame)
                    field_frame = processImage.set_image_filter(frame = field_frame , filterJsonFile = processImage.json_frame_config["FrameConfig"],
                                                                    Blur  = False,GaussianBlur = False , Segmentation = False,
                                                                    Res   = True)
                    Frame_Data = processImage._start_process(field_frame = field_frame)
                    # teProcess = time.time()
                    # print(f'\ntime for Img Process {Process-tsProccess}')
                    # print(f'fps for Img Process {1 // (teProccess-tsProccess)}\n')
                    
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
        """_summary_

        Args:
            img_path (str, optional):  _description_. Defaults to None, get the Image
            Path from it's slot.
        """
            
        # deactivate all unnecessary btn and txt field 
        self.deactivate()
        
        # Frame index for saving frame
        frameIdx = 0
        
        """ Create Objects """
        capturingImage      = Capture_Image(image_path= None)
        processImage        = Image_Processing()
        detectRobot         = DetectRobot(FuncRotateImage = processImage.rotate_image_by_degree, FuncfindContours = processImage.find_contours_mask,
                                            CircleArea = [processImage.area_of_circle_min, processImage.area_of_circle_max],
                                            FuncCalContoursArea = processImage.calculate_contours_area, FuncCalMoment = processImage.calculate_moment)
        

        # Check if submodules Server is Imported
        if SERVER_STATUS:
            if self.checkBox_ConnectServer.isChecked():
                sendDataToServer    = UDP_Send(Main.DICT_SSL_LIST) # Create Server Object
                sendDataToServer.start() # Start Process (it is MultiProcess class)
        
        """ start capturing image from path"""
        if len(img_path) != 0: # check if path is not None
            if path.exists(img_path): # check if path exist
                frameIdx, startCap = self.check_num_frame_to_save() # Frame Index End on 1
                
                strFieldName = "RobotSoccer Image Capturing\tHit Escape or Q to Exit" # Field name
                self.FieldWindowName = cv2.namedWindow(strFieldName, cv2.WINDOW_AUTOSIZE) # Naming Frame
                
                while startCap: # Start Image Processing loop       
                    
                    field_frame = capturingImage.load_image(image_path=img_path) # Load Image from Path
                    
                    # Set Img filters (load from json file)
                    field_frame = processImage.set_image_filter(frame = field_frame , filterJsonFile = processImage.json_frame_config["FrameConfig"],
                                                                    Blur  = False,GaussianBlur = False , Segmentation = False,
                                                                    Res   = True)
                    
                    
                    Frame_Data = processImage._start_process(field_frame = field_frame) # Image Processing

                    # Find Robots from processed image
                    DetectedRobot = detectRobot.detect_robot(frame_data= Frame_Data,
                                                             CircleArea = [processImage.area_of_circle_min, processImage.area_of_circle_max])
                    
                    # show robots ifo if checkbox is activated
                    if self.checkBox_ShowRobotsInfo.isChecked():
                        for i in DetectedRobot:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(field_frame, f'Id: {DetectedRobot[i][0]} ', (DetectedRobot[i][2],DetectedRobot[i][3]-40), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.putText(field_frame, f'Degree {DetectedRobot[i][1]}', (DetectedRobot[i][2],DetectedRobot[i][3]-30), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    # Save Image the image index reduced by one
                    if frameIdx != 0: 
                        processImage.saveFrame(field_frame, frameIdx)
                        frameIdx -= 1
                    
                    # Send data to server if checkbox is active
                    if self.checkBox_ConnectServer.isChecked():
                        Main.DICT_SSL_LIST.put(DetectedRobot)
                    
                    cv2.imshow(strFieldName, field_frame)
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
                
                # Active all deactivated btn and txt field
                self.active()
            
            else: # Msg box for checking img path
                self.msg.setIcon(QMessageBox.Information)
                self.msg.setText("The Image Path does not exist, PLEASE REENTER YOUR FILE PATH !!")
                self.msg.setWindowTitle("Error")
                self.msg.setStandardButtons(QMessageBox.Ok)
                self.msg.show()
                self.active() # Active all deactivated btn and txt field
        else: # Msg box for checking img path
            self.msg.setIcon(QMessageBox.Information)
            self.msg.setText("The Image Path is Empty, PLEASE LOAD YOUR FILE !!")
            self.msg.setWindowTitle("Error")
            self.msg.setStandardButtons(QMessageBox.Ok)
            self.msg.show()
            self.active() # Active all deactivated btn and txt field
      
    def video_color_configuration(self):
        """_summary_
        This function is use for video color config
        Config HSV color range for image processing
        """        
        colorPicker = ColorPicker(image_path= None)
        colorPicker.color_picker()
                
    def image_color_configuration(self):
        """_summary_
        This function is use for image color config
        Config HSV color range for image processing
        """        
        imgPath = self.txt_FilePath.toPlainText()
        if len(imgPath) != 0: # check img path len
            if path.exists(imgPath): # check img path exist
                colorPicker = ColorPicker(image_path= imgPath)
                colorPicker.color_picker()
            else: # Err Msg for wrong img path
                self.msg.setIcon(QMessageBox.Information)
                self.msg.setText("The Image Path does not exist, PLEASE REENTER YOUR FILE PATH !!")
                self.msg.setWindowTitle("Error")
                self.msg.setStandardButtons(QMessageBox.Ok)
                self.msg.show()
        else: # Err Msg for wrong img path
            self.msg.setIcon(QMessageBox.Information)
            self.msg.setText("The Image Path is Empty, PLEASE LOAD YOUR FILE !!")
            self.msg.setWindowTitle("Error")
            self.msg.setStandardButtons(QMessageBox.Ok)
            self.msg.show()
        
#################################  General Methods #################################
    def deactivate(self):
        """_summary_
        Deactivate part GUI which are not needed to be active during process
        """ 
        self.btn_ImageColorConfiguration.setEnabled(False)
        self.btn_LoadImageFile.setEnabled(False)
        self.btn_StartImageCapturing.setEnabled(False)
        self.btn_StartVideoCapturing.setEnabled(False)
        self.btn_VideoColorConfiguration.setEnabled(False)
        self.txt_FilePath.setEnabled(False)
        self.txt_NumFrameToSave.setEnabled(False)
        self.checkBox_SaveFrame.setEnabled(False)
        self.checkBox_ConnectServer.setEnabled(False)

    def active(self):
        """_summary_
        Active part GUI which where deactivated
        """
        self.btn_ImageColorConfiguration.setEnabled(True)
        self.btn_LoadImageFile.setEnabled(True)
        self.btn_StartImageCapturing.setEnabled(True)
        self.btn_StartVideoCapturing.setEnabled(True)
        self.btn_VideoColorConfiguration.setEnabled(True)
        self.txt_FilePath.setEnabled(True)
        self.txt_NumFrameToSave.setEnabled(True)
        self.checkBox_SaveFrame.setEnabled(True)
        self.checkBox_ConnectServer.setEnabled(True)
    
    def check_num_frame_to_save(self):
        """_summary_
        check frame index is correct integer value
        Returns:
            _type_: _description_
            frameIdx -> number of frame to be saved
            startCap -> if number of index is correct return True
        """        
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
    
    # Make GUI capable of working in different resolution
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
