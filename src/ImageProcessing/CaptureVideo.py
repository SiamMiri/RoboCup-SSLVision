import cv2
import json
import time
import logging

# Logging Config
logging.basicConfig(filename="RoboCupLoggingFile", level=logging.DEBUG) #encoding="utf-8",


class Capture_Video():
    LOG_FPS = False # Set to True to log the FPS of reading Video 
    def __init__(self) -> None:

        # Define Variable for Class Object
        self.camera_config = None # Load json camera config file

        # Get Camera
        self.camera_capture = cv2.VideoCapture(0)

        """ Load Json File For Setting Camera Configuration """
        self.load_json_config_file()

        """ Start capturing get camera config from start_process_command """
        self.set_camera_config(self.camera_config['CameraConfig'], Fps=False, Res=True, Focus=False)

    def __del__(self) -> None:
        # Release Camera
        self.finish_capturing()
        print("Camera Released")

    ##########################################
    # start image capturing
    ##########################################
    def start_video_capturing(self): # FRAME_FROM_CAM
        if Capture_Video.LOG_FPS:
            startTime =  time.time()
        ret, frame = self.camera_capture.read() # FIXME: Changed to load Image
        
        if not ret:
            print("failed to grab frame")
    
        if Capture_Video.LOG_FPS:
            endTime = time.time()
            logging.info(f'Current Resolution is:    {len(frame[0])} {len(frame[1])}')
            logging.info(f'Time takes to read frame: {endTime - startTime}')
            logging.info(f'Raw FPS:                  {1/(endTime - startTime)}\n\n')

        if ret:
            return frame
        else:
            return None

    ##########################################
    # destroy opencv camera and free the camera 
    ##########################################
    def finish_capturing(self):
        # Release Camera
        self.camera_capture.release()
        pass

    ##########################################
    # set camera configuration
    ##########################################
    def set_camera_config(self, Fps=False, Res=False, Focus=False):
        """_summary_

        Args:
            Fps (bool, optional): _description_. Defaults to False.   set to True to change FPS of the Camera
            Res (bool, optional): _description_. Defaults to False.   set to True to change Resolution of the Camera
            Focus (bool, optional): _description_. Defaults to False. set to True to change FPS of reading image
        """        
        try:
            if isinstance(Fps, bool) and isinstance(Res, bool) and isinstance(Focus, bool):
                
                if Fps is not False:
                    # Change FPS
                    if isinstance(self.camera_config["CameraConfig"]["FPS"], int):
                        self.camera_capture.set(cv2.CAP_PROP_FPS, self.camera_config["CameraConfig"]["FPS"])  # set camera FPS from Json file
                        print("FPS Set")
                    elif isinstance(self.camera_config["CameraConfig"]["FPS"], float):
                        self.camera_capture.set(cv2.CAP_PROP_FPS, int(self.camera_config["CameraConfig"]["FPS"]))  # set camera FPS from Json file
                        print("FPS Set")
                    else:
                        print("FPS Configuration is incorrect")
                else:
                    print("FPS is not set")

                if Res is not False:

                    # Change Resolution
                    if isinstance(self.camera_config["CameraConfig"]["Resolution"], list):
                        # set camera Resolution from Json file
                        self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.camera_config["CameraConfig"]["Resolution"][0]))
                        self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.camera_config["CameraConfig"]["Resolution"][1]))
                        print("Resolution Set")
                    else:
                        print("Resolution Configuration is incorrect")
                else:
                    print("Resolution is not Set")

                if Focus is not False:
                    # Change Focus
                    if isinstance(self.camera_config["CameraConfig"]["focus"], int):
                        # set camera Resolution from Json file
                        self.camera_capture.set(cv2.CAP_PROP_FOCUS, self.camera_config["CameraConfig"]["focus"])
                        print("focus Set")
                    elif isinstance(self.camera_config["CameraConfig"]["focus"], float):
                        # set camera Resolution from Json file
                        # self.camera_capture.set(cv2.CAP_PROP_FOCUS, int(self.camera_config["CameraConfig"]["focus"])) // This may not work
                        self.camera_capture.set(28, int(self.camera_config["CameraConfig"]["focus"]))
                        print("focus Set")
                    else:
                        print("Focus Configuration is incorrect")
                else:
                    print("Focus is not Set")
            else:
                print("Set Boolean value for Camera filter setting ")
        
        except Exception as e:
            print(f'Set Camera Config Failed {e}')
    
    ##########################################
    # load json file as dictionary in python
    ##########################################
    def load_json_config_file(self):
        """_summary_
        Loading json file
        """
        
        """ with this function you can load json file """
        try:
            # try to load the json file if exist
            with open("./src/Config/CameraConfig.json") as config_file:
                self.camera_config = json.load(config_file)

        # Catch Err in this case might be naming diff in json file and print defined
        except Exception as e:
            print(f"ERROR: Unable To Load Json File {e}")
            self.camera_config = None
