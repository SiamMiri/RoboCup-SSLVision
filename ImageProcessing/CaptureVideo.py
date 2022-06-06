from http.client import IM_USED
from ImageProcessing.ImageProcessing import Image_Processing
import cv2
import json
import time

# FIXME: Change the frame size would cause high FPS Drop


class Capture_Video():

    def __init__(self) -> None:

        # Define Variable for Class Object
        self.camera_config = None # Load json camera config file
        self.pTime         = 0 # Calculate FPS

        # Get Camera
        self.camera_capture = cv2.VideoCapture(0)

        """ Load Json File For Setting Camera Configuration """
        # self.load_json_config_file()

        """ Start capturing get camera config from start_process_command """
        # self.set_camera_config(self.camera_config['CameraConfig'], Fps=False, Res=True, Focus=False)

    def __del__(self) -> None:
        self.finish_capturing()
        print("Camera Released")

    ##########################################
    # start image capturing
    ##########################################
    def start_video_capturing(self):

        startTime =  time.time()
        ret, frame = self.camera_capture.read() # FIXME: Changed to load Image
        if not ret:
            print("failed to grab frame")
            return None
        endTime = time.time()
        
        print(f'passed time = {endTime - startTime}')
        print(f'FPS : {1/(endTime - startTime)}')
        return frame

    ##########################################
    # destroy opencv camera and free the camera 
    ##########################################
    def finish_capturing(self):
        cv2.destroyAllWindows()
    
    ##########################################
    # set camera configuration
    ##########################################
    def set_camera_config(self, Fps=False, Res=False, Focus=False):
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
                    # self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    # self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
        """ with this function you can load json file """
        try:
            # try to load the json file if exist
            with open("./src/CameraConfig.json") as config_file:
                self.camera_config = json.load(config_file)

        # Catch Err in this case might be naming diff in json file and print defined
        except Exception as e:
            print(f"ERROR: Unable To Load Json File {e}")
            self.camera_config = None

