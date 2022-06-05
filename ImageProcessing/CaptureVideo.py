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
        self.load_json_config_file()

        """ Start capturing get camera config from start_process_command """
        self.set_camera_config(self.camera_config['CameraConfig'], Fps=False, Res=False, Focus=False)

    def __del__(self) -> None:
        self.finish_capturing()
        print("Camera Released")

    ##########################################
    # start image capturing
    ##########################################
    def start_video_capturing(self):

        # Initiate Windows Name
        cv2.namedWindow("RobotSoccer\tHit Escape or Q to Exit")
        
        image_processing = Image_Processing()

        # while True:
            # Calculate FPS
            # cTime = time.time()

        ret, frame = self.camera_capture.read() # FIXME: Changed to load Image
        if not ret:
            print("failed to grab frame")
            return None
        
        return frame
            # frame, _ = image_processing.start_process(frame= frame)
        
            # fps = 1 / (cTime - self.pTime)
            # self.pTime = cTime

            # # Set FPS on the frame
            # cv2.putText(frame, str(abs(int(fps))), (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # show image
            # cv2.imshow("RobotSoccer\tHit Escape or Q to Exit", frame)

    ##########################################
    # destroy opencv camera and free the camera 
    ##########################################
    def finish_capturing(self):
        cv2.destroyAllWindows()
    
    ##########################################
    # set camera configuration
    ##########################################
    def set_camera_config(self, camera_config: json, Fps=False, Res=False, Focus=False):
        try:
            if isinstance(Fps, bool) and isinstance(Res, bool) and isinstance(Focus, bool):
                
                if Fps is not False:
                    # Change FPS
                    if isinstance(camera_config["FPS"], int):
                        self.camera_capture.set(cv2.CAP_PROP_FPS, camera_config["FPS"])  # set camera FPS from Json file
                        print("FPS Set")
                    elif isinstance(camera_config["FPS"], float):
                        self.camera_capture.set(cv2.CAP_PROP_FPS, int(camera_config["FPS"]))  # set camera FPS from Json file
                        print("FPS Set")
                    else:
                        print("FPS Configuration is incorrect")
                else:
                    print("FPS is not set")

                if Res is not False:
                    # self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    # self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    # Change Resolution
                    if isinstance(camera_config["Resolution"], list):
                        # set camera Resolution from Json file
                        self.camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config["Resolution"][0]))
                        self.camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config["Resolution"][1]))
                        print("Resolution Set")
                    else:
                        print("Resolution Configuration is incorrect")
                else:
                    print("Resolution is not Set")

                if Focus is not False:
                    # Change Focus
                    if isinstance(camera_config["focus"], int):
                        # set camera Resolution from Json file
                        self.camera_capture.set(cv2.CAP_PROP_FOCUS, camera_config["focus"])
                        print("focus Set")
                    elif isinstance(camera_config["focus"], float):
                        # set camera Resolution from Json file
                        # self.camera_capture.set(cv2.CAP_PROP_FOCUS, int(camera_config["focus"])) // This may not work
                        self.camera_capture.set(28, int(camera_config["focus"]))
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

