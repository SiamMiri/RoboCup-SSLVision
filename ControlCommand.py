import json
import cv2
from ImageProcessing import imgProc

# load json file as dictionary in python
def load_json_config_file():
    """ with this function you can load json file """
    try:
        # try to load the json file if exist
        with open("CameraConfig.json") as config_file:
            data = json.load(config_file)

    # Catch Err in this case might be naming diff in json file and print defined
    except Exception as e:
        print(e)
        data = None

    print(data["CameraConfig"])
    print(data["FrameConfig"])
    return data


# Function: Calling Class ImageProcessing
# slot: start_capturing
# Return: self.robot_id, self.robot_location, self.robot_orientation 
# Date of issue: 05.03.2021
# Date of modify: XX.XX.xXXx
def start_capturing_command(self):
    """ Calls Class ImageProcessing object """
    imgProc.start_capturing(self.cameraConfig)

'''
    self.robot_id = imgProc.detect_robot_id()
    self.robot_location = imgProc.detect_robot_location()
    self.robot_orientation = imgProc.detect_robot_orientation()
'''

def finish_capturing_command(self):
    imgProc.finish_capturing()
