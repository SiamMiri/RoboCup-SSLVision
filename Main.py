from ControlCommand import *


class CaptureImage():
    def __init__(self):
        self.img_counter = 0

        self.cameraConfig = load_json_config_file()  # Loading Camera Configuration from Json File

    def __del__(self):
        finish_capturing_command()

    def start_capturing(self):
        start_capturing_command(self)

    def finish_capturing(self):
        finish_capturing_command(self)


cam = CaptureImage()
cam.start_capturing()
