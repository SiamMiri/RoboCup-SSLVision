import json
import cv2


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

    return data


def start_capturing_command(self):
    cam = cv2.VideoCapture(0)

    # Change FPS
    if isinstance(self.cameraConfig["FPS"], int):
        cam.set(cv2.CAP_PROP_FPS, self.cameraConfig["FPS"])  # set camera FPS from Json file
    elif isinstance(self.cameraConfig["FPS"], float):
        cam.set(cv2.CAP_PROP_FPS, int(self.cameraConfig["FPS"]))  # set camera FPS from Json file
    else:
        print(".json file config FPS is not correct")

    # Change Resolution
    if isinstance(self.cameraConfig["Resolution"], list):
        # set camera Resolution from Json file
        cam.set(3, int(self.cameraConfig["Resolution"][0]))
        cam.set(4, int(self.cameraConfig["Resolution"][1]))
    else:
        print(".json file config Resolution is not correct")

    # Change Focus
    if isinstance(self.cameraConfig["focus"], int):
        # set camera Resolution from Json file
        cam.set(cv2.CAP_PROP_FOCUS, self.cameraConfig["focus"])
    elif isinstance(self.cameraConfig["focus"], float):
        # set camera Resolution from Json file
        cam.set(cv2.CAP_PROP_FOCUS, int(self.cameraConfig["focus"]))
    else:
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    cv2.namedWindow("RobotSoccer\tHit Escape to Exit")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("RobotSoccer\tHit Escape to Exit", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break


def finish_capturing_command():
    cv2.destroyAllWindows()
