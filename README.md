# server_robot_vision

This application is developed as part of master project by Siamak Mirifar.

1. This Application is Fully functional on Linux distribution based on Debian

2. To install all required python libraries run following script in command prompt:

`pip install -r requirements.txt`

## Supported cameras

For this project the BRIO ULTRA HD PRO BUSINESS WEBCAM is mounted.
To have full  specification of camera check following link:

[Logitech BRIO](https://www.logitech.com/content/dam/logitech/vc/en_hk/pdf/Brio-Datasheet.pdf)

The experience shows that using USB.2 cable could led to low frame per second (fps) in video recording. It is suggested to use USB.3 or above.

Other cameras can be use for this project. However, the resolution of the camera should be minimum 1280*720.

## Compilation

To be able to connect to the server the subgroups project should be clone to the project folder with the same name. The clone can be done by running following script (cloning project) in the project folder.

`git clone https://inf-git.fh-rosenheim.de/ing/labore/rechneranwendungen/robosoccer/software/server_robot_client.git`

## Running

### Python version:
To run the application using python interpreter run main.py. The Python version of the application only works if all libraries are installed.

### Executable version:

To run the application independent of the python libraries, Main file should be run. This executable file completely independent of python libraries and it is tested on Debian based linux.

#### Robot detection on image:
To test application it is possible to upload image or enter image path and start the detection process on the image.

#### Robot detection on video:
The principle of detecting robots in the video section works the same as detecting robots in the image section. The difference is for detecting robots using images there is no need that the camera to be connected. But for the video section, the camera should be connected and recognized by the system.
