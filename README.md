# server_robot_vision

This application is developed as part of master project by Siamak Mirifar.

1. This Application is Fully functional on Linux distribution based on Debian

2. To install all required python libraries run following script in command prompt:

pip install -r requirements.txt

### Supported cameras

For this project the BRIO ULTRA HD PRO BUSINESS WEBCAM is mounted.
To have full  specification of camera check following link:

[Logitech BRIO](https://www.logitech.com/content/dam/logitech/vc/en_hk/pdf/Brio-Datasheet.pdf)

The experience shows that using USB.2 cable could led to low frame per second (fps) in video recording. It is suggested to use USB.3 or above.

Other cameras can be use for this project. However, the resolution of the camera should be minimum 1280*720.

### Compilation

To be able to connect to the server the subgroups project should be clone to the project folder with the same name. The clone can be done by running following script (cloning project) in the project folder.

git clone https://inf-git.fh-rosenheim.de/ing/labore/rechneranwendungen/robosoccer/software/server_robot_client.git