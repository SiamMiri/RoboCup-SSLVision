# server_robot_vision

## Getting started

This application is developed as part of master project.

This Application is Fully functional on Linux distribution based on Debian.

Following dependencies are required to build the software:

 * QT >= 4.3 with opengl and networking support
 * Google protocol buffers (protoc)
 * OpenCV >= 3
 * video for linux 2 (v4l)

To install all required python libraries run following script in command prompt:

`pip install -r requirements.txt`

## Supported cameras

For this project the BRIO ULTRA HD PRO BUSINESS WEBCAM is mounted.
To have full  specification of camera check following link:

[Logitech BRIO](https://www.logitech.com/content/dam/logitech/vc/en_hk/pdf/Brio-Datasheet.pdf)

The experience shows that using **USB.2** cable could led to low frame per second (fps) in video recording. It is suggested to use **USB.3** or above.

Other cameras can be use for this project. However, the resolution of the camera should be minimum **1280*720**.

## Compilation

To be able to connect to the server the subgroups project should be clone to the project folder with the same name. The clone can be done by running following script (cloning project) in the project folder.

`git clone https://inf-git.fh-rosenheim.de/ing/labore/rechneranwendungen/robosoccer/software/server_robot_client.git`

## Running

#### Options

<details open>
<summary>1: Python version:</summary>
<br>
To run the application using python interpreter run main.py. The Python version of the application only works if all libraries are installed.
</details>

<details open>
<summary>2: Executable version:</summary>
<br>
To run the application independent of the python libraries, Main file should be run. This executable file completely independent of python libraries and it is tested on Debian based linux.
</details>

#### Logging

To check errors and time lapse all the processing classes have attribute **PRINT_DEBUG**, which should be set to True to active logging.


## Main Libraries and resources

 - [RoboCup-SSL / ssl-vision](https://github.com/RoboCup-SSL/ssl-vision)
 - [OpenCv](https://opencv.org)
 - [Qt for Python](https://www.qt.io/qt-for-python)

## Application graphical user interface

![alt text](/Images/ApplicationInterface.png?raw=true)

## Application structure

[Wiki](https://inf-git.fh-rosenheim.de/ing/labore/rechneranwendungen/robosoccer/software/server_robot_vision/-/wikis/Application-Wiki)

## Future work

The current application is using the 1080p frame for detecting robot. The reason is that for finding the angle of the robot we need a fix line in the image that could give us this chance to calculate the angle.

The only possible way for this problem was finding the minimum length between the circle on the top of robots. As the robots are small in the field finding smallest length is so dependent on the image pixel. Low resolution image will cause uncertain measurement.

What is suggested is base on the final design of the robots, the front design of the robots should be straight. It means that the robots are not complete circle. With the help of edge detection and contours in OpenCv it is possible to detect this line and calculate the angle from this line.

The advantage of this method is that it reduced the computation, and code. It is make it possible to work with images with lower resolution, which means higher performance.
