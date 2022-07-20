# server_robot_vision
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#About The Project">About The Project</a></li>
    <li><a href="#Getting started">Getting started</a></li>
    <li><a href="#Supported cameras">Supported cameras</a></li>
    <li><a href="#Compilation">Compilation</a></li>
    <li><a href="#Running">Running</a></li>
    <li><a href="#Application graphical user interface">Application graphical user interface</a></li>
    <li><a href="#Application structure">Application structure</a></li>
    <li><a href="#Future work">Future work</a></li>
    <li><a href="#Main Libraries and references">Main Libraries and references</a></li>
  </ol>
</details>

## About The Project

This project is part of a master's project at Rosenheim University of Applied Sciences under the supervision of Professor Dietrich. The purpose of this project is to determine the position and Id of the small robot on the soccer field. After the position and IDs have been found, the data will be sent to the server. The whole project's purpose is to make it possible for different teams to play soccer with the available robots. 

## Getting started

This Application is Fully functional on Linux distribution based on Debian.

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

## Application graphical user interface

![alt text](/Images/ApplicationInterface.png?raw=true)

## Application structure

[Wiki](https://inf-git.fh-rosenheim.de/ing/labore/rechneranwendungen/robosoccer/software/server_robot_vision/-/wikis/Application-Wiki)

## Future work

The current application is using the 1080p frame for detecting robot. The reason is that for finding the angle of the robot we need a fix line in the image that could give us this chance to calculate the angle.

The only possible way for this problem was finding the minimum length between the circle on the top of robots. As the robots are small in the field finding smallest length is so dependent on the image pixel. Low resolution image will cause uncertain measurement.

What is suggested is base on the final design of the robots, the front design of the robots should be straight. It means that the robots are not complete circle. With the help of edge detection and contours in OpenCv it is possible to detect this line and calculate the angle from this line.

The advantage of this method is that it reduced the computation, and code. It is make it possible to work with images with lower resolution, which means higher performance.


## Main Libraries and references

 - [RoboCup-SSL / ssl-vision](https://github.com/RoboCup-SSL/ssl-vision)
 - [OpenCv](https://opencv.org)
 - [Qt for Python](https://www.qt.io/qt-for-python)
