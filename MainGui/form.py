# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(544, 352)
        MainWindow.setMaximumSize(QtCore.QSize(550, 352))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lbl_University = QtWidgets.QLabel(self.centralwidget)
        self.lbl_University.setGeometry(QtCore.QRect(350, 0, 171, 61))
        self.lbl_University.setObjectName("lbl_University")
        self.lbl_Date = QtWidgets.QLabel(self.centralwidget)
        self.lbl_Date.setGeometry(QtCore.QRect(20, 0, 201, 51))
        self.lbl_Date.setObjectName("lbl_Date")
        self.btn_LoadImageFile = QtWidgets.QPushButton(self.centralwidget)
        self.btn_LoadImageFile.setGeometry(QtCore.QRect(20, 90, 111, 31))
        self.btn_LoadImageFile.setObjectName("btn_LoadImageFile")
        self.btn_StartImageCapturing = QtWidgets.QPushButton(self.centralwidget)
        self.btn_StartImageCapturing.setGeometry(QtCore.QRect(20, 170, 171, 31))
        self.btn_StartImageCapturing.setObjectName("btn_StartImageCapturing")
        self.btn_StartVideoCapturing = QtWidgets.QPushButton(self.centralwidget)
        self.btn_StartVideoCapturing.setGeometry(QtCore.QRect(350, 170, 171, 31))
        self.btn_StartVideoCapturing.setObjectName("btn_StartVideoCapturing")
        self.btn_ImageColorConfiguration = QtWidgets.QPushButton(self.centralwidget)
        self.btn_ImageColorConfiguration.setGeometry(QtCore.QRect(20, 210, 171, 31))
        self.btn_ImageColorConfiguration.setObjectName("btn_ImageColorConfiguration")
        self.txt_FilePath = QtWidgets.QTextEdit(self.centralwidget)
        self.txt_FilePath.setGeometry(QtCore.QRect(150, 90, 371, 31))
        self.txt_FilePath.setObjectName("txt_FilePath")
        self.lbl_version = QtWidgets.QLabel(self.centralwidget)
        self.lbl_version.setGeometry(QtCore.QRect(220, 270, 101, 21))
        self.lbl_version.setObjectName("lbl_version")
        self.btn_VideoColorConfiguration = QtWidgets.QPushButton(self.centralwidget)
        self.btn_VideoColorConfiguration.setGeometry(QtCore.QRect(350, 210, 171, 31))
        self.btn_VideoColorConfiguration.setObjectName("btn_VideoColorConfiguration")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 544, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lbl_University.setText(_translate("MainWindow", "TextLabel"))
        self.lbl_Date.setText(_translate("MainWindow", "TextLabel"))
        self.btn_LoadImageFile.setText(_translate("MainWindow", "Load Image"))
        self.btn_StartImageCapturing.setText(_translate("MainWindow", "Start Image Capturing"))
        self.btn_StartVideoCapturing.setText(_translate("MainWindow", "Start Video Capturing"))
        self.btn_ImageColorConfiguration.setText(_translate("MainWindow", "Configure Color Image"))
        self.lbl_version.setText(_translate("MainWindow", "version: 1.1.1"))
        self.btn_VideoColorConfiguration.setText(_translate("MainWindow", "Configure Color Video"))