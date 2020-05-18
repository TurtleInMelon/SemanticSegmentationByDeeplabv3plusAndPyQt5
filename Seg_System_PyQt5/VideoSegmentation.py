# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PictureSegmentation.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget
from PyQt5.Qt import QUrl, QVideoWidget
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
import os

from myVideoWidget import myVideoWidget


class VideoSegmentation_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(830, 550)
        self.comboBox = QtWidgets.QComboBox(Form)
        self.comboBox.setGeometry(QtCore.QRect(180, 30, 111, 21))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(130, 30, 41, 16))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(120, 0, 56, 21))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.open_file)
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(180, 0, 241, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(570, 30, 56, 21))
        self.pushButton_2.setObjectName("pushButton")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(570, 0, 56, 21))
        self.pushButton_3.setObjectName("pushButton")


        self.btn_play = QtWidgets.QPushButton(Form)
        self.btn_play.setGeometry(QtCore.QRect(290, 430, 50, 50))
        self.btn_play.setObjectName("btn_play")
        self.btn_stop = QtWidgets.QPushButton(Form)
        self.btn_stop.setGeometry(QtCore.QRect(400, 430, 50, 50))
        self.btn_stop.setObjectName("btn_stop")

        self.wgt_video = myVideoWidget(Form)
        self.wgt_video.setGeometry(QtCore.QRect(170, 70, 450, 300))
        self.wgt_video.setObjectName("wgt_video")

        self.btn_open = QtWidgets.QPushButton(Form)
        self.btn_open.setGeometry(QtCore.QRect(180, 430, 50, 50))
        self.btn_open.setObjectName("btn_open")

        # self.label_4 = QtWidgets.QLabel(Form)
        # self.label_4.setText("显示视频")
        # self.label_4.setGeometry(QtCore.QRect(150, 70, 450, 300))
        # # self.label_4.move(310, 380)
        # self.label_4.setStyleSheet("QLabel{background:white;}"
        #                            "QLabel{color:rgb(300,300,120);font-size:10px;font-weight:bold;font-family:幼圆;}")
        # self.label_4.setAttribute(Qt.Qt.WA_TranslucentBackground)

        self.sld_video = QtWidgets.QSlider(Form)
        self.sld_video.setGeometry(QtCore.QRect(200, 380, 251, 20))
        self.sld_video.setMaximum(100)
        self.sld_video.setOrientation(QtCore.Qt.Horizontal)
        self.sld_video.setObjectName("sld_video")

        self.lab_video = QtWidgets.QLabel(Form)
        self.lab_video.setGeometry(QtCore.QRect(510, 380, 91, 20))
        self.lab_video.setObjectName("lab_video")

        self.statusbar = QtWidgets.QStatusBar(Form)
        self.statusbar.setObjectName("statusbar")
        # Form.setStatusBar(self.statusbar)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "视频分割"))
        self.comboBox.setItemText(0, _translate("Form", "Xception65"))
        self.comboBox.setItemText(1, _translate("Form", "改进模型"))
        self.comboBox.setItemText(2, _translate("Form", "MobileNetV2"))
        self.comboBox.setItemText(3, _translate("Form", "ResNet101"))
        self.label.setText(_translate("Form", "选择模型"))
        self.pushButton.setText(_translate("Form", "选择视频"))
        self.pushButton_2.setText(_translate("Form", "视频分割"))
        self.pushButton_3.setText(_translate("Form", "返回上一级"))
        self.lab_video.setText(_translate("Form", "0%"))
        # self.btn_play.setText(_translate("Form", "播放"))
        # self.btn_stop.setText(_translate("Form", "暂停"))
        # self.btn_open.setText(_translate("Form", "打开视频文件"))

    def open_file(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, "打开文件", os.getcwd(), "Video files (*.mp4 *.avi)")
        self.lineEdit.setText(fileName)



