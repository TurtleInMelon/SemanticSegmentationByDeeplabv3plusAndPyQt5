# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PictureSegmentation.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot

class PictureSegmentation_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1080, 720)
        self.comboBox = QtWidgets.QComboBox(Form)
        self.comboBox.setGeometry(QtCore.QRect(60, 30, 111, 21))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(10, 30, 41, 16))
        self.label.setObjectName("label")

        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(0, 0, 56, 21))
        self.pushButton.setObjectName("pushButton")


        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(480, 0, 56, 21))
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(480, 30, 56, 21))

        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(60, 0, 241, 20))
        self.lineEdit.setObjectName("lineEdit")

        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setText("显示原图片")
        self.label_2.setFixedSize(500, 250)
        self.label_2.move(0, 70)
        self.label_2.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:幼圆;}")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setText("显示分割图片")
        self.label_3.setFixedSize(500, 250)
        self.label_3.move(520, 70)
        self.label_3.setStyleSheet("QLabel{background:white;}"
                                   "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:幼圆;}")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setText("显示融合图片")
        self.label_4.setFixedSize(500, 250)
        self.label_4.move(310, 380)
        self.label_4.setStyleSheet("QLabel{background:white;}"
                                   "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:幼圆;}")

        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(480, 350, 56, 21))

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "图片分割"))
        self.comboBox.setItemText(0, _translate("Form", "Xception65"))
        self.comboBox.setItemText(1, _translate("Form", "改进模型"))
        self.comboBox.setItemText(2, _translate("Form", "MobileNetV2"))
        self.comboBox.setItemText(3, _translate("Form", "ResNet101"))
        self.label.setText(_translate("Form", "选择模型"))
        self.pushButton.setText(_translate("Form", "选择图片"))
        self.pushButton_2.setText(_translate("Form", "返回上一级"))
        self.pushButton_3.setText(_translate("Form", "语义分割"))
        self.pushButton_4.setText(_translate("Form", "图像融合"))



