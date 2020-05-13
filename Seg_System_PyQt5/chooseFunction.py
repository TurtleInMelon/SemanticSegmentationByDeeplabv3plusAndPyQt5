# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'chooseFunction.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class ChooseFunction_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(600, 400)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(240, 90, 80, 40))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(240, 160, 100, 40))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(240, 220, 100, 40))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(240, 280, 100, 40))
        self.pushButton_3.setObjectName("pushButton_3")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "选择系统"))
        self.label.setText(_translate("Form", "选择功能:"))
        self.pushButton.setText(_translate("Form", "语义分割"))
        self.pushButton_2.setText(_translate("Form", "视频分割"))
        self.pushButton_3.setText(_translate("Form", "返回上一级"))
