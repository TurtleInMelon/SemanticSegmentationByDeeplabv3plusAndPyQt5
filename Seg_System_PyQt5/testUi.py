# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testUi.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(544, 385)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 50, 401, 211))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 544, 18))
        self.menubar.setObjectName("menubar")
        self.menu_F = QtWidgets.QMenu(self.menubar)
        self.menu_F.setObjectName("menu_F")
        self.menu_E = QtWidgets.QMenu(self.menubar)
        self.menu_E.setObjectName("menu_E")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.fileOpenAction = QtWidgets.QAction(MainWindow)
        self.fileOpenAction.setObjectName("fileOpenAction")
        self.fileNewAction = QtWidgets.QAction(MainWindow)
        self.fileNewAction.setObjectName("fileNewAction")
        self.fileCloseAction = QtWidgets.QAction(MainWindow)
        self.fileCloseAction.setObjectName("fileCloseAction")
        self.addWinAction = QtWidgets.QAction(MainWindow)
        self.addWinAction.setObjectName("addWinAction")
        self.menubar.addAction(self.menu_F.menuAction())
        self.menubar.addAction(self.menu_E.menuAction())
        self.toolBar.addAction(self.fileOpenAction)
        self.toolBar.addAction(self.fileNewAction)
        self.toolBar.addAction(self.fileCloseAction)
        self.toolBar.addAction(self.addWinAction)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menu_F.setTitle(_translate("MainWindow", "文件(F)"))
        self.menu_E.setTitle(_translate("MainWindow", "编辑(E)"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.fileOpenAction.setText(_translate("MainWindow", "打开"))
        self.fileOpenAction.setToolTip(_translate("MainWindow", "打开"))
        self.fileOpenAction.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.fileNewAction.setText(_translate("MainWindow", "新建"))
        self.fileNewAction.setToolTip(_translate("MainWindow", "新建"))
        self.fileNewAction.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.fileCloseAction.setText(_translate("MainWindow", "关闭"))
        self.fileCloseAction.setToolTip(_translate("MainWindow", "关闭"))
        self.fileCloseAction.setShortcut(_translate("MainWindow", "Ctrl+C"))
        self.addWinAction.setText(_translate("MainWindow", "添加窗体"))
        self.addWinAction.setToolTip(_translate("MainWindow", "添加窗体"))
