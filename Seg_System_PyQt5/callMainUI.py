'''
主系统调用入口
'''
import sys
import os
from PyQt5 import QtCore, QtGui
from PyQt5.Qt import *
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtMultimedia import *
from MainUI import Ui_MainWindow
from chooseFunction import ChooseFunction_Form
from PictureSegmentation import PictureSegmentation_Form
from VideoSegmentation import VideoSegmentation_Form
from myVideoWidget import myVideoWidget
from PIL.ImageQt import ImageQt
from DeeplabForPicture import *
from DeeplabForVideo import get_mixed_video
import time

# 训练并冻结好的模型文件路径
xception_pd_file_path = r"D:\pb文件\xception\frozen_inference_graph.pb"
mobilenetv2_pd_file_path = r"D:\pb文件\mobileNetV2\frozen_inference_graph.pb"
resnet101_pd_file_path = r"D:\pb文件\resNet101\frozen_inference_graph.pb"
modifymodel_pd_file_path = r"D:\pb文件\modify_model\frozen_inference_graph.pb"

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__()

        self.setupUi(self)
        self.label.setStyleSheet('''QLabel{color:white;font-size:40px;font-family:幼圆;}
                                }''')
        self.pushButton.setIcon(QIcon("./image/enter.ico"))
        self.pushButton.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:red}"
                                      "QPushButton{background-color:lightgreen}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}")
        self.pushButton_2.setIcon(QIcon("./image/close.ico"))
        self.pushButton_2.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:red}"
                                      "QPushButton{background-color:lightblue}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}")

class MyChooseFunction_Window(QWidget, ChooseFunction_Form):
    def __init__(self):
        super(MyChooseFunction_Window, self).__init__()
        self.setupUi(self)
        self.label.setStyleSheet('''QLabel{color:lightblue;font-size:20px;font-family:幼圆;}
                                        }''')
        self.pushButton.setIcon(QIcon("./image/enter.ico"))
        self.pushButton.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:red}"
                                      "QPushButton{background-color:lightgreen}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}")
        self.pushButton_2.setIcon(QIcon("./image/enter.ico"))
        self.pushButton_2.setStyleSheet("QPushButton{color:black}"
                                        "QPushButton:hover{color:red}"
                                        "QPushButton{background-color:lightgreen}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:10px}"
                                        "QPushButton{padding:2px 4px}")
        self.pushButton_3.setIcon(QIcon("./image/close.ico"))
        self.pushButton_3.setStyleSheet("QPushButton{color:black}"
                                        "QPushButton:hover{color:red}"
                                        "QPushButton{background-color:lightblue}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:10px}"
                                        "QPushButton{padding:2px 4px}")

class MyPictureSegmentation_Window(QWidget, PictureSegmentation_Form):
    def __init__(self):
        super(MyPictureSegmentation_Window, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.open_picture)
        self.pushButton_3.clicked.connect(self.segmantetion)
        self.pushButton_4.clicked.connect(self.mixed_picture)
        self.label.setStyleSheet('''QLabel{color:black;font-size:10px;font-family:幼圆;}
                                                }''')
        # self.pushButton.setIcon(QIcon("./image/enter.ico"))
        self.pushButton.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:blue}"
                                      "QPushButton{background-color:lightgreen}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}")
        # self.pushButton_2.setIcon(QIcon("./image/enter.ico"))
        self.pushButton_2.setStyleSheet("QPushButton{color:black}"
                                        "QPushButton:hover{color:red}"
                                        "QPushButton{background-color:lightgreen}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:10px}"
                                        "QPushButton{padding:2px 4px}")
        # self.pushButton_3.setIcon(QIcon("./image/close.ico"))
        self.pushButton_3.setStyleSheet("QPushButton{color:black}"
                                        "QPushButton:hover{color:red}"
                                        "QPushButton{background-color:lightblue}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:10px}"
                                        "QPushButton{padding:2px 4px}")
        self.pushButton_4.setStyleSheet("QPushButton{color:black}"
                                        "QPushButton:hover{color:red}"
                                        "QPushButton{background-color:lightyellow}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:10px}"
                                        "QPushButton{padding:2px 4px}")
        self.lineEdit.setClearButtonEnabled(True)
        self.lineEdit.setStyleSheet("QLineEdit{border-radius:5px}"
                                    "QLineEdit{color:black}")
        self.comboBox.setStyleSheet("QComboBox{border-radius:5px}")


    def open_picture(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '打开图片', os.getcwd(), "Image files (*.jpg *.png)")
        jpg = QtGui.QPixmap(fileName).scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(jpg)
        self.lineEdit.setText(fileName)

    def segmantetion(self):
        progress = QProgressDialog(self)
        progress.setWindowTitle("语义分割")
        progress.setMinimumDuration(5)
        progress.setLabelText("正在操作...")
        progress.setCancelButtonText("取消")
        progress.setWindowModality(Qt.WindowModal)
        progress.setRange(0, 1000)
        for i in range(500):
            progress.setValue(i)
            if progress.wasCanceled():
                QMessageBox.warning(self, "提示", "操作失败")
                break
            time.sleep(0.01)
            # else:
            #     progress.setValue(i)
            #     QMessageBox.information(self, "提示", "操作成功")
        model_name = self.comboBox.currentText()
        pd_file_path = choose_model(model_name)
        progress.setValue(500)
        progress.setLabelText("模型加载完成，正在进行图像语义分割...")
        image_path = self.lineEdit.text()
        seg_image, mixed_image = get_mixed_picture(pd_file_path, image_path)
        self.mixed_image = mixed_image
        for i in range(501, 1000):
            progress.setValue(i)
            if progress.wasCanceled():
                QMessageBox.warning(self, "提示", "操作失败")
                break
            time.sleep(0.01)
        # mixed_image.show()
        qim = ImageQt(seg_image)
        seg_image = QtGui.QImage(qim)
        jpg = QtGui.QPixmap.fromImage(seg_image).scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(jpg)
        progress.setValue(1000)
        QMessageBox.information(self, "提示", "分割成功!!")
    def mixed_picture(self):
        qim = ImageQt(self.mixed_image)
        mixed_image = QtGui.QImage(qim)
        jpg = QtGui.QPixmap.fromImage(mixed_image).scaled(self.label_4.width(), self.label_4.height())
        self.label_4.setPixmap(jpg)
        QMessageBox.information(self, "提示", "融合成功!!")






class MyVideoSegmentation_Window(QWidget, VideoSegmentation_Form):
    def __init__(self):
        super(MyVideoSegmentation_Window, self).__init__()
        self.setupUi(self)
        self.videoFullScreen = False  # 判断当前widget是否全屏
        self.videoFullScreenWidget = myVideoWidget()  # 创建一个全屏的widget
        self.videoFullScreenWidget.setFullScreen(1)
        self.videoFullScreenWidget.hide()  # 不用的时候隐藏起来
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.wgt_video)
        self.btn_open.clicked.connect(self.openVideoFile)  # 打开视频文件按钮
        self.btn_play.clicked.connect(self.playVideo)
        self.btn_stop.clicked.connect(self.pauseVideo)
        self.player.positionChanged.connect(self.changeSlide)  # change Slide
        self.videoFullScreenWidget.doubleClickedItem.connect(self.videoDoubleClicked)  # 双击响应
        self.wgt_video.doubleClickedItem.connect(self.videoDoubleClicked)  # 双击响应
        self.pushButton_2.clicked.connect(self.segmentation)
        # self.pushButton_3.clicked.connect(self.)

        self.label.setStyleSheet('''QLabel{color:black;font-size:10px;font-family:幼圆;}
                                                        }''')
        # self.pushButton.setIcon(QIcon("./image/enter.ico"))
        self.pushButton.setStyleSheet("QPushButton{color:black}"
                                      "QPushButton:hover{color:blue}"
                                      "QPushButton{background-color:lightgreen}"
                                      "QPushButton{border:2px}"
                                      "QPushButton{border-radius:10px}"
                                      "QPushButton{padding:2px 4px}")
        # self.pushButton_2.setIcon(QIcon("./image/enter.ico"))
        self.pushButton_2.setStyleSheet("QPushButton{color:black}"
                                        "QPushButton:hover{color:red}"
                                        "QPushButton{background-color:blue}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:10px}"
                                        "QPushButton{padding:2px 4px}")
        self.pushButton_3.setStyleSheet("QPushButton{color:black}"
                                        "QPushButton:hover{color:red}"
                                        "QPushButton{background-color:lightgreen}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:10px}"
                                        "QPushButton{padding:2px 4px}")
        # self.pushButton_3.setIcon(QIcon("./image/close.ico"))
        self.btn_open.setStyleSheet("QPushButton{color:black}"
                                    "QPushButton{border-image: url(./image/open.png)}"
                                    "QPushButton:hover{color:red}"
                                    "QPushButton{background-color:rgb(255,255,255)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:10px}"
                                    "QPushButton{padding:2px 4px}")
        self.btn_play.setStyleSheet("QPushButton{color:black}"
                                    "QPushButton{border-image: url(./image/play.png)}"
                                    "QPushButton:hover{color:red}"
                                    "QPushButton{background-color:rgb(255,255,255)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:10px}"
                                    "QPushButton{padding:2px 4px}")
        self.btn_stop.setStyleSheet("QPushButton{color:black}"
                                    "QPushButton{border-image: url(./image/pause.png)}"
                                    "QPushButton:hover{color:red}"
                                    "QPushButton{background-color:rgb(255,255,255)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:10px}"
                                    "QPushButton{padding:2px 4px}")
        self.lineEdit.setClearButtonEnabled(True)
        self.lineEdit.setStyleSheet("QLineEdit{border-radius:5px}"
                                    "QLineEdit{color:black}")
        self.comboBox.setStyleSheet("QComboBox{border-radius:5px}")

    def playVideo(self):
        self.player.play()

    def pauseVideo(self):
        self.player.pause()

    def changeSlide(self, position):
        self.vidoeLength = self.player.duration() + 0.1
        self.sld_video.setValue(round((position / self.vidoeLength) * 100))
        self.lab_video.setText(str(round((position / self.vidoeLength) * 100, 2)) + '%')

    def videoDoubleClicked(self, text):
        if self.player.duration() > 0:  # 开始播放后才允许进行全屏操作
            if self.videoFullScreen:
                self.player.pause()
                self.videoFullScreenWidget.hide()
                self.player.setVideoOutput(self.wgt_video)
                self.player.play()
                self.videoFullScreen = False
            else:
                self.player.pause()
                self.videoFullScreenWidget.show()
                self.player.setVideoOutput(self.videoFullScreenWidget)
                self.player.play()
                self.videoFullScreen = True

    def segmentation(self):
        print(QUrl.fromLocalFile(self.lineEdit.text()))
        model_name = self.comboBox.currentText()
        pd_file_path = choose_model(model_name)
        _, out_put_path = get_mixed_video(pd_file_path, self.lineEdit.text())
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(out_put_path)))
        self.player.play()

    def openVideoFile(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, "打开文件", os.getcwd(), "Video files (*.mp4 *.avi)")
        self.lineEdit.setText(fileName)
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
        self.player.play()



def enterWindow(curWindow, nextWindow):
    '''

    :param curWindow: 当前界面
    :param nextWindow: 要进入的界面
    :return:
    '''
    curWindow.close()
    nextWindow.show()

def closeWindow(curWindow, lastWindow= None):
    '''
        :param curWindow: 当前窗口
        :param lastWindow: 上一级窗口
        :return:
    '''
    if lastWindow == None:      # 说明当前窗口为主窗口，直接关闭
        curWindow.close()
    else:               # 关闭当前窗口，打开上一级窗口
        curWindow.close()
        lastWindow.show()

def choose_model(model_name):
    if model_name == "Xception65":
        pd_file_path = xception_pd_file_path
    elif model_name == "改进模型":
        pd_file_path = modifymodel_pd_file_path
    elif model_name == "MobileNetV2":
        pd_file_path = mobilenetv2_pd_file_path
    else:
        pd_file_path = resnet101_pd_file_path
    return pd_file_path

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./image/cartoon3.ico"))
    myWin = MyMainWindow()  # 主窗口
    myChooseFunctionWindow = MyChooseFunction_Window()  # 选择功能窗口
    myPictureSegmentationWindow = MyPictureSegmentation_Window()    # 图片分割窗口
    myVideoSegmentetionWindow = MyVideoSegmentation_Window()    # 视频分割窗口
    myWin.pushButton.clicked.connect(lambda: enterWindow(myWin, myChooseFunctionWindow))
    myWin.pushButton_2.clicked.connect(lambda: closeWindow(myWin))

    myChooseFunctionWindow.pushButton.clicked.connect(
        lambda: enterWindow(myChooseFunctionWindow, myPictureSegmentationWindow))
    myChooseFunctionWindow.pushButton_2.clicked.connect(
        lambda: enterWindow(myChooseFunctionWindow, myVideoSegmentetionWindow))
    myChooseFunctionWindow.pushButton_3.clicked.connect(lambda: closeWindow(myChooseFunctionWindow, myWin))

    myPictureSegmentationWindow.pushButton_2.clicked.connect(
        lambda: closeWindow(myPictureSegmentationWindow, myChooseFunctionWindow))
    myVideoSegmentetionWindow.pushButton_3.clicked.connect(
        lambda: closeWindow(myVideoSegmentetionWindow, myChooseFunctionWindow))
    myWin.show()


    # print(myPictureSegmentationWindow.comboBox.currentText())
    # print(QDesktopWidget().screenGeometry())
    sys.exit(app.exec_())