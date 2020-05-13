import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget
from PyQt5 import QtCore
from VideoSegmentation import *

class MyMainWindow(QMainWindow, VideoSegmentation_Form):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())