import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog,QWidget
from testUi import *
from childrenForm import *

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)

        self.child = ChildrenForm()
        self.fileCloseAction.triggered.connect(self.close)
        self.fileOpenAction.triggered.connect(self.openMsg)
        self.addWinAction.triggered.connect(self.childShow)

    def openMsg(self):
        file, ok = QFileDialog.getOpenFileName(self, "打开", "C:/", "All Files (*);;Text Files (*.txt)")
        self.statusbar.showMessage(file)

    def childShow(self):
        self.gridLayout.addWidget(self.child)
        self.child.show()

class ChildrenForm(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(ChildrenForm, self).__init__()
        self.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())