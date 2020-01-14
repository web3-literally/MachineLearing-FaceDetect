from PyQt5.QtWidgets import QApplication, QCheckBox, QMessageBox, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFileDialog, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
import cv2
from Main.DetectObject import DetectObject as DT
from PIL import Image
import threading
import subprocess
import os
from Main.GenerateTrainData import TrainDataMaker as TM

class MainWnd(QMainWindow):
    def __init__(self, parent=None):
        super(MainWnd, self).__init__(parent)
        self.b_Trained = False
        self.object_detector = DT()

        layout = QVBoxLayout()

        lay_hor_1 = QHBoxLayout()
        self.edit_FilePath = QLineEdit()
        self.btn_OpenFile = QPushButton("...")
        self.btn_OpenFile.clicked.connect(self.on_btnOpenFile_Clicked)
        lay_hor_1.addWidget(QLabel("Please Input FileName!"))
        lay_hor_1.addWidget(self.edit_FilePath)
        lay_hor_1.addWidget(self.btn_OpenFile)
        layout.addLayout(lay_hor_1)

        self.wnd_Picture = QLabel()
        self.wnd_Picture.setScaledContents(True);
        layout.addWidget(self.wnd_Picture)

        lay_hor_2 = QHBoxLayout()
        self.chk_Deep_Check = QCheckBox("DeepCheckEnable")
        self.btn_execVott = QPushButton("Execute Vott!")
        self.btn_genTrainData = QPushButton("Generate TrainData and Set TrainPath!")
        self.btn_Train = QPushButton("Train")
        self.btn_Detect = QPushButton("Detect")
        self.btn_Exit = QPushButton("Exit")

        self.btn_Detect.setEnabled(False)

        self.btn_execVott.clicked.connect(self.execVott)
        self.btn_genTrainData.clicked.connect(self.generateTrainData)
        self.btn_Train.clicked.connect(self.on_Train)
        self.btn_Detect.clicked.connect(self.on_Detect)
        self.btn_Exit.clicked.connect(self.on_Close)

        #lay_hor_2.addWidget(self.btn_execVott)
        lay_hor_2.addWidget(self.chk_Deep_Check)
        lay_hor_2.addWidget(self.btn_genTrainData)
        lay_hor_2.addWidget(self.btn_Train)
        lay_hor_2.addWidget(self.btn_Detect)
        lay_hor_2.addWidget(self.btn_Exit)

        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready")

        layout.addLayout(lay_hor_2)
        centralWnd = QWidget()
        centralWnd.setLayout(layout)
        self.setCentralWidget(centralWnd)
        self.setFixedSize(800, 600)
        self.setWindowTitle("Object Detect!")

    def execVott(self):
        currentDirectory = os.getcwd()
        os.system(currentDirectory + "\\vott\\vott.exe")

    def generateTrainData(self):
        dir_PathTrain = QFileDialog.getExistingDirectory(self, "Select TrainData Directory!")
        if dir_PathTrain:
            trainDataMaker = TM()
            trainDataMaker.generateTrainData(dir_PathTrain)
            #class_dict = CTT.create_class_dict(dir_PathTrain)
            #CTT.create_map_files(dir_PathTrain, class_dict, training_set=True)
            #CTT.create_map_files(dir_PathTrain, class_dict, training_set=False)
        QMessageBox.information(self,"", "TrainData Generated!", QMessageBox.Yes)

    def setImage(self, fileName):
        image = cv2.imread(str(fileName))
        img_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_cv)

        img.save("tmp.jpg")

        self.wnd_Picture.setPixmap(QPixmap("tmp.jpg"))

    def setStatusMessage(self, strMessage):
        self.statusbar.showMessage(strMessage)
    def on_Train(self):
        self.setStatusMessage("Training...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Model Files (*.dat)",
                                                  options=options)
        if fileName:
            self.object_detector.pre_Train(fileName)
        self.setStatusMessage("Train Successed!")
        self.b_Trained = True
        if (self.edit_FilePath.text()):
            self.btn_Detect.setEnabled(True)
        QMessageBox.information(self, "", "Train Finished!", QMessageBox.Yes)
    def on_Detect(self):
        ret_msg = self.object_detector.Detect_Object(self.edit_FilePath.text(), self.chk_Deep_Check.checkState())
        self.setImage("result.jpg")
        QMessageBox.information(self, "Detect-Result", ret_msg, QMessageBox.Yes);
    def on_btnOpenFile_Clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Image Files (*.png | *.bmp | *.jpg )",
                                                  options=options)
        if fileName:
            self.edit_FilePath.setText(fileName)
            self.setImage(fileName)
            self.btn_Detect.setEnabled(False)
            if (self.b_Trained == True):
                self.btn_Detect.setEnabled(True)
    def on_Close(self):
        exit(1)