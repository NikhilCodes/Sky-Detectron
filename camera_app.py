import sys
import cv2
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from functools import partial
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtRemoveInputHook, QTimer, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QLabel, QFileDialog
from PyQt5 import uic
from tensorflow.keras.models import load_model

Ui_MainWindow, QtBaseClass = uic.loadUiType('UI/VisualUI.ui')
sky_masking_model = load_model("model_dir/model.h5")


def generate_masked_img(img, k=5, erode_iter=4):
    """
    k: de-noising factor
    """
    img = cv2.resize(img, (512, 512)) / 255
    masked_img = sky_masking_model.predict(np.array([img]))[0].T[0].T
    
    blurred = cv2.GaussianBlur(masked_img, (11, 11), 0) ** k
    
    thresh = cv2.threshold(blurred*255, 100, 142, cv2.THRESH_BINARY)[1]

    print(thresh.shape)
    # perform a series of erosion to remove
    # any small blobs of noise from the threshold image
    thresh = cv2.erode(thresh, None, iterations=erode_iter)

    return thresh


class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.camera = cv2.VideoCapture(0)
        self.WIDTH, self.HEIGHT = (850, 550)
        status, _ = self.camera.read()

        menu_bar = self.menuBar()
        action_menu = menu_bar.addMenu('&Actions')

        load_menu = action_menu.addMenu('&Load From')

        # From Camera
        self.camera_action = QAction('&Camera', self)

        if not status:
            self.camera_action.setEnabled(False)

        self.camera_action.triggered.connect(self.start_cam)
        load_menu.addAction(self.camera_action)

        self.ui.detectButton.clicked.connect(self.detect_objects)

        # From Local
        self.local_action = QAction('&Local', self)
        self.local_action.triggered.connect(self.load_local)
        load_menu.addAction(self.local_action)

        # Save Current Frame
        self.save_frame_action = QAction('&Save Frame', self)
        self.save_frame_action.setShortcut("Ctrl+S")
        self.save_frame_action.triggered.connect(self.save_frame)
        action_menu.addAction(self.save_frame_action)

        # Some Working Variables
        self.CLOSE_ALL_THREAD = False
        self.RUN_FRAMES_FROM_CAMERA = True
        self.MODE = "CAM" # Can be "CAM" or "LOC"

        if not status:
            self.MODE = "LOC"
            self.RUN_FRAMES_FROM_CAMERA = False

        self.frame = None
        self.t1 = threading.Thread(target=self.runVideoFromCam, args=())
        self.t1.start()

    def detect_objects(self):
        threading.Thread(target=self.detect_objects_sub, args=()).start()

    def detect_objects_sub(self):
        """
        if self.ui.detectButton.text() == "New\n":
            if self.MODE == "LOC":
                self.load_local()
                return

            self.start_cam()
            self.ui.detectButton.setText("DETECT\n")
            return
        """

        if self.ui.detectButton.text() == "DONE\n":
            return

        self.pause_cam()
        self.ui.detectButton.setEnabled(False)
        self.ui.detectButton.setText("WAIT\n")
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        masked_img = generate_masked_img(self.frame)
        masked_img = masked_img / np.max(masked_img)
        masked_img = masked_img * 255
        masked_img = np.stack((masked_img,)*3, axis=-1)

        masked_img = self.frame = cv2.resize(masked_img, (self.WIDTH, self.HEIGHT))
        cv2.imwrite("last_processed_img.jpg", cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
        masked_img = cv2.imread("last_processed_img.jpg")
        self.display_mat_frame(masked_img)
        self.ui.detectButton.setText("DONE\n")
        self.ui.detectButton.setEnabled(True)

    def closeEvent(self, event):
        self.CLOSE_ALL_THREAD = True

    def runVideoFromCam(self):
        """
        Type: EventListener
        Must Run Infinitely all time!
        Must be called Once Only!
        :return: None
        """

        while True:
            if self.CLOSE_ALL_THREAD:
                break

            if self.RUN_FRAMES_FROM_CAMERA:
                ret, self.frame = self.camera.read()
                self.frame = cv2.cvtColor(cv2.resize(self.frame, (self.WIDTH, self.HEIGHT)), cv2.COLOR_BGR2RGB)
                if ret:
                    self.display_mat_frame(self.frame)

    def display_mat_frame(self, mat_src, mode='rgb'):
        print('>>', mat_src.shape)
        height, width = mat_src.shape[:2]
        channels = 3

        bytesPerLine = width * channels
        qImg = QImage(mat_src.data, width, height, bytesPerLine, QImage.Format_RGB888)

        pixmap01 = QPixmap.fromImage(qImg)
        pixmap_image = QPixmap(pixmap01)
        self.ui.frame_display.setPixmap(pixmap_image)

    def load_local(self):
        self.pause_cam()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image File", "","All Picture Files (*.jpg *.png *jpeg *.tif)")
        if fileName == '':
            if self.MODE == "CAM":
                self.start_cam()
            return

        self.MODE = "LOC"
        self.frame = cv2.resize(cv2.cvtColor(cv2.imread(fileName), cv2.COLOR_BGR2RGB), (self.WIDTH, self.HEIGHT))
        self.display_mat_frame(self.frame)
        self.ui.detectButton.setText("DETECT\n")

    def save_frame(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save File', 'frame_'+str(time.time())+'.jpg', 'All Picture Files (*.jpg *.png *jpeg *.tif)')
        if filename == '':
            return

        cv2.imwrite(filename, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))

    def pause_cam(self):
        self.RUN_FRAMES_FROM_CAMERA = False

    def start_cam(self):
        self.MODE = "CAM"
        self.RUN_FRAMES_FROM_CAMERA = True


if __name__ == '__main__':
    pyqtRemoveInputHook()
    app = QApplication(sys.argv)
    window = MyApp()
    window.setWindowTitle('SKY-Detectron')
    window.show()
    sys.exit(app.exec())
