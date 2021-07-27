#!/usr/bin/env python3
import os
import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from camera_ui import *
from PIL import Image
import rospy
import numpy as np
from sensor_msgs.msg import Image as RosImage


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        icon_path = os.path.join(sys.path[0], '../camera.svg')
        self.setWindowIcon(QtGui.QIcon(icon_path))
        rospy.init_node('dataset_capture', anonymous=False)
        self.image_sub = rospy.Subscriber('/usb_cam/image_rect_color', RosImage, self.image_callback)
        self.pushButton_select.clicked.connect(self.select_folder)
        self.pushButton_save.clicked.connect(self.save_picture)
        self.snapshot = None

    def save_picture(self):
        if self.snapshot is not None:
            image = Image.fromarray(self.snapshot)
            filename = self.lineEdit_prefix.text()
            save_path = self.lineEdit_path.text()
            index = self.lineEdit_index.text()
            index = int(index)
            while os.path.exists(os.path.join(save_path, filename + '_' + str(index) + '.jpg')):
                index += 1
            image.save(os.path.join(save_path, filename + '_' + str(index) + '.jpg'), quality=100)

    def select_folder(self):
        s = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if s:
            self.lineEdit_path.setText(s)

    def image_callback(self, ros_image):
        frame = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data)
        self.snapshot = frame
        img = QImage(frame.data, ros_image.width, ros_image.height, QImage.Format_RGB888).scaled(400, 300)
        self.label_img.setPixmap(QPixmap.fromImage(img))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
