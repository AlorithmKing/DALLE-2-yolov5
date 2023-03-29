from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction, QMessageBox

from DetThread import DetThread
from TaskThread import TaskThread
from win_main.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon
from Logger import Logger
import sys

import json

import os
import time
import cv2
import configparser

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox

from utils.capnums import Camera
# from dialog.rtsp_win import Window
from ProgressDialogWidget import ProgressDialogWidget


class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        # style 1: window can be stretched
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: window can not be stretched
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # self.setWindowOpacity(0.85)  # Transparency of window

        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # show Maximized window
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # search models automatically
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5 thread
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '0'
        # self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        # self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        # self.cameraButton.clicked.connect(self.chose_cam)
        # self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        # self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))
        self.stopButton.clicked.connect(self.generate_image)
        self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()
        self.init_config()

    def init_config(self):
        config = configparser.ConfigParser()
        config.read('config/config.ini')
        self.api_key = config.get('openai', 'api_key')
        self.dir_path = config.get('openai', 'dir_path')

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    def checkrate(self):
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    # def chose_rtsp(self):
    #     self.rtsp_window = Window()
    #     config_file = 'config/ip.json'
    #     if not os.path.exists(config_file):
    #         ip = "rtsp://admin:admin888@192.168.1.67:555"
    #         new_config = {"ip": ip}
    #         new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
    #         with open(config_file, 'w', encoding='utf-8') as f:
    #             f.write(new_json)
    #     else:
    #         config = json.load(open(config_file, 'r', encoding='utf-8'))
    #         ip = config['ip']
    #     self.rtsp_window.rtspEdit.setText(ip)
    #     self.rtsp_window.show()
    #     self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading rtsp stream', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def generate_image(self):

        if not self.api_key:
            self.show_error_message("请先配置 OpenAI API Key!")
            return
        else:
            print("successfully input api_key")
        self.prompt = self.prompt_field.text().strip()
        if not self.prompt:
            self.show_error_message("请输入用于生成图像的提示语!")
            return
        else:
            print("successfully input  prompt")
        # 获取当前时间戳
        timestamp = str(int(time.time()))
        # 组成文件名
        filename = f"generated_image_{timestamp}"

        self.task_thread = TaskThread(self.prompt, filename, self.api_key, self.dir_path)
        self.progress_dialog = ProgressDialogWidget()
        self.progress_dialog.canceled.connect(self.task_thread.stop)
        self.task_thread.progress_signal.connect(self.progress_dialog.set_progress_value)
        self.task_thread.error_signal.connect(self.show_error_message)
        self.task_thread.finished.connect(self.progress_dialog.close)
        self.progress_dialog.show()
        self.task_thread.start()

    def show_error_message(self, message):
        QMessageBox.critical(self, '错误', message)
        print(message)

    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading camera', time=2000, auto=True).exec_()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('Loading camera：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x * 100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x / 100)
            self.det_thread.conf_thres = x / 100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x * 100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x / 100)
            self.det_thread.iou_thres = x / 100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def open_file(self):

        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                              "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' ' + str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)

    def set_api_key(self, api_key):
        self.api_key = api_key

    def set_dir_path(self, dir_path):
        self.dir_path = dir_path


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    config = configparser.ConfigParser()
    # config.read('config/config.ini')
    api_key = config.get('openai', 'api_key')
    dir_path = config.get('openai', 'dir_path')
    myWin.set_api_key(api_key)
    myWin.set_dir_path(dir_path)
    # # 添加日志记录器配置
    # Logger()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())
