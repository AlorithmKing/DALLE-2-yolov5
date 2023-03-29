import time
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QApplication


class ProgressDialogWidget(QWidget):
    canceled = pyqtSignal()

    def __init__(self, parent=None):
        super(ProgressDialogWidget, self).__init__(parent)
        self.progress_bar = QProgressBar()
        self.resize(100,200)
        self.progress_bar.setTextVisible(False)  # 隐藏文本
        self.progress_bar.setRange(0, 100)  # 设置范围
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                background-color: #FFFFFF;
            }

            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 lime, stop:1 green);
                border-radius: 5px;
            }
        """)  # 设置样式表
        layout = QVBoxLayout()
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)
        self.setWindowTitle("正在生成图像")

    def set_progress_value(self, value):
        self.progress_bar.setValue(value)
        time.sleep(0.1)  # 模拟任务执行时间
        QApplication.processEvents()  # 刷新界面，实现动态效果
