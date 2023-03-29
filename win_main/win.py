# -*- coding: utf-8 -*-


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QBrush, QPixmap, QPalette

from PyQt5.QtWidgets import QLineEdit


# pyrcc5 resource.qrc -o resource_rc.py
class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1394, 762)
        mainWindow.setMouseTracking(True)


        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/img/icon/openai_big.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        mainWindow.setWindowIcon(icon)
        mainWindow.setStyleSheet("#mainWindow{border:none;}")

        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)  # 垂直排列
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        # 组件控制组
        self.groupBox_18 = QtWidgets.QGroupBox(self.centralwidget)
        # 背景图片
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("icon/OpenAI-Dall-E-2.jpg")))
        self.setPalette(palette)

        # 一系列垂直排列
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_18)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")

        self.groupBox = QtWidgets.QGroupBox(self.groupBox_18)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 45))
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 45))
        self.groupBox.setStyleSheet("#groupBox{\n"
                                    "background-color: rgba(75, 75, 75, 200);\n"
                                    "border: 0px solid #42adff;\n"
                                    "border-left: 0px solid rgba(29, 83, 185, 255);\n"
                                    "border-right: 0px solid rgba(29, 83, 185, 255);\n"
                                    "border-bottom: 1px solid rgba(200, 200, 200,100);\n"
                                    ";\n"
                                    "border-radius:0px;}")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")

        # 一系列水平排列
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setMinimumSize(QtCore.QSize(40, 40))
        self.label_7.setMaximumSize(QtCore.QSize(40, 40))
        self.label_7.setStyleSheet("image: url(:/img/icon/openai.png);")
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        #   YOLOv5 GUI icon
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setStyleSheet("QLabel\n"
                                   "{\n"
                                   "    font-size: 24px;\n"
                                   "    font-family: \"Microsoft YaHei\";\n"
                                   "    font-weight: bold;\n"
                                   "         border-radius:9px;\n"
                                   "        background:rgba(66, 195, 255, 0);\n"
                                   "color: rgb(218, 218, 218);\n"
                                   "}\n"
                                   "")
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")

        # 最小化icon的UI设计
        self.minButton = QtWidgets.QPushButton(self.groupBox)
        self.minButton.setMinimumSize(QtCore.QSize(50, 28))
        self.minButton.setMaximumSize(QtCore.QSize(50, 28))
        self.minButton.setStyleSheet("QPushButton {\n"
                                     "border-style: solid;\n"
                                     "border-width: 0px;\n"
                                     "border-radius: 0px;\n"
                                     "background-color: rgba(223, 223, 223, 0);}\n"
                                     "QPushButton::focus{outline: none;}\n"
                                     "QPushButton::hover {\n"
                                     "border-style: solid;\n"
                                     "border-width: 0px;\n"
                                     "border-radius: 0px;\n"
                                     "background-color: rgba(223, 223, 223, 150);}")
        self.minButton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/img/icon/最小化.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.minButton.setIcon(icon1)
        self.minButton.setObjectName("minButton")
        self.horizontalLayout_5.addWidget(self.minButton)  # add min_button icon
        # 最大化icon的UI设计
        self.maxButton = QtWidgets.QPushButton(self.groupBox)
        self.maxButton.setMinimumSize(QtCore.QSize(50, 28))
        self.maxButton.setMaximumSize(QtCore.QSize(50, 28))
        self.maxButton.setStyleSheet("QPushButton {\n"
                                     "border-style: solid;\n"
                                     "border-width: 0px;\n"
                                     "border-radius: 0px;\n"
                                     "background-color: rgba(223, 223, 223, 0);}\n"
                                     "QPushButton::focus{outline: none;}\n"
                                     "QPushButton::hover {\n"
                                     "border-style: solid;\n"
                                     "border-width: 0px;\n"
                                     "border-radius: 0px;\n"
                                     "background-color: rgba(223, 223, 223, 150);}")
        self.maxButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/img/icon/正方形.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon2.addPixmap(QtGui.QPixmap(":/img/icon/还原.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        icon2.addPixmap(QtGui.QPixmap(":/img/icon/还原.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.maxButton.setIcon(icon2)
        self.maxButton.setCheckable(True)
        self.maxButton.setObjectName("maxButton")
        self.horizontalLayout_5.addWidget(self.maxButton)
        # 关闭iconUI设计
        self.closeButton = QtWidgets.QPushButton(self.groupBox)
        self.closeButton.setMinimumSize(QtCore.QSize(50, 28))
        self.closeButton.setMaximumSize(QtCore.QSize(50, 28))
        self.closeButton.setStyleSheet("QPushButton {\n"
                                       "border-style: solid;\n"
                                       "border-width: 0px;\n"
                                       "border-radius: 0px;\n"
                                       "background-color: rgba(223, 223, 223, 0);}\n"
                                       "QPushButton::focus{outline: none;}\n"
                                       "QPushButton::hover {\n"
                                       "border-style: solid;\n"
                                       "border-width: 0px;\n"
                                       "border-radius: 0px;\n"
                                       "background-color: rgba(223, 223, 223, 150);}")
        self.closeButton.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/img/icon/关闭.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.closeButton.setIcon(icon3)
        self.closeButton.setObjectName("closeButton")
        self.horizontalLayout_5.addWidget(self.closeButton)
        self.horizontalLayout.addLayout(self.horizontalLayout_5)
        self.verticalLayout_6.addWidget(self.groupBox)

        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setSpacing(0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox_18)
        self.groupBox_8.setMinimumSize(QtCore.QSize(320, 0))
        self.groupBox_8.setMaximumSize(QtCore.QSize(320, 16777215))
        self.groupBox_8.setStyleSheet("#groupBox_8{\n"
                                      "background-color: rgba(75, 75, 75, 200);\n"
                                      "border: 0px solid #42adff;\n"
                                      "border-radius:0px;}\n"
                                      "")
        self.groupBox_8.setTitle("")
        self.groupBox_8.setObjectName("groupBox_8")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBox_8)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setSpacing(11)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_2.setMinimumSize(QtCore.QSize(0, 42))
        self.groupBox_2.setMaximumSize(QtCore.QSize(16777215, 42))
        self.groupBox_2.setStyleSheet("#groupBox_2{\n"
                                      "border: 0px solid #42adff;\n"
                                      "border-bottom: 1px solid rgba(200, 200, 200,100);\n"
                                      "border-radius:0px;}")
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayout_35 = QtWidgets.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_35.setContentsMargins(11, 0, 11, 0)
        self.horizontalLayout_35.setObjectName("horizontalLayout_35")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setMinimumSize(QtCore.QSize(0, 0))
        self.label_5.setMaximumSize(QtCore.QSize(16777215, 40))
        self.label_5.setStyleSheet("QLabel\n"
                                   "{\n"
                                   "    font-size: 22px;\n"
                                   "    font-family: \"Microsoft YaHei\";\n"
                                   "    font-weight: bold;\n"
                                   "         border-radius:9px;\n"
                                   "        background:rgba(66, 195, 255, 0);\n"
                                   "color: rgb(218, 218, 218);\n"
                                   "\n"
                                   "}\n"
                                   "")
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_35.addWidget(self.label_5)
        spacerItem1 = QtWidgets.QSpacerItem(37, 39, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_35.addItem(spacerItem1)
        self.verticalLayout_8.addWidget(self.groupBox_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(11, -1, 11, -1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox_8)
        self.label_3.setMinimumSize(QtCore.QSize(0, 28))
        self.label_3.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label_3.setStyleSheet("QLabel\n"
                                   "{\n"
                                   "    font-size: 18px;\n"
                                   "    font-family: \"Microsoft YaHei\";\n"
                                   "    font-weight: bold;\n"
                                   "         border-radius:9px;\n"
                                   "        background:rgba(66, 195, 255, 0);\n"
                                   "color: rgb(218, 218, 218);\n"
                                   "}\n"
                                   "")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_2.addWidget(self.label_3)
        self.comboBox = QtWidgets.QComboBox(self.groupBox_8)
        self.comboBox.setMinimumSize(QtCore.QSize(0, 28))
        self.comboBox.setStyleSheet("QComboBox QAbstractItemView {\n"
                                    "font-family: \"Microsoft YaHei\";\n"
                                    "font-size: 16px;\n"
                                    "background:rgba(200, 200, 200,150);\n"
                                    "selection-background-color: rgba(200, 200, 200,50);\n"
                                    "color: rgb(218, 218, 218);\n"
                                    "outline:none;\n"
                                    "border:none;}\n"
                                    "QComboBox{\n"
                                    "font-family: \"Microsoft YaHei\";\n"
                                    "font-size: 16px;\n"
                                    "color: rgb(218, 218, 218);\n"
                                    "border-width:0px;\n"
                                    "border-color:white;\n"
                                    "border-style:solid;\n"
                                    "background-color: rgba(200, 200, 200,0);}\n"
                                    "\n"
                                    "QComboBox::drop-down {\n"
                                    "margin-top:8;\n"
                                    "height:20;\n"
                                    "background:rgba(255,255,255,0);\n"
                                    "border-image: url(:/img/icon/下拉_白色.png);\n"
                                    "}\n"
                                    "")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("1")
        self.comboBox.addItem("2")
        self.comboBox.addItem("3")
        self.comboBox.addItem("4")
        self.horizontalLayout_2.addWidget(self.comboBox)
        self.verticalLayout_8.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setContentsMargins(11, -1, 0, -1)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox_8)
        self.label_10.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label_10.setStyleSheet("QLabel\n"
                                    "{\n"
                                    "    font-size: 18px;\n"
                                    "    font-family: \"Microsoft YaHei\";\n"
                                    "    font-weight: bold;\n"
                                    "         border-radius:9px;\n"
                                    "        background:rgba(66, 195, 255, 0);\n"
                                    "color: rgb(218, 218, 218);\n"
                                    "}\n"
                                    "")
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_9.addWidget(self.label_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_5.setStyleSheet("#groupBox_5{\n"
                                      "background-color: rgba(48,148,243,0);\n"
                                      "border: 0px solid #42adff;\n"
                                      "border-left: 0px solid #d9d9d9;\n"
                                      "border-right: 0px solid rgba(29, 83, 185, 255);\n"
                                      "border-radius:0px;}")
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.groupBox_5)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.fileButton = QtWidgets.QPushButton(self.groupBox_5)
        self.fileButton.setMinimumSize(QtCore.QSize(55, 28))
        self.fileButton.setMaximumSize(QtCore.QSize(16777215, 28))
        self.fileButton.setStyleSheet("QPushButton{font-family: \"Microsoft YaHei\";\n"
                                      "font-size: 14px;\n"
                                      "font-weight: bold;\n"
                                      "color:white;\n"
                                      "text-align: center center;\n"
                                      "padding-left: 5px;\n"
                                      "padding-right: 5px;\n"
                                      "padding-top: 4px;\n"
                                      "padding-bottom: 4px;\n"
                                      "border-style: solid;\n"
                                      "border-width: 0px;\n"
                                      "border-color: rgba(255, 255, 255, 255);\n"
                                      "border-radius: 3px;\n"
                                      "background-color: rgba(200, 200, 200,0);}\n"
                                      "\n"
                                      "QPushButton:focus{outline: none;}\n"
                                      "\n"
                                      "QPushButton::pressed{font-family: \"Microsoft YaHei\";\n"
                                      "                     font-size: 14px;\n"
                                      "                     font-weight: bold;\n"
                                      "                     color:rgb(200,200,200);\n"
                                      "                     text-align: center center;\n"
                                      "                     padding-left: 5px;\n"
                                      "                     padding-right: 5px;\n"
                                      "                     padding-top: 4px;\n"
                                      "                     padding-bottom: 4px;\n"
                                      "                     border-style: solid;\n"
                                      "                     border-width: 0px;\n"
                                      "                     border-color: rgba(255, 255, 255, 255);\n"
                                      "                     border-radius: 3px;\n"
                                      "                     background-color:  #bf513b;}\n"
                                      "\n"
                                      "QPushButton::disabled{font-family: \"Microsoft YaHei\";\n"
                                      "                     font-size: 14px;\n"
                                      "                     font-weight: bold;\n"
                                      "                     color:rgb(200,200,200);\n"
                                      "                     text-align: center center;\n"
                                      "                     padding-left: 5px;\n"
                                      "                     padding-right: 5px;\n"
                                      "                     padding-top: 4px;\n"
                                      "                     padding-bottom: 4px;\n"
                                      "                     border-style: solid;\n"
                                      "                     border-width: 0px;\n"
                                      "                     border-color: rgba(255, 255, 255, 255);\n"
                                      "                     border-radius: 3px;\n"
                                      "                     background-color:  #bf513b;}\n"
                                      "QPushButton::hover {\n"
                                      "border-style: solid;\n"
                                      "border-width: 0px;\n"
                                      "border-radius: 0px;\n"
                                      "background-color: rgba(48,148,243,80);}")
        self.fileButton.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/img/icon/打开.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.fileButton.setIcon(icon4)
        self.fileButton.setObjectName("fileButton")
        self.horizontalLayout_8.addWidget(self.fileButton)
        self.cameraButton = QtWidgets.QPushButton(self.groupBox_5)
        self.cameraButton.setMinimumSize(QtCore.QSize(55, 28))
        self.cameraButton.setMaximumSize(QtCore.QSize(16777215, 28))
        self.cameraButton.setStyleSheet("QPushButton{font-family: \"Microsoft YaHei\";\n"
                                        "font-size: 14px;\n"
                                        "font-weight: bold;\n"
                                        "color:white;\n"
                                        "text-align: center center;\n"
                                        "padding-left: 5px;\n"
                                        "padding-right: 5px;\n"
                                        "padding-top: 4px;\n"
                                        "padding-bottom: 4px;\n"
                                        "border-style: solid;\n"
                                        "border-width: 0px;\n"
                                        "border-color: rgba(255, 255, 255, 255);\n"
                                        "border-radius: 3px;\n"
                                        "background-color: rgba(48,148,243,0);}\n"
                                        "\n"
                                        "QPushButton:focus{outline: none;}\n"
                                        "\n"
                                        "QPushButton::pressed{font-family: \"Microsoft YaHei\";\n"
                                        "                     font-size: 14px;\n"
                                        "                     font-weight: bold;\n"
                                        "                     color:rgb(200,200,200);\n"
                                        "                     text-align: center center;\n"
                                        "                     padding-left: 5px;\n"
                                        "                     padding-right: 5px;\n"
                                        "                     padding-top: 4px;\n"
                                        "                     padding-bottom: 4px;\n"
                                        "                     border-style: solid;\n"
                                        "                     border-width: 0px;\n"
                                        "                     border-color: rgba(255, 255, 255, 255);\n"
                                        "                     border-radius: 3px;\n"
                                        "                     background-color:  #bf513b;}\n"
                                        "\n"
                                        "QPushButton::disabled{font-family: \"Microsoft YaHei\";\n"
                                        "                     font-size: 14px;\n"
                                        "                     font-weight: bold;\n"
                                        "                     color:rgb(200,200,200);\n"
                                        "                     text-align: center center;\n"
                                        "                     padding-left: 5px;\n"
                                        "                     padding-right: 5px;\n"
                                        "                     padding-top: 4px;\n"
                                        "                     padding-bottom: 4px;\n"
                                        "                     border-style: solid;\n"
                                        "                     border-width: 0px;\n"
                                        "                     border-color: rgba(255, 255, 255, 255);\n"
                                        "                     border-radius: 3px;\n"
                                        "                     background-color:  #bf513b;}\n"
                                        "QPushButton::hover {\n"
                                        "border-style: solid;\n"
                                        "border-width: 0px;\n"
                                        "border-radius: 0px;\n"
                                        "background-color: rgba(48,148,243,80);}")
        self.cameraButton.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/img/icon/摄像头开.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.cameraButton.setIcon(icon5)
        self.cameraButton.setObjectName("cameraButton")
        self.horizontalLayout_8.addWidget(self.cameraButton)
        self.rtspButton = QtWidgets.QPushButton(self.groupBox_5)
        self.rtspButton.setMinimumSize(QtCore.QSize(55, 28))
        self.rtspButton.setMaximumSize(QtCore.QSize(16777215, 28))
        self.rtspButton.setStyleSheet("QPushButton{font-family: \"Microsoft YaHei\";\n"
                                      "font-size: 14px;\n"
                                      "font-weight: bold;\n"
                                      "color:white;\n"
                                      "text-align: center center;\n"
                                      "padding-left: 5px;\n"
                                      "padding-right: 5px;\n"
                                      "padding-top: 4px;\n"
                                      "padding-bottom: 4px;\n"
                                      "border-style: solid;\n"
                                      "border-width: 0px;\n"
                                      "border-color: rgba(255, 255, 255, 255);\n"
                                      "border-radius: 3px;\n"
                                      "background-color: rgba(48,148,243,0);}\n"
                                      "\n"
                                      "QPushButton:focus{outline: none;}\n"
                                      "\n"
                                      "QPushButton::pressed{font-family: \"Microsoft YaHei\";\n"
                                      "                     font-size: 14px;\n"
                                      "                     font-weight: bold;\n"
                                      "                     color:rgb(200,200,200);\n"
                                      "                     text-align: center center;\n"
                                      "                     padding-left: 5px;\n"
                                      "                     padding-right: 5px;\n"
                                      "                     padding-top: 4px;\n"
                                      "                     padding-bottom: 4px;\n"
                                      "                     border-style: solid;\n"
                                      "                     border-width: 0px;\n"
                                      "                     border-color: rgba(255, 255, 255, 255);\n"
                                      "                     border-radius: 3px;\n"
                                      "                     background-color:  #bf513b;}\n"
                                      "\n"
                                      "QPushButton::disabled{font-family: \"Microsoft YaHei\";\n"
                                      "                     font-size: 14px;\n"
                                      "                     font-weight: bold;\n"
                                      "                     color:rgb(200,200,200);\n"
                                      "                     text-align: center center;\n"
                                      "                     padding-left: 5px;\n"
                                      "                     padding-right: 5px;\n"
                                      "                     padding-top: 4px;\n"
                                      "                     padding-bottom: 4px;\n"
                                      "                     border-style: solid;\n"
                                      "                     border-width: 0px;\n"
                                      "                     border-color: rgba(255, 255, 255, 255);\n"
                                      "                     border-radius: 3px;\n"
                                      "                     background-color:  #bf513b;}\n"
                                      "QPushButton::hover {\n"
                                      "border-style: solid;\n"
                                      "border-width: 0px;\n"
                                      "border-radius: 0px;\n"
                                      "background-color: rgba(48,148,243,80);}")
        self.rtspButton.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/img/icon/实时视频流解析.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rtspButton.setIcon(icon6)
        self.rtspButton.setObjectName("rtspButton")
        self.horizontalLayout_8.addWidget(self.rtspButton)
        self.horizontalLayout_11.addWidget(self.groupBox_5)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_11)
        self.verticalLayout_8.addLayout(self.horizontalLayout_9)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(11, -1, 11, -1)
        self.verticalLayout_3.setSpacing(4)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox_8)
        self.label_2.setStyleSheet("QLabel\n"
                                   "{\n"
                                   "    font-size: 18px;\n"
                                   "    font-family: \"Microsoft YaHei\";\n"
                                   "    font-weight: bold;\n"
                                   "         border-radius:9px;\n"
                                   "        background:rgba(66, 195, 255, 0);\n"
                                   "color: rgb(218, 218, 218);\n"
                                   "}\n"
                                   "")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(5)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.iouSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_8)
        self.iouSpinBox.setMinimumSize(QtCore.QSize(50, 0))
        self.iouSpinBox.setMaximumSize(QtCore.QSize(50, 16777215))
        self.iouSpinBox.setStyleSheet("QDoubleSpinBox{\n"
                                      "background:rgba(200, 200, 200,50);\n"
                                      "color:white;\n"
                                      "font-size: 14px;\n"
                                      "font-family: \"Microsoft YaHei UI\";\n"
                                      "border-style: solid;\n"
                                      "border-width: 1px;\n"
                                      "border-color: rgba(200, 200, 200,100);\n"
                                      "border-radius: 3px;}\n"
                                      "\n"
                                      "QDoubleSpinBox::down-button{\n"
                                      "background:rgba(200, 200, 200,0);\n"
                                      "border-image: url(:/img/icon/箭头_列表展开.png);}\n"
                                      "QDoubleSpinBox::down-button::hover{\n"
                                      "background:rgba(200, 200, 200,100);\n"
                                      "border-image: url(:/img/icon/箭头_列表展开.png);}\n"
                                      "\n"
                                      "QDoubleSpinBox::up-button{\n"
                                      "background:rgba(200, 200, 200,0);\n"
                                      "border-image: url(:/img/icon/箭头_列表收起.png);}\n"
                                      "QDoubleSpinBox::up-button::hover{\n"
                                      "background:rgba(200, 200, 200,100);\n"
                                      "border-image: url(:/img/icon/箭头_列表收起.png);}\n"
                                      "")
        self.iouSpinBox.setMaximum(1.0)
        self.iouSpinBox.setSingleStep(0.01)
        self.iouSpinBox.setProperty("value", 0.45)
        self.iouSpinBox.setObjectName("iouSpinBox")
        self.horizontalLayout_4.addWidget(self.iouSpinBox)
        self.iouSlider = QtWidgets.QSlider(self.groupBox_8)
        self.iouSlider.setStyleSheet("QSlider{\n"
                                     "border-color: #bcbcbc;\n"
                                     "color:#d9d9d9;\n"
                                     "}\n"
                                     "QSlider::groove:horizontal {                                \n"
                                     "     border: 1px solid #999999;                             \n"
                                     "     height: 3px;                                           \n"
                                     "    margin: 0px 0;                                         \n"
                                     "     left: 5px; right: 5px; \n"
                                     " }\n"
                                     "QSlider::handle:horizontal {                               \n"
                                     "     border: 0px ; \n"
                                     "     border-image: url(:/img/icon/圆.png);\n"
                                     "     width:15px;\n"
                                     "     margin: -7px -7px -7px -7px;                  \n"
                                     "} \n"
                                     "QSlider::add-page:horizontal{\n"
                                     "background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #d9d9d9, stop:0.25 #d9d9d9, stop:0.5 #d9d9d9, stop:1 #d9d9d9); \n"
                                     "\n"
                                     "}\n"
                                     "QSlider::sub-page:horizontal{                               \n"
                                     " background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #373737, stop:0.25 #373737, stop:0.5 #373737, stop:1 #373737);                     \n"
                                     "}")
        self.iouSlider.setMaximum(100)
        self.iouSlider.setProperty("value", 45)
        self.iouSlider.setOrientation(QtCore.Qt.Horizontal)
        self.iouSlider.setObjectName("iouSlider")
        self.horizontalLayout_4.addWidget(self.iouSlider)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.verticalLayout_8.addLayout(self.verticalLayout_3)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(11, -1, 11, -1)
        self.verticalLayout.setSpacing(4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.groupBox_8)
        self.label.setStyleSheet("QLabel\n"
                                 "{\n"
                                 "    font-size: 18px;\n"
                                 "    font-family: \"Microsoft YaHei\";\n"
                                 "    font-weight: bold;\n"
                                 "         border-radius:9px;\n"
                                 "        background:rgba(66, 195, 255, 0);\n"
                                 "color: rgb(218, 218, 218);\n"
                                 "}\n"
                                 "")
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSpacing(5)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.confSpinBox = QtWidgets.QDoubleSpinBox(self.groupBox_8)
        self.confSpinBox.setMinimumSize(QtCore.QSize(50, 0))
        self.confSpinBox.setMaximumSize(QtCore.QSize(50, 16777215))
        self.confSpinBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.confSpinBox.setStyleSheet("QDoubleSpinBox{\n"
                                       "background:rgba(200, 200, 200,50);\n"
                                       "color:white;\n"
                                       "font-size: 14px;\n"
                                       "font-family: \"Microsoft YaHei UI\";\n"
                                       "border-style: solid;\n"
                                       "border-width: 1px;\n"
                                       "border-color: rgba(200, 200, 200,100);\n"
                                       "border-radius: 3px;}\n"
                                       "\n"
                                       "QDoubleSpinBox::down-button{\n"
                                       "background:rgba(200, 200, 200,0);\n"
                                       "border-image: url(:/img/icon/箭头_列表展开.png);}\n"
                                       "QDoubleSpinBox::down-button::hover{\n"
                                       "background:rgba(200, 200, 200,100);\n"
                                       "border-image: url(:/img/icon/箭头_列表展开.png);}\n"
                                       "\n"
                                       "QDoubleSpinBox::up-button{\n"
                                       "background:rgba(200, 200, 200,0);\n"
                                       "border-image: url(:/img/icon/箭头_列表收起.png);}\n"
                                       "QDoubleSpinBox::up-button::hover{\n"
                                       "background:rgba(200, 200, 200,100);\n"
                                       "border-image: url(:/img/icon/箭头_列表收起.png);}\n"
                                       "")
        self.confSpinBox.setMaximum(1.0)
        self.confSpinBox.setSingleStep(0.01)
        self.confSpinBox.setProperty("value", 0.25)
        self.confSpinBox.setObjectName("confSpinBox")
        self.horizontalLayout_3.addWidget(self.confSpinBox)
        self.confSlider = QtWidgets.QSlider(self.groupBox_8)
        self.confSlider.setStyleSheet("QSlider{\n"
                                      "border-color: #bcbcbc;\n"
                                      "color:#d9d9d9;\n"
                                      "}\n"
                                      "QSlider::groove:horizontal {                                \n"
                                      "     border: 1px solid #999999;                             \n"
                                      "     height: 3px;                                           \n"
                                      "    margin: 0px 0;                                         \n"
                                      "     left: 5px; right: 5px; \n"
                                      " }\n"
                                      "QSlider::handle:horizontal {                               \n"
                                      "     border: 0px ; \n"
                                      "     border-image: url(:/img/icon/圆.png);\n"
                                      "     width:15px;\n"
                                      "     margin: -7px -7px -7px -7px;                  \n"
                                      "} \n"
                                      "QSlider::add-page:horizontal{\n"
                                      "background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #d9d9d9, stop:0.25 #d9d9d9, stop:0.5 #d9d9d9, stop:1 #d9d9d9); \n"
                                      "\n"
                                      "}\n"
                                      "QSlider::sub-page:horizontal{                               \n"
                                      " background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #373737, stop:0.25 #373737, stop:0.5 #373737, stop:1 #373737);                     \n"
                                      "}")
        self.confSlider.setMaximum(100)
        self.confSlider.setProperty("value", 25)
        self.confSlider.setOrientation(QtCore.Qt.Horizontal)
        self.confSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.confSlider.setObjectName("confSlider")
        self.horizontalLayout_3.addWidget(self.confSlider)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.verticalLayout_8.addLayout(self.verticalLayout)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setContentsMargins(11, -1, 11, -1)
        self.verticalLayout_5.setSpacing(4)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_8 = QtWidgets.QLabel(self.groupBox_8)
        self.label_8.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label_8.setStyleSheet("QLabel\n"
                                   "{\n"
                                   "    font-size: 18px;\n"
                                   "    font-family: \"Microsoft YaHei\";\n"
                                   "    font-weight: bold;\n"
                                   "         border-radius:9px;\n"
                                   "        background:rgba(66, 195, 255, 0);\n"
                                   "color: rgb(218, 218, 218);\n"
                                   "}\n"
                                   "")
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_14.addWidget(self.label_8)
        self.checkBox = QtWidgets.QCheckBox(self.groupBox_8)
        self.checkBox.setStyleSheet("\n"
                                    "QCheckBox\n"
                                    "{font-size: 16px;\n"
                                    "    font-family: \"Microsoft YaHei\";\n"
                                    "    font-weight: bold;\n"
                                    "         border-radius:9px;\n"
                                    "        background:rgba(66, 195, 255, 0);\n"
                                    "color: rgb(218, 218, 218);;}\n"
                                    "\n"
                                    "QCheckBox::indicator {\n"
                                    "    width: 20px;\n"
                                    "    height: 20px;\n"
                                    "}\n"
                                    "QCheckBox::indicator:unchecked {\n"
                                    "    image: url(:/img/icon/button-off.png);\n"
                                    "}\n"
                                    "\n"
                                    "QCheckBox::indicator:checked {\n"
                                    "    \n"
                                    "    image: url(:/img/icon/button-on.png);\n"
                                    "}\n"
                                    "")
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_14.addWidget(self.checkBox)
        self.verticalLayout_5.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setSpacing(5)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.rateSpinBox = QtWidgets.QSpinBox(self.groupBox_8)
        self.rateSpinBox.setMinimumSize(QtCore.QSize(50, 0))
        self.rateSpinBox.setMaximumSize(QtCore.QSize(50, 16777215))
        self.rateSpinBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.rateSpinBox.setStyleSheet("QSpinBox{\n"
                                       "background:rgba(200, 200, 200,50);\n"
                                       "color:white;\n"
                                       "font-size: 14px;\n"
                                       "font-family: \"Microsoft YaHei UI\";\n"
                                       "border-style: solid;\n"
                                       "border-width: 1px;\n"
                                       "border-color: rgba(200, 200, 200,100);\n"
                                       "border-radius: 3px;}\n"
                                       "\n"
                                       "QSpinBox::down-button{\n"
                                       "background:rgba(200, 200, 200,0);\n"
                                       "border-image: url(:/img/icon/箭头_列表展开.png);}\n"
                                       "QDoubleSpinBox::down-button::hover{\n"
                                       "background:rgba(200, 200, 200,100);\n"
                                       "border-image: url(:/img/icon/箭头_列表展开.png);}\n"
                                       "\n"
                                       "QSpinBox::up-button{\n"
                                       "background:rgba(200, 200, 200,0);\n"
                                       "border-image: url(:/img/icon/箭头_列表收起.png);}\n"
                                       "QSpinBox::up-button::hover{\n"
                                       "background:rgba(200, 200, 200,100);\n"
                                       "border-image: url(:/img/icon/箭头_列表收起.png);}\n"
                                       "")
        self.rateSpinBox.setMinimum(1)
        self.rateSpinBox.setMaximum(20)
        self.rateSpinBox.setSingleStep(1)
        self.rateSpinBox.setProperty("value", 1)
        self.rateSpinBox.setObjectName("rateSpinBox")
        self.horizontalLayout_13.addWidget(self.rateSpinBox)
        self.rateSlider = QtWidgets.QSlider(self.groupBox_8)
        self.rateSlider.setStyleSheet("QSlider{\n"
                                      "border-color: #bcbcbc;\n"
                                      "color:#d9d9d9;\n"
                                      "}\n"
                                      "QSlider::groove:horizontal {                                \n"
                                      "     border: 1px solid #999999;                             \n"
                                      "     height: 3px;                                           \n"
                                      "    margin: 0px 0;                                         \n"
                                      "     left: 5px; right: 5px; \n"
                                      " }\n"
                                      "QSlider::handle:horizontal {                               \n"
                                      "     border: 0px ; \n"
                                      "     border-image: url(:/img/icon/圆.png);\n"
                                      "     width:15px;\n"
                                      "     margin: -7px -7px -7px -7px;                  \n"
                                      "} \n"
                                      "QSlider::add-page:horizontal{\n"
                                      "background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #d9d9d9, stop:0.25 #d9d9d9, stop:0.5 #d9d9d9, stop:1 #d9d9d9); \n"
                                      "\n"
                                      "}\n"
                                      "QSlider::sub-page:horizontal{                               \n"
                                      " background: qlineargradient(spread:pad, x1:0, y1:1, x2:0, y2:0, stop:0 #373737, stop:0.25 #373737, stop:0.5 #373737, stop:1 #373737);                     \n"
                                      "}")
        self.rateSlider.setMinimum(1)
        self.rateSlider.setMaximum(20)
        self.rateSlider.setSingleStep(1)
        self.rateSlider.setPageStep(1)
        self.rateSlider.setProperty("value", 1)
        self.rateSlider.setOrientation(QtCore.Qt.Horizontal)
        self.rateSlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.rateSlider.setObjectName("rateSlider")
        self.horizontalLayout_13.addWidget(self.rateSlider)
        self.verticalLayout_5.addLayout(self.horizontalLayout_13)
        self.verticalLayout_8.addLayout(self.verticalLayout_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_6.setMinimumSize(QtCore.QSize(0, 0))
        self.groupBox_6.setMaximumSize(QtCore.QSize(16777215, 42))
        self.groupBox_6.setStyleSheet("#groupBox_6{\n"
                                      "border: 0px solid #42adff;\n"
                                      "border-radius:0px;}")
        self.groupBox_6.setTitle("")
        self.groupBox_6.setObjectName("groupBox_6")
        self.horizontalLayout_36 = QtWidgets.QHBoxLayout(self.groupBox_6)
        self.horizontalLayout_36.setContentsMargins(11, 0, 11, 0)
        self.horizontalLayout_36.setObjectName("horizontalLayout_36")
        self.saveCheckBox = QtWidgets.QCheckBox(self.groupBox_6)
        self.saveCheckBox.setStyleSheet("\n"
                                        "QCheckBox\n"
                                        "{font-size: 18px;\n"
                                        "    font-family: \"Microsoft YaHei\";\n"
                                        "    font-weight: bold;\n"
                                        "         border-radius:9px;\n"
                                        "        background:rgba(66, 195, 255, 0);\n"
                                        "color: rgb(218, 218, 218);;}\n"
                                        "\n"
                                        "QCheckBox::indicator {\n"
                                        "    width: 20px;\n"
                                        "    height: 20px;\n"
                                        "}\n"
                                        "QCheckBox::indicator:unchecked {\n"
                                        "    image: url(:/img/icon/button-off.png);\n"
                                        "}\n"
                                        "\n"
                                        "QCheckBox::indicator:checked {\n"
                                        "    \n"
                                        "    image: url(:/img/icon/button-on.png);\n"
                                        "}\n"
                                        "")
        self.saveCheckBox.setChecked(True)
        self.saveCheckBox.setObjectName("saveCheckBox")
        self.horizontalLayout_36.addWidget(self.saveCheckBox)
        self.verticalLayout_8.addWidget(self.groupBox_6)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setContentsMargins(-1, 0, -1, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.groupBox_9 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_9.setMinimumSize(QtCore.QSize(0, 42))
        self.groupBox_9.setMaximumSize(QtCore.QSize(16777215, 42))
        self.groupBox_9.setStyleSheet("#groupBox_9{\n"
                                      "border: 0px solid #42adff;\n"
                                      "border-top: 1px solid rgba(200, 200, 200,100);\n"
                                      "border-bottom: 1px solid rgba(200, 200, 200,100);\n"
                                      "border-radius:0px;}")
        self.groupBox_9.setTitle("")
        self.groupBox_9.setObjectName("groupBox_9")
        self.horizontalLayout_38 = QtWidgets.QHBoxLayout(self.groupBox_9)
        self.horizontalLayout_38.setContentsMargins(11, 0, 11, 0)
        self.horizontalLayout_38.setObjectName("horizontalLayout_38")
        self.label_11 = QtWidgets.QLabel(self.groupBox_9)
        self.label_11.setStyleSheet("QLabel\n"
                                    "{\n"
                                    "    font-size: 22px;\n"
                                    "    font-family: \"Microsoft YaHei\";\n"
                                    "    font-weight: bold;\n"
                                    "         border-radius:9px;\n"
                                    "        background:rgba(66, 195, 255, 0);\n"
                                    "color: rgb(218, 218, 218);\n"
                                    "}\n"
                                    "")
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_38.addWidget(self.label_11)
        spacerItem2 = QtWidgets.QSpacerItem(37, 39, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_38.addItem(spacerItem2)
        self.verticalLayout_7.addWidget(self.groupBox_9)
        self.groupBox_10 = QtWidgets.QGroupBox(self.groupBox_8)
        self.groupBox_10.setMinimumSize(QtCore.QSize(0, 42))
        self.groupBox_10.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.groupBox_10.setStyleSheet("#groupBox_10{\n"
                                       "border: 0px solid #42adff;\n"
                                       "\n"
                                       "border-radius:0px;}")
        self.groupBox_10.setTitle("")
        self.groupBox_10.setObjectName("groupBox_10")
        self.horizontalLayout_39 = QtWidgets.QHBoxLayout(self.groupBox_10)
        self.horizontalLayout_39.setContentsMargins(11, 0, 11, 0)
        self.horizontalLayout_39.setObjectName("horizontalLayout_39")
        self.resultWidget = QtWidgets.QListWidget(self.groupBox_10)
        self.resultWidget.setStyleSheet("QListWidget{\n"
                                        "background-color: rgba(12, 28, 77, 0);\n"
                                        "\n"
                                        "border-radius:0px;\n"
                                        "font-family: \"Microsoft YaHei\";\n"
                                        "font-size: 16px;\n"
                                        "color: rgb(218, 218, 218);\n"
                                        "}\n"
                                        "")
        self.resultWidget.setObjectName("resultWidget")
        self.horizontalLayout_39.addWidget(self.resultWidget)
        self.verticalLayout_7.addWidget(self.groupBox_10)
        self.verticalLayout_7.setStretch(1, 1)
        self.verticalLayout_8.addLayout(self.verticalLayout_7)
        self.horizontalLayout_7.addWidget(self.groupBox_8)
        self.groupBox_201 = QtWidgets.QGroupBox(self.groupBox_18)
        self.groupBox_201.setStyleSheet("#groupBox_201{\n"
                                        "background-color: rgba(95, 95, 95, 200);\n"
                                        "border: 0px solid #42adff;\n"
                                        "border-left: 1px solid rgba(200, 200, 200,100);\n"
                                        "border-right: 0px solid rgba(29, 83, 185, 255);\n"
                                        "border-radius:0px;}")
        self.groupBox_201.setTitle("")
        self.groupBox_201.setObjectName("groupBox_201")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_201)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_201)
        self.groupBox_3.setMinimumSize(QtCore.QSize(0, 42))
        self.groupBox_3.setMaximumSize(QtCore.QSize(16777215, 42))
        self.groupBox_3.setStyleSheet("#groupBox_3{\n"
                                      "border: 0px solid #42adff;\n"
                                      "border-bottom: 1px solid rgba(200, 200, 200,100);\n"
                                      "border-radius:0px;}")
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_6.setContentsMargins(11, 0, 11, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setMinimumSize(QtCore.QSize(0, 0))
        self.label_6.setMaximumSize(QtCore.QSize(16777215, 40))
        self.label_6.setStyleSheet("QLabel\n"
                                   "{\n"
                                   "    font-size: 22px;\n"
                                   "    font-family: \"Microsoft YaHei\";\n"
                                   "    font-weight: bold;\n"
                                   "         border-radius:9px;\n"
                                   "        background:rgba(66, 195, 255, 0);\n"
                                   "color: rgb(218, 218, 218);\n"
                                   "}\n"
                                   "")
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_6.addWidget(self.label_6)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem3)
        self.fps_label = QtWidgets.QLabel(self.groupBox_3)
        self.fps_label.setMinimumSize(QtCore.QSize(100, 40))
        self.fps_label.setMaximumSize(QtCore.QSize(100, 40))
        self.fps_label.setStyleSheet("QLabel\n"
                                     "{\n"
                                     "    font-size: 20px;\n"
                                     "    font-family: \"Microsoft YaHei\";\n"
                                     "    font-weight: bold;\n"
                                     "         border-radius:9px;\n"
                                     "        background:rgba(66, 195, 255, 0);\n"
                                     "color: rgb(218, 218, 218);\n"
                                     "}\n"
                                     "")
        self.fps_label.setText("")
        self.fps_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fps_label.setObjectName("fps_label")
        self.horizontalLayout_6.addWidget(self.fps_label)
        self.verticalLayout_4.addWidget(self.groupBox_3)
        self.splitter = QtWidgets.QSplitter(self.groupBox_201)
        self.splitter.setEnabled(True)
        self.splitter.setStyleSheet("#splitter::handle{background: 1px solid  rgba(200, 200, 200,100);}")
        self.splitter.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.splitter.setLineWidth(10)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setObjectName("splitter")
        self.raw_video = Label_click_Mouse(self.splitter)
        self.raw_video.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.raw_video.sizePolicy().hasHeightForWidth())
        self.raw_video.setSizePolicy(sizePolicy)
        self.raw_video.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(36)
        self.raw_video.setFont(font)
        self.raw_video.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.raw_video.setStyleSheet("color: rgb(218, 218, 218);\n"
                                     "")
        self.raw_video.setText("")
        self.raw_video.setScaledContents(False)
        self.raw_video.setAlignment(QtCore.Qt.AlignCenter)
        self.raw_video.setObjectName("raw_video")
        self.out_video = Label_click_Mouse(self.splitter)
        self.out_video.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.out_video.sizePolicy().hasHeightForWidth())
        self.out_video.setSizePolicy(sizePolicy)
        self.out_video.setMinimumSize(QtCore.QSize(200, 0))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(36)
        self.out_video.setFont(font)
        self.out_video.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.out_video.setStyleSheet("color: rgb(218, 218, 218);\n"
                                     "")
        self.out_video.setText("")
        self.out_video.setScaledContents(False)
        self.out_video.setAlignment(QtCore.Qt.AlignCenter)
        self.out_video.setObjectName("out_video")
        self.verticalLayout_4.addWidget(self.splitter)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setContentsMargins(11, -1, 11, -1)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")

        self.prompt_field = QLineEdit()
        self.prompt_field.setPlaceholderText("请输入用于生成图像的提示语")
        self.prompt_field.setMinimumSize(QtCore.QSize(100, 60))
        self.horizontalLayout_12.addWidget(self.prompt_field)
        self.prompt_field.setStyleSheet("QLineEdit {\n"
                                        "border-style: solid;\n"
                                        "border-width: 0px;\n"
                                        "border-radius: 0px;\n"
                                        "background-color: rgba(223, 223, 223, 100);\n"
                                        "}\n"
                                        "QPushButton::focus{outline: none;}\n"
                                        "QPushButton::hover {\n"
                                        "border-style: solid;\n"
                                        "border-width: 0px;\n"
                                        "border-radius: 0px;\n"
                                        "background-color: rgba(223, 223, 223, 150);}")

        self.stopButton = QtWidgets.QPushButton("生成图像")
        self.stopButton.setMinimumSize(QtCore.QSize(100, 60))
        self.stopButton.setStyleSheet("QPushButton {\n"
                                      "border-style: solid;\n"
                                      "border-width: 0px;\n"
                                      "border-radius: 0px;\n"
                                      "background-color: rgba(223, 223, 223, 100);\n"
                                      "}\n"
                                      "QPushButton::focus{outline: none;}\n"
                                      "QPushButton::hover {\n"
                                      "border-style: solid;\n"
                                      "border-width: 0px;\n"
                                      "border-radius: 0px;\n"
                                      "background-color: rgba(223, 223, 223, 150);}")

        self.horizontalLayout_12.addWidget(self.stopButton)

        self.runButton = QtWidgets.QPushButton(self.groupBox_201)

        self.runButton.setMinimumSize(QtCore.QSize(100, 60))
        self.runButton.setStyleSheet("QPushButton {\n"
                                     "border-style: solid;\n"
                                     "border-width: 0px;\n"
                                     "border-radius: 0px;\n"
                                     "background-color: rgba(223, 223, 223, 100);\n"
                                     "}\n"
                                     "QPushButton::focus{outline: none;}\n"
                                     "QPushButton::hover {\n"
                                     "border-style: solid;\n"
                                     "border-width: 0px;\n"
                                     "border-radius: 0px;\n"
                                     "background-color: rgba(223, 223, 223, 150);}")
        self.runButton.setText("目标检测")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/img/icon/运行.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)

        self.runButton.setIcon(icon7)
        self.runButton.setIconSize(QtCore.QSize(30, 30))
        self.runButton.setCheckable(True)
        self.runButton.setObjectName("runButton")
        self.horizontalLayout_12.addWidget(self.runButton)

        self.verticalLayout_4.addLayout(self.horizontalLayout_12)
        self.verticalLayout_4.setStretch(1, 1)
        self.horizontalLayout_7.addWidget(self.groupBox_201)
        self.verticalLayout_6.addLayout(self.horizontalLayout_7)

        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_18)
        self.groupBox_4.setMinimumSize(QtCore.QSize(0, 30))
        self.groupBox_4.setMaximumSize(QtCore.QSize(16777215, 30))
        self.groupBox_4.setStyleSheet("#groupBox_4{\n"
                                      "background-color: rgba(75, 75, 75, 200);\n"
                                      "border: 0px solid #42adff;\n"
                                      "border-left: 0px solid rgba(29, 83, 185, 255);\n"
                                      "border-right: 0px solid rgba(29, 83, 185, 255);\n"
                                      "border-top: 1px solid rgba(200, 200, 200,100);\n"
                                      "border-radius:0px;}")
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_10.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_10.setSpacing(0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.statistic_label = QtWidgets.QLabel(self.groupBox_4)
        self.statistic_label.setMouseTracking(False)
        self.statistic_label.setStyleSheet("QLabel\n"
                                           "{\n"
                                           "    font-size: 16px;\n"
                                           "    font-family: \"Microsoft YaHei\";\n"
                                           "    font-weight: light;\n"
                                           "         border-radius:9px;\n"
                                           "        background:rgba(66, 195, 255, 0);\n"
                                           "color: rgb(218, 218, 218);\n"
                                           "}\n"
                                           "")
        self.statistic_label.setText("")
        self.statistic_label.setObjectName("statistic_label")
        self.horizontalLayout_10.addWidget(self.statistic_label)
        self.verticalLayout_6.addWidget(self.groupBox_4)
        self.verticalLayout_2.addWidget(self.groupBox_18)
        mainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "基于YOLOv5与DallE-2的图像自动生成并识别"))
        self.label_4.setText(_translate("mainWindow", "YOLOv5-DallE-2 GUI"))
        self.label_5.setText(_translate("mainWindow", "setting"))
        self.label_3.setText(_translate("mainWindow", "model"))
        self.comboBox.setItemText(0, _translate("mainWindow", "yolov5s.pt"))
        self.comboBox.setItemText(1, _translate("mainWindow", "yolov5m.pt"))
        self.comboBox.setItemText(2, _translate("mainWindow", "yolov5l.pt"))
        self.comboBox.setItemText(3, _translate("mainWindow", "yolov5x.pt"))
        self.label_10.setText(_translate("mainWindow", "选项"))
        self.fileButton.setToolTip(_translate("mainWindow", "file"))
        self.cameraButton.setToolTip(_translate("mainWindow", "camera"))
        self.rtspButton.setToolTip(_translate("mainWindow", "rtsp"))
        self.label_2.setText(_translate("mainWindow", "IoU"))
        self.label.setText(_translate("mainWindow", "conf"))
        self.label_8.setText(_translate("mainWindow", "latency"))
        self.checkBox.setText(_translate("mainWindow", "enable"))
        self.saveCheckBox.setText(_translate("mainWindow", "save automatically"))
        self.label_11.setText(_translate("mainWindow", "result statistics"))
        self.label_6.setText(_translate("mainWindow", "view"))


from MouseLabel import Label_click_Mouse
import apprcc_rc
