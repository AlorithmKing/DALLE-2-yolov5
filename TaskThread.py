import imghdr
import os
import requests
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from GetImageURL import generate_image_url


class TaskThread(QThread):
    progress_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, prompt, filename, api_key, dir_path):
        QThread.__init__(self)
        self._stop = None
        self.prompt = prompt
        self.filename = filename
        self.api_key = api_key
        self.dir_path = dir_path
        self.mutex = QMutex()

    def run(self):
        try:
            image_url = generate_image_url(self.prompt, self.api_key)
            if image_url:
                file_content = self.download_image(image_url)
                if file_content:
                    file_extension = imghdr.what(None, h=file_content)
                    if file_extension:
                        file_name_with_extension = f"{self.filename}.{file_extension}"
                        self.save_file(file_content, file_name_with_extension)
        except Exception as e:
            self.error_signal.emit(str(e))

    def download_image(self, image_url):
        response = requests.get(image_url)
        if response.status_code == 200:
            return response.content
        else:
            print(f"Error downloading image: {response.status_code}")
            return None

    def save_file(self, content, file_name):
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        with open(f"{self.dir_path}/{file_name}", "wb") as f:
            f.write(content)
            print(f"File saved as {file_name}")

    def update_progress(self, value):
        with QMutexLocker(self.mutex):
            self.progress_signal.emit(value)

    def stop(self):
        with QMutexLocker(self.mutex):
            self._stop = True

    def stopped(self):
        with QMutexLocker(self.mutex):
            return self._stop
