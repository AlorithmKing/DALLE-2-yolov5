import logging
import os


def Logger():
    if not os.path.exists('Logger'):
        os.makedirs('Logger')
        # 添加日志记录器配置
    logging.basicConfig(filename='Logger/app.log', level=logging.DEBUG,
                                 format='%(asctime)s %(levelname)s: %(message)s')
    logging.debug("123")

