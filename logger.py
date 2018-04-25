import logging
import logging.handlers
import os

curr_path = os.path.abspath(os.path.dirname(__file__))


def _init_logger(this_logger, file_path):
    this_logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        '[%(lineno)s] [%(asctime)s] [%(levelname)s] %(message)s',
        '%Y-%m-%d %H:%M:%S')
    # 设置CMD日志
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    # 时间轮转文件日志
    trfh = logging.handlers.TimedRotatingFileHandler(filename=file_path, when='D', interval=1, backupCount=30)
    trfh.setFormatter(fmt)
    this_logger.addHandler(sh)
    this_logger.addHandler(trfh)


logFilename = os.path.join(curr_path,'log/log')
logger = logging.getLogger('monitor')
_init_logger(logger, logFilename)
