import logging
from logging.handlers import RotatingFileHandler

def get_handler(file):
    log_formatter = logging.Formatter('%(asctime)s (%(name)s) %(levelname)s: %(message)s', '%m/%d/%Y %H:%M:%S')
    handler = RotatingFileHandler(file, mode='a', maxBytes=1024*1024*10, backupCount=1)
    handler.setFormatter(log_formatter)
    return handler
