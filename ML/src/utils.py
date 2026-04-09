import os
import logging
import colorlog
from logging.handlers import RotatingFileHandler
from config.settings import LOG_FILE

def _setup_logger():
    """Sets up a logger that logs to both console (with colors)"""
    
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    logger = logging.getLogger("VVC_Pipeline")
    logger.setLevel(logging.DEBUG) 
    
    if logger.handlers:
        return logger

    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=5*1024*1024, # 5MB
        backupCount=3, 
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    log_colors = {
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
        'STAGE':    'purple',
    }
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors=log_colors
    )

    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    STAGE_LEVEL = 25
    logging.addLevelName(STAGE_LEVEL, "STAGE")
    def stage(self, message, *args, **kwargs):
        if self.isEnabledFor(STAGE_LEVEL):
            self._log(STAGE_LEVEL, message, args, **kwargs)
    logging.Logger.stage = stage
    return logger

_logger = _setup_logger()


def log_message(message: str, level: str = 'info'):
    """
    Logs a message to console (colored) and file (clean).
    
    Args:
        message (str): essage
        level (str): 'info', 'warning', 'error', 'debug'.
    """
    lvl = level.lower()
    if lvl == 'info':
        _logger.info(message)
    elif lvl == 'warning':
        _logger.warning(message)
    elif lvl == 'error':
        _logger.error(message)
    elif lvl == 'debug':
        _logger.debug(message)
    elif lvl == 'critical':
        _logger.critical(message)
    elif lvl == 'stage':
        _logger.stage(message)
    else:
        _logger.info(message)