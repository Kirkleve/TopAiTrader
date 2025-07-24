import logging
from pathlib import Path

def setup_logger(log_file='logs/training.log'):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('TradingBotLogger')
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
