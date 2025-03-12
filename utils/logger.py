import logging


def setup_logger():
    logger = logging.getLogger('TradingBot')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('bot.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
