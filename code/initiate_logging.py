import sys
import logging


def init_logging():
    """  Initiate logging and configure
     Returns:
         logger(class): Fully formatted and configured logging-class
     """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(fhandler)
    return logger
