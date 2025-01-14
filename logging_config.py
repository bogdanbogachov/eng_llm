import logging


def setup_logger(name="text_tuning"):
    # Create a logger with the given name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a console handler and set the level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a file handler and set the level to debug
    fh = logging.FileHandler('log_file.log')
    fh.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

logger = setup_logger()
