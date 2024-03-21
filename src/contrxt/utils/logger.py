from typing import Text
import logging
import os

def build_logger(
    log_level: Text,
    log_name : Text,
    out_file : Text  = None
)-> logging.Logger:
    """Build logger instance.

    Parameters
    ----------
    log_level : Text
        Set logging level.
    logger_name : Text
        Name of the logger instance.
    out_file : Text
        Path for logger output.

    Returns
    -------
    out: logging.Logger
        Logger instance.

    """
    logger = logging.Logger(name = log_name)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(hdlr= stream_handler)

    if out_file:
        out_dir = os.path.dirname(p = out_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        file_hander = logging.FileHandler(out_file)
        file_hander.se(formatter)
        logger.addHandler(hdlr= file_hander)
    
    logger.setLevel(log_level)
    return logger

if __name__ == '__main__':
    pass