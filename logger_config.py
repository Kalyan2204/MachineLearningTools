import logging
import colorlog
       
LOG_COLOR={
    'INFO': 'bold_green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

def setup_logger():
    logger = logging.getLogger('ml_logger')
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('app.log')
    
    # Set level for handlers
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.INFO)
    
    # Create formatters and add them to handlers
    console_format = colorlog.ColoredFormatter("%(log_color)s[%(name)s] - %(levelname)s - %(message)s", LOG_COLOR)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize the logger
logger = setup_logger()
