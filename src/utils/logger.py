import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(name, log_level=logging.INFO, log_dir="logs"):
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Name of the logger
        log_level: Log level (default: INFO)
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is specified
    if log_dir:
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for the log file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        # Add rotating file handler (10MB max size, keep 5 backups)
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    
    return logger 