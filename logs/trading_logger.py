import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

class TradingLogger:
    def __init__(self):
        self.current_time = datetime(2025, 3, 6, 23, 57, 8)
        self.current_user = 'dhineshk6'
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Configure root logger
        logging.basicConfig(level=logging.INFO)
        root_logger = logging.getLogger()

        # Clear any existing handlers
        root_logger.handlers = []

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | User: %(user)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # Setup file handlers
        self._setup_trading_logs(detailed_formatter)
        self._setup_error_logs(detailed_formatter)
        self._setup_performance_logs(detailed_formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

    def _setup_trading_logs(self, formatter):
        """Setup trading activity logs"""
        trading_handler = RotatingFileHandler(
            'logs/trading.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        trading_handler.setFormatter(formatter)
        trading_handler.setLevel(logging.INFO)
        
        trading_logger = logging.getLogger('trading')
        trading_logger.addHandler(trading_handler)
        trading_logger.setLevel(logging.INFO)

    def _setup_error_logs(self, formatter):
        """Setup error logs"""
        error_handler = RotatingFileHandler(
            'logs/errors.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        
        error_logger = logging.getLogger('errors')
        error_logger.addHandler(error_handler)
        error_logger.setLevel(logging.ERROR)

    def _setup_performance_logs(self, formatter):
        """Setup performance monitoring logs"""
        perf_handler = RotatingFileHandler(
            'logs/performance.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        perf_handler.setFormatter(formatter)
        perf_handler.setLevel(logging.INFO)
        
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)

    def get_logger(self, name):
        """Get a logger with the specified name"""
        logger = logging.getLogger(name)
        logger = logging.LoggerAdapter(logger, {'user': self.current_user})
        return logger