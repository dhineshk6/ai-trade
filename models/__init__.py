from .indicators import TechnicalIndicators
from .pattern_recognition import PatternRecognition
from .ai_model import AIModel

__all__ = ['TechnicalIndicators', 'PatternRecognition', 'AIModel']

# Module metadata
__version__ = '1.0.0'
__author__ = 'dhineshk6'
__updated__ = '2025-03-07 00:00:20'

# Module configuration
MODULE_CONFIG = {
    'indicators': {
        'default_timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
        'default_indicators': ['RSI', 'MACD', 'BB', 'EMA']
    },
    'pattern_recognition': {
        'min_pattern_size': 5,
        'confidence_threshold': 0.7
    },
    'ai_model': {
        'model_version': '1.0.0',
        'training_frequency': '1d'
    }
}

# Current session information
CURRENT_TIME = '2025-03-07 00:00:20'
CURRENT_USER = 'dhineshk6'