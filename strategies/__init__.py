from .trading_strategies import TradingStrategy
from .risk_management import RiskManager

__all__ = ['TradingStrategy', 'RiskManager']

# Module metadata
__version__ = '1.0.0'
__author__ = 'dhineshk6'
__updated__ = '2025-03-07 00:00:20'

# Strategy configuration
STRATEGY_CONFIG = {
    'trading': {
        'default_timeframe': '1h',
        'max_positions': 5,
        'risk_per_trade': 0.02
    },
    'risk_management': {
        'max_drawdown': 0.10,
        'max_risk_per_day': 0.05,
        'position_sizing_method': 'risk_based'
    }
}

# Current session information
CURRENT_TIME = '2025-03-07 00:00:20'
CURRENT_USER = 'dhineshk6'