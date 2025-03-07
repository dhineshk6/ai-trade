from .strategy_base import StrategyBase
from .trading_strategy import TradingStrategy
from .risk_manager import RiskManager
from .backtest_engine import BacktestEngine

__all__ = [
    'StrategyBase',
    'TradingStrategy',
    'RiskManager',
    'BacktestEngine'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'dhineshk6'
__updated__ = '2025-03-07 00:44:24'

# Strategy configuration
STRATEGY_CONFIG = {
    'default_timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
    'max_positions': 5,
    'risk_per_trade': 0.02,
    'max_drawdown': 0.10,
    'position_sizing': {
        'method': 'risk_based',
        'max_position_size': 0.1
    },
    'indicators': {
        'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'ema': {'short': 9, 'long': 21},
        'bb': {'period': 20, 'std_dev': 2}
    }
}

# Current session information
CURRENT_TIME = '2025-03-07 00:44:24'
CURRENT_USER = 'dhineshk6'