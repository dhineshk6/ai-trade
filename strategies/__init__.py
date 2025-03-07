"""
Trading Strategies Package
------------------------
Created: 2025-03-07 03:51:58
Author: dhineshk6
"""

from .trading_strategy import TradingStrategy
from .risk_manager import RiskManager
from .backtest_engine import BacktestEngine

__all__ = [
    'TradingStrategy',
    'RiskManager',
    'BacktestEngine'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'dhineshk6'
__timestamp__ = '2025-03-07 03:51:58'
