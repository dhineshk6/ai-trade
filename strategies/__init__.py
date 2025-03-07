"""
Trading Strategies Package
------------------------
Created: 2025-03-07 04:11:59
Author: dhineshk6
"""

# Import strategy components
from strategies.trading_strategy import TradingStrategy
from strategies.risk_manager import RiskManager
from strategies.backtest_engine import BacktestEngine
from strategies.strategy_base import StrategyBase

# Export components
__all__ = [
    'StrategyBase',
    'TradingStrategy',
    'RiskManager',
    'BacktestEngine'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'dhineshk6'
__timestamp__ = '2025-03-07 04:11:59'
