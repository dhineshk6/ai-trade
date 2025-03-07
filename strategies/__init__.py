"""
Trading Strategies Package
------------------------
Created: 2025-03-07 04:22:40
Author: dhineshk6
"""

import os
import sys

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from .trading_strategy import TradingStrategy
    from .risk_manager import RiskManager
    from .backtest_engine import BacktestEngine
    from .strategy_base import StrategyBase
except ImportError as e:
    print(f"Import error in __init__.py: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Python path: {sys.path}")
    raise

__all__ = [
    'StrategyBase',
    'TradingStrategy',
    'RiskManager',
    'BacktestEngine'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'dhineshk6'
__timestamp__ = '2025-03-07 04:22:40'
