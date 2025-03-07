from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class StrategyBase(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self):
        self.current_time = datetime(2025, 3, 7, 0, 44, 24)
        self.current_user = 'dhineshk6'
        self.name = self.__class__.__name__
        self.description = "Base trading strategy class"
        self.version = "1.0.0"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, positions: List = None) -> Dict:
        """
        Generate trading signals from market data
        
        Args:
            data (pd.DataFrame): Market data with OHLCV
            positions (List): Current open positions
            
        Returns:
            Dict: Signal dictionary with action, direction, etc.
        """
        pass

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            data (pd.DataFrame): Market data with OHLCV
            
        Returns:
            pd.DataFrame: Data with added indicators
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data requirements
        
        Args:
            data (pd.DataFrame): Market data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Check required columns
            if not all(col in data.columns for col in required_columns):
                logger.error(f"Missing required columns. Need: {required_columns}")
                return False
            
            # Check for empty data
            if data.empty:
                logger.error("Empty dataset provided")
                return False
            
            # Check for NaN values
            if data[required_columns].isna().any().any():
                logger.error("Dataset contains NaN values")
                return False
            
            # Check correct ordering
            if not all(data[col].is_monotonic_increasing for col in ['high', 'low']):
                logger.error("Invalid OHLC relationships found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False

    def get_strategy_info(self) -> Dict:
        """
        Get strategy information and metadata
        
        Returns:
            Dict: Strategy information
        """
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'timestamp': self.current_time,
            'user': self.current_user
        }