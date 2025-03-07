from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import talib
import logging
from .strategy_base import StrategyBase

logger = logging.getLogger(__name__)

class TradingStrategy(StrategyBase):
    """Implementation of a combined technical analysis trading strategy"""
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: int = 70,
                 rsi_oversold: int = 30,
                 ema_short: int = 9,
                 ema_long: int = 21,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9):
        """
        Initialize trading strategy with parameters
        
        Args:
            rsi_period (int): RSI calculation period
            rsi_overbought (int): RSI overbought threshold
            rsi_oversold (int): RSI oversold threshold
            ema_short (int): Short EMA period
            ema_long (int): Long EMA period
            bb_period (int): Bollinger Bands period
            bb_std (float): Bollinger Bands standard deviation
            macd_fast (int): MACD fast period
            macd_slow (int): MACD slow period
            macd_signal (int): MACD signal period
        """
        super().__init__()
        
        # Strategy metadata
        self.description = "Combined technical analysis strategy using RSI, EMA, BB, and MACD"
        self.version = "1.0.0"
        
        # Strategy parameters
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            df = data.copy()
            
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
            
            # EMAs
            df['ema_short'] = talib.EMA(df['close'], timeperiod=self.ema_short)
            df['ema_long'] = talib.EMA(df['close'], timeperiod=self.ema_long)
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'],
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                df['close'],
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return data

    def generate_signals(self, data: pd.DataFrame, positions: List = None) -> Dict:
        """Generate trading signals"""
        try:
            if len(data) < self.ema_long:
                return {'action': 'hold'}
            
            # Validate data
            if not self.validate_data(data):
                return {'action': 'hold'}
            
            # Calculate indicators
            df = self.calculate_indicators(data)
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Initialize signal
            signal = {
                'action': 'hold',
                'direction': None,
                'strategy': self.name,
                'risk_per_trade': 0.02,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'timestamp': self.current_time,
                'user': self.current_user
            }
            
            # Check for long signals
            long_conditions = (
                current['rsi'] < self.rsi_oversold and
                current['close'] < current['bb_lower'] and
                current['ema_short'] > current['ema_long'] and
                current['macd_hist'] > 0 and
                previous['macd_hist'] <= 0
            )
            
            # Check for short signals
            short_conditions = (
                current['rsi'] > self.rsi_overbought and
                current['close'] > current['bb_upper'] and
                current['ema_short'] < current['ema_long'] and
                current['macd_hist'] < 0 and
                previous['macd_hist'] >= 0
            )
            
            # Generate signal based on conditions
            if long_conditions:
                signal.update({
                    'action': 'enter',
                    'direction': 'long',
                    'reason': 'RSI oversold + BB lower + EMA & MACD crossover',
                    'indicators': {
                        'rsi': current['rsi'],
                        'bb_lower': current['bb_lower'],
                        'macd_hist': current['macd_hist']
                    }
                })
            elif short_conditions:
                signal.update({
                    'action': 'enter',
                    'direction': 'short',
                    'reason': 'RSI overbought + BB upper + EMA & MACD crossover',
                    'indicators': {
                        'rsi': current['rsi'],
                        'bb_upper': current['bb_upper'],
                        'macd_hist': current['macd_hist']
                    }
                })
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {'action': 'hold'}

    def get_strategy_parameters(self) -> Dict:
        """Get strategy parameters"""
        return {
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'ema_short': self.ema_short,
            'ema_long': self.ema_long,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal
        }