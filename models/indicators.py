import pandas as pd
import numpy as np
import ta
from typing import Dict, List
from datetime import datetime

from config.settings import (
    RSI_PERIOD,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    EMA_SHORT,
    EMA_LONG,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BB_PERIOD,
    BB_STD,
    ATR_PERIOD,
    VOLUME_MA_PERIOD
)

class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataframe"""
        if df.empty:
            return df
            
        try:
            # Trend Indicators
            df = TechnicalIndicators._add_moving_averages(df)
            df = TechnicalIndicators._add_macd(df)
            df = TechnicalIndicators._add_adx(df)
            
            # Momentum Indicators
            df = TechnicalIndicators._add_rsi(df)
            df = TechnicalIndicators._add_stochastic(df)
            df = TechnicalIndicators._add_cci(df)
            
            # Volatility Indicators
            df = TechnicalIndicators._add_bollinger_bands(df)
            df = TechnicalIndicators._add_atr(df)
            df = TechnicalIndicators._add_keltner_channels(df)
            
            # Volume Indicators
            df = TechnicalIndicators._add_volume_indicators(df)
            
            # Custom Indicators
            df = TechnicalIndicators._add_custom_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"Error adding indicators: {e}")
            return df
    
    @staticmethod
    def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages"""
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=EMA_SHORT)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=EMA_LONG)
        return df
    
    @staticmethod
    def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator"""
        df['macd_line'] = ta.trend.macd(df['close'], 
                                       window_slow=MACD_SLOW, 
                                       window_fast=MACD_FAST)
        df['macd_signal'] = ta.trend.macd_signal(df['close'], 
                                                window_slow=MACD_SLOW,
                                                window_fast=MACD_FAST, 
                                                window_sign=MACD_SIGNAL)
        df['macd_diff'] = ta.trend.macd_diff(df['close'],
                                            window_slow=MACD_SLOW,
                                            window_fast=MACD_FAST,
                                            window_sign=MACD_SIGNAL)
        return df
    
    @staticmethod
    def _add_rsi(df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicator"""
        df['rsi'] = ta.momentum.rsi(df['close'], window=RSI_PERIOD)
        df['rsi_signal'] = np.where(df['rsi'] > RSI_OVERBOUGHT, -1,
                                   np.where(df['rsi'] < RSI_OVERSOLD, 1, 0))
        return df
    
    @staticmethod
    def _add_stochastic(df: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        return df
    
    @staticmethod
    def _add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands"""
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], 
                                                      window=BB_PERIOD,
                                                      window_dev=BB_STD)
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'],
                                                      window=BB_PERIOD)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'],
                                                      window=BB_PERIOD,
                                                      window_dev=BB_STD)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        return df
    
    @staticmethod
    def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range"""
        df['atr'] = ta.volatility.average_true_range(df['high'],
                                                    df['low'],
                                                    df['close'],
                                                    window=ATR_PERIOD)
        df['atr_ratio'] = df['atr'] / df['close']
        return df
    
    @staticmethod
    def _add_adx(df: pd.DataFrame) -> pd.DataFrame:
        """Add Average Directional Index"""
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['di_plus'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
        df['di_minus'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])
        return df
    
    @staticmethod
    def _add_cci(df: pd.DataFrame) -> pd.DataFrame:
        """Add Commodity Channel Index"""
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        return df
    
    @staticmethod
    def _add_keltner_channels(df: pd.DataFrame) -> pd.DataFrame:
        """Add Keltner Channels"""
        df['kc_upper'] = ta.volatility.keltner_channel_hband(df['high'],
                                                            df['low'],
                                                            df['close'])
        df['kc_lower'] = ta.volatility.keltner_channel_lband(df['high'],
                                                            df['low'],
                                                            df['close'])
        return df
    
    @staticmethod
    def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume-based indicators"""
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['mfi'] = ta.volume.money_flow_index(df['high'],
                                              df['low'],
                                              df['close'],
                                              df['volume'])
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=VOLUME_MA_PERIOD)
        return df
    
    @staticmethod
    def _add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom indicators"""
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(periods=5)
        
        # Volatility ratio
        df['volatility_ratio'] = df['atr'] / df['close'].rolling(window=20).std()
        
        # Custom RSI divergence
        df['rsi_slope'] = ta.momentum.rsi(df['close']).diff(periods=5)
        df['price_slope'] = df['close'].diff(periods=5)
        df['rsi_divergence'] = np.where(
            (df['rsi_slope'] > 0) & (df['price_slope'] < 0), 1,
            np.where((df['rsi_slope'] < 0) & (df['price_slope'] > 0), -1, 0)
        )
        
        return df
    
    @staticmethod
    def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on indicators"""
        signals = pd.DataFrame(index=df.index)
        
        # RSI signals
        signals['rsi_signal'] = np.where(df['rsi'] > RSI_OVERBOUGHT, -1,
                                       np.where(df['rsi'] < RSI_OVERSOLD, 1, 0))
        
        # MACD signals
        signals['macd_signal'] = np.where(df['macd_diff'] > 0, 1,
                                        np.where(df['macd_diff'] < 0, -1, 0))
        
        # Bollinger Bands signals
        signals['bb_signal'] = np.where(df['close'] < df['bb_lower'], 1,
                                      np.where(df['close'] > df['bb_upper'], -1, 0))
        
        # Volume signals
        signals['volume_signal'] = np.where(df['volume'] > df['volume_sma'], 1, -1)
        
        # Trend signals
        signals['trend_signal'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
        
        # Combined signal
        signals['combined_signal'] = (
            signals['rsi_signal'] * 0.2 +
            signals['macd_signal'] * 0.3 +
            signals['bb_signal'] * 0.2 +
            signals['volume_signal'] * 0.1 +
            signals['trend_signal'] * 0.2
        )
        
        return signals

    @staticmethod
    def get_current_signals(df: pd.DataFrame) -> Dict:
        """Get current trading signals"""
        current = df.iloc[-1]
        
        signals = {
            'timestamp': datetime(2025, 3, 4, 20, 6, 2),  # Current time
            'rsi': current['rsi'],
            'macd': current['macd_diff'],
            'bb_position': (current['close'] - current['bb_middle']) / 
                         (current['bb_upper'] - current['bb_lower']),
            'trend': 'uptrend' if current['ema_9'] > current['ema_21'] else 'downtrend',
            'volatility': current['atr_ratio'],
            'volume_trend': 'high' if current['volume'] > current['volume_sma'] else 'low'
        }
        
        return signals