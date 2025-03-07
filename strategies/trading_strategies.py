from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from models.indicators import TechnicalIndicators
from models.pattern_recognition import ChartPatternRecognition
from models.ai_model import AIModel

class TradingStrategy:
    def __init__(self):
        self.current_time = datetime(2025, 3, 6, 22, 57, 14)
        self.current_user = 'dhineshk6'
        self.indicators = TechnicalIndicators()
        self.pattern_recognition = ChartPatternRecognition()
        self.ai_model = AIModel()
        
    def generate_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals using multiple strategies"""
        try:
            # Get signals from different strategies
            technical_signals = self._get_technical_signals(df)
            pattern_signals = self._get_pattern_signals(df)
            ai_signals = self._get_ai_signals(df)
            
            # Combine signals with weights
            combined_signal = self._combine_signals(
                technical_signals,
                pattern_signals,
                ai_signals
            )
            
            return {
                'timestamp': self.current_time,
                'technical_signals': technical_signals,
                'pattern_signals': pattern_signals,
                'ai_signals': ai_signals,
                'combined_signal': combined_signal,
                'metadata': {
                    'user': self.current_user,
                    'strategy_version': '1.0.0'
                }
            }
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            return self._get_empty_signals()
    
    def _get_technical_signals(self, df: pd.DataFrame) -> Dict:
        """Generate signals based on technical indicators"""
        try:
            # Add technical indicators if not present
            if 'rsi' not in df.columns:
                df = self.indicators.add_all_indicators(df)
            
            current = df.iloc[-1]
            
            # RSI signals
            rsi_signal = 1 if current['rsi'] < 30 else (-1 if current['rsi'] > 70 else 0)
            
            # MACD signals
            macd_signal = 1 if current['macd_diff'] > 0 else (-1 if current['macd_diff'] < 0 else 0)
            
            # Bollinger Bands signals
            bb_signal = 1 if current['close'] < current['bb_lower'] else (
                -1 if current['close'] > current['bb_upper'] else 0
            )
            
            # Moving Average signals
            ma_signal = 1 if current['ema_9'] > current['ema_21'] else -1
            
            # Volume signals
            volume_signal = 1 if current['volume'] > current['volume_sma'] else -1
            
            # Combine technical signals
            technical_score = (
                rsi_signal * 0.2 +
                macd_signal * 0.3 +
                bb_signal * 0.2 +
                ma_signal * 0.2 +
                volume_signal * 0.1
            )
            
            return {
                'signal': technical_score,
                'components': {
                    'rsi': rsi_signal,
                    'macd': macd_signal,
                    'bollinger': bb_signal,
                    'moving_average': ma_signal,
                    'volume': volume_signal
                },
                'indicators': {
                    'rsi': current['rsi'],
                    'macd': current['macd_diff'],
                    'bb_width': current['bb_width'],
                    'atr': current['atr']
                }
            }
            
        except Exception as e:
            print(f"Error in technical signals: {e}")
            return {'signal': 0, 'components': {}, 'indicators': {}}
    
    def _get_pattern_signals(self, df: pd.DataFrame) -> Dict:
        """Generate signals based on chart patterns"""
        try:
            patterns = self.pattern_recognition.identify_patterns(df)
            
            # Analyze recent patterns
            recent_patterns = self._filter_recent_patterns(patterns)
            
            if not recent_patterns:
                return {'signal': 0, 'patterns': [], 'confidence': 0}
            
            # Calculate pattern-based signal
            pattern_signal = 0
            total_confidence = 0
            
            pattern_weights = {
                'head_and_shoulders': -0.8,
                'inverse_head_and_shoulders': 0.8,
                'double_top': -0.6,
                'double_bottom': 0.6,
                'bull_flag': 0.5,
                'bear_flag': -0.5,
                'ascending_triangle': 0.7,
                'descending_triangle': -0.7
            }
            
            for pattern in recent_patterns:
                weight = pattern_weights.get(pattern['type'], 0)
                confidence = pattern['confidence']
                pattern_signal += weight * confidence
                total_confidence += confidence
            
            if total_confidence > 0:
                pattern_signal /= total_confidence
            
            return {
                'signal': pattern_signal,
                'patterns': recent_patterns,
                'confidence': total_confidence
            }
            
        except Exception as e:
            print(f"Error in pattern signals: {e}")
            return {'signal': 0, 'patterns': [], 'confidence': 0}
    
    def _get_ai_signals(self, df: pd.DataFrame) -> Dict:
        """Generate signals using AI model"""
        try:
            ai_prediction = self.ai_model.get_trading_signals(df)
            
            return {
                'signal': ai_prediction['signal'],
                'confidence': ai_prediction['confidence'],
                'metadata': {
                    'model_version': '1.0.0',
                    'timestamp': self.current_time
                }
            }
            
        except Exception as e:
            print(f"Error in AI signals: {e}")
            return {'signal': 0, 'confidence': 0, 'metadata': {}}
    
    def _combine_signals(
        self,
        technical_signals: Dict,
        pattern_signals: Dict,
        ai_signals: Dict
    ) -> Dict:
        """Combine signals from different sources"""
        try:
            # Weights for different signal sources
            weights = {
                'technical': 0.4,
                'pattern': 0.3,
                'ai': 0.3
            }
            
            # Calculate weighted average
            combined_signal = (
                technical_signals['signal'] * weights['technical'] +
                pattern_signals['signal'] * weights['pattern'] +
                ai_signals['signal'] * weights['ai']
            )
            
            # Calculate confidence score
            confidence = (
                weights['technical'] +
                pattern_signals['confidence'] * weights['pattern'] +
                ai_signals['confidence'] * weights['ai']
            )
            
            # Determine signal strength
            signal_strength = abs(combined_signal)
            
            return {
                'signal': combined_signal,
                'direction': 'buy' if combined_signal > 0.2 else (
                    'sell' if combined_signal < -0.2 else 'neutral'
                ),
                'strength': signal_strength,
                'confidence': confidence,
                'timestamp': self.current_time
            }
            
        except Exception as e:
            print(f"Error combining signals: {e}")
            return self._get_empty_signals()
    
    def _filter_recent_patterns(self, patterns: Dict[str, List[Dict]]) -> List[Dict]:
        """Filter patterns to get only recent ones"""
        recent_patterns = []
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if pattern['end_idx'] >= len(pattern_list) - 5:  # Consider last 5 bars
                    pattern['type'] = pattern_type
                    recent_patterns.append(pattern)
        return recent_patterns
    
    def _get_empty_signals(self) -> Dict:
        """Return empty signals structure"""
        return {
            'signal': 0,
            'direction': 'neutral',
            'strength': 0,
            'confidence': 0,
            'timestamp': self.current_time
        }
    
    def get_strategy_info(self) -> Dict:
        """Get information about the trading strategy"""
        return {
            'name': 'Multi-Strategy Trading System',
            'version': '1.0.0',
            'components': {
                'technical_analysis': True,
                'pattern_recognition': True,
                'ai_model': True
            },
            'weights': {
                'technical': 0.4,
                'pattern': 0.3,
                'ai': 0.3
            },
            'metadata': {
                'last_updated': self.current_time,
                'user': self.current_user
            }
        }