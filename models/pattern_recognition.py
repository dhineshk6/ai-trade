import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from scipy.signal import argrelextrema

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Pattern:
    type: str
    start_idx: int
    end_idx: int
    confidence: float
    direction: str  # 'bullish' or 'bearish'
    support_level: float = None
    resistance_level: float = None
    target_price: float = None
    stop_loss: float = None

class PatternRecognition:
    def __init__(self):
        """Initialize pattern recognition system"""
        self.current_time = datetime(2025, 3, 6, 23, 52, 14)
        self.current_user = 'dhineshk6'
        self.patterns_config = {
            'head_and_shoulders': {
                'min_points': 5,
                'symmetry_threshold': 0.15,
                'height_threshold': 0.02
            },
            'double_top_bottom': {
                'min_points': 3,
                'level_threshold': 0.02,
                'time_threshold': 20
            },
            'triangle': {
                'min_points': 5,
                'min_touches': 3,
                'slope_threshold': 0.02
            },
            'channel': {
                'min_points': 4,
                'parallel_threshold': 0.02,
                'min_touches': 2
            }
        }
    
    def identify_all_patterns(self, df: pd.DataFrame) -> Dict[str, List[Pattern]]:
        """Identify all chart patterns in the data"""
        try:
            patterns = {
                'head_and_shoulders': self._find_head_and_shoulders(df),
                'inverse_head_and_shoulders': self._find_inverse_head_and_shoulders(df),
                'double_top': self._find_double_top(df),
                'double_bottom': self._find_double_bottom(df),
                'ascending_triangle': self._find_ascending_triangle(df),
                'descending_triangle': self._find_descending_triangle(df),
                'symmetrical_triangle': self._find_symmetrical_triangle(df),
                'channel': self._find_channel(df)
            }
            
            # Log pattern detection results
            for pattern_type, pattern_list in patterns.items():
                if pattern_list:
                    logger.info(f"Found {len(pattern_list)} {pattern_type} patterns")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern identification: {e}")
            return {}

    def _find_peaks_and_troughs(self, df: pd.DataFrame, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks and troughs using scipy's argrelextrema"""
        try:
            peaks = argrelextrema(df['high'].values, np.greater, order=order)[0]
            troughs = argrelextrema(df['low'].values, np.less, order=order)[0]
            return peaks, troughs
        except Exception as e:
            logger.error(f"Error finding peaks and troughs: {e}")
            return np.array([]), np.array([])

    def _find_head_and_shoulders(self, df: pd.DataFrame, window: int = 20) -> List[Pattern]:
        """Identify head and shoulders patterns"""
        patterns = []
        try:
            peaks, _ = self._find_peaks_and_troughs(df)
            
            for i in range(len(peaks)-2):
                # Get three consecutive peaks
                p1, p2, p3 = peaks[i:i+3]
                
                # Check if middle peak is higher (head)
                if (df['high'].iloc[p2] > df['high'].iloc[p1] and 
                    df['high'].iloc[p2] > df['high'].iloc[p3]):
                    
                    # Check if shoulders are at similar levels
                    shoulder_diff = abs(df['high'].iloc[p1] - df['high'].iloc[p3])
                    if shoulder_diff <= self.patterns_config['head_and_shoulders']['height_threshold'] * df['high'].iloc[p2]:
                        
                        # Calculate neckline
                        neckline = min(df['low'].iloc[p1:p3+1])
                        
                        # Calculate pattern metrics
                        height = df['high'].iloc[p2] - neckline
                        target = neckline - height  # Projected target
                        
                        patterns.append(Pattern(
                            type='head_and_shoulders',
                            start_idx=p1,
                            end_idx=p3,
                            confidence=self._calculate_pattern_confidence(df, p1, p3),
                            direction='bearish',
                            support_level=neckline,
                            resistance_level=df['high'].iloc[p2],
                            target_price=target,
                            stop_loss=df['high'].iloc[p2]
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in head and shoulders detection: {e}")
            return []

    def _find_inverse_head_and_shoulders(self, df: pd.DataFrame) -> List[Pattern]:
        """Identify inverse head and shoulders patterns"""
        patterns = []
        try:
            _, troughs = self._find_peaks_and_troughs(df)
            
            for i in range(len(troughs)-2):
                # Get three consecutive troughs
                t1, t2, t3 = troughs[i:i+3]
                
                # Check if middle trough is lower (inverse head)
                if (df['low'].iloc[t2] < df['low'].iloc[t1] and 
                    df['low'].iloc[t2] < df['low'].iloc[t3]):
                    
                    # Check if shoulders are at similar levels
                    shoulder_diff = abs(df['low'].iloc[t1] - df['low'].iloc[t3])
                    if shoulder_diff <= self.patterns_config['head_and_shoulders']['height_threshold'] * df['low'].iloc[t2]:
                        
                        # Calculate neckline
                        neckline = max(df['high'].iloc[t1:t3+1])
                        
                        # Calculate pattern metrics
                        height = neckline - df['low'].iloc[t2]
                        target = neckline + height  # Projected target
                        
                        patterns.append(Pattern(
                            type='inverse_head_and_shoulders',
                            start_idx=t1,
                            end_idx=t3,
                            confidence=self._calculate_pattern_confidence(df, t1, t3),
                            direction='bullish',
                            support_level=df['low'].iloc[t2],
                            resistance_level=neckline,
                            target_price=target,
                            stop_loss=df['low'].iloc[t2]
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in inverse head and shoulders detection: {e}")
            return []

    def _find_double_top(self, df: pd.DataFrame) -> List[Pattern]:
        """Identify double top patterns"""
        patterns = []
        try:
            peaks, _ = self._find_peaks_and_troughs(df)
            
            for i in range(len(peaks)-1):
                p1, p2 = peaks[i:i+2]
                
                # Check if peaks are at similar levels
                price_diff = abs(df['high'].iloc[p1] - df['high'].iloc[p2])
                if price_diff <= self.patterns_config['double_top_bottom']['level_threshold'] * df['high'].iloc[p1]:
                    
                    # Check time spacing
                    time_diff = p2 - p1
                    if time_diff <= self.patterns_config['double_top_bottom']['time_threshold']:
                        
                        # Calculate neckline (support level)
                        neckline = min(df['low'].iloc[p1:p2+1])
                        
                        # Calculate pattern metrics
                        height = df['high'].iloc[p1] - neckline
                        target = neckline - height  # Projected target
                        
                        patterns.append(Pattern(
                            type='double_top',
                            start_idx=p1,
                            end_idx=p2,
                            confidence=self._calculate_pattern_confidence(df, p1, p2),
                            direction='bearish',
                            support_level=neckline,
                            resistance_level=df['high'].iloc[p1],
                            target_price=target,
                            stop_loss=max(df['high'].iloc[p1], df['high'].iloc[p2])
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in double top detection: {e}")
            return []

    def _find_double_bottom(self, df: pd.DataFrame) -> List[Pattern]:
        """Identify double bottom patterns"""
        patterns = []
        try:
            _, troughs = self._find_peaks_and_troughs(df)
            
            for i in range(len(troughs)-1):
                t1, t2 = troughs[i:i+2]
                
                # Check if troughs are at similar levels
                price_diff = abs(df['low'].iloc[t1] - df['low'].iloc[t2])
                if price_diff <= self.patterns_config['double_top_bottom']['level_threshold'] * df['low'].iloc[t1]:
                    
                    # Check time spacing
                    time_diff = t2 - t1
                    if time_diff <= self.patterns_config['double_top_bottom']['time_threshold']:
                        
                        # Calculate neckline (resistance level)
                        neckline = max(df['high'].iloc[t1:t2+1])
                        
                        # Calculate pattern metrics
                        height = neckline - df['low'].iloc[t1]
                        target = neckline + height  # Projected target
                        
                        patterns.append(Pattern(
                            type='double_bottom',
                            start_idx=t1,
                            end_idx=t2,
                            confidence=self._calculate_pattern_confidence(df, t1, t2),
                            direction='bullish',
                            support_level=df['low'].iloc[t1],
                            resistance_level=neckline,
                            target_price=target,
                            stop_loss=min(df['low'].iloc[t1], df['low'].iloc[t2])
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in double bottom detection: {e}")
            return []

    def _find_ascending_triangle(self, df: pd.DataFrame) -> List[Pattern]:
        """Identify ascending triangle patterns"""
        patterns = []
        try:
            peaks, troughs = self._find_peaks_and_troughs(df)
            
            for i in range(len(peaks)-1):
                start_idx = peaks[i]
                end_idx = peaks[i+1]
                
                # Get data segment
                segment = df.iloc[start_idx:end_idx+1]
                
                # Check for horizontal resistance
                resistance = segment['high'].mean()
                resistance_dev = segment['high'].std() / resistance
                
                # Check for rising support
                troughs_in_segment = [t for t in troughs if start_idx <= t <= end_idx]
                if len(troughs_in_segment) >= 2:
                    support_slope = np.polyfit(troughs_in_segment, 
                                            segment['low'].iloc[troughs_in_segment].values, 1)[0]
                    
                    if (resistance_dev < self.patterns_config['triangle']['slope_threshold'] and 
                        support_slope > 0):
                        
                        # Calculate pattern metrics
                        height = resistance - segment['low'].iloc[0]
                        target = resistance + height  # Projected target
                        
                        patterns.append(Pattern(
                            type='ascending_triangle',
                            start_idx=start_idx,
                            end_idx=end_idx,
                            confidence=self._calculate_pattern_confidence(df, start_idx, end_idx),
                            direction='bullish',
                            support_level=segment['low'].iloc[0],
                            resistance_level=resistance,
                            target_price=target,
                            stop_loss=segment['low'].iloc[0]
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in ascending triangle detection: {e}")
            return []

    def _calculate_pattern_confidence(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> float:
        """Calculate confidence score for a pattern"""
        try:
            # Calculate various metrics for confidence
            price_range = df['high'].iloc[start_idx:end_idx+1].max() - df['low'].iloc[start_idx:end_idx+1].min()
            volume_trend = df['volume'].iloc[start_idx:end_idx+1].mean() > df['volume'].iloc[:start_idx].mean()
            price_trend_before = df['close'].iloc[start_idx] > df['close'].iloc[start_idx-10:start_idx].mean()
            
            # Combine metrics into confidence score
            confidence = 0.0
            confidence += 0.3 if price_range > df['atr'].iloc[end_idx] * 2 else 0  # Significant price range
            confidence += 0.3 if volume_trend else 0  # Increasing volume
            confidence += 0.2 if price_trend_before else 0  # Prior trend
            confidence += 0.2  # Base confidence
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating pattern confidence: {e}")
            return 0.5

    def get_recent_patterns(self, df: pd.DataFrame, lookback: int = 100) -> Dict[str, List[Pattern]]:
        """Get patterns from recent data"""
        try:
            recent_df = df.iloc[-lookback:]
            patterns = self.identify_all_patterns(recent_df)
            
            # Filter out old patterns
            recent_patterns = {}
            for pattern_type, pattern_list in patterns.items():
                recent_patterns[pattern_type] = [
                    p for p in pattern_list 
                    if p.end_idx >= len(recent_df) - 10
                ]
            
            return recent_patterns
            
        except Exception as e:
            logger.error(f"Error getting recent patterns: {e}")
            return {}

    def get_pattern_stats(self) -> Dict:
        """Get pattern recognition statistics"""
        return {
            'timestamp': self.current_time,
            'user': self.current_user,
            'config': self.patterns_config,
            'supported_patterns': [
                'head_and_shoulders',
                'inverse_head_and_shoulders',
                'double_top',
                'double_bottom',
                'ascending_triangle',
                'descending_triangle',
                'symmetrical_triangle',
                'channel'
            ]
        }