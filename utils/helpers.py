from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import json
import pandas as pd
import numpy as np
import logging
import hashlib
import hmac
import base64
import time
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class TradingUtils:
    def __init__(self):
        self.current_time = datetime(2025, 3, 7, 0, 6, 15)
        self.current_user = 'dhineshk6'
    
    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade: float,
        stop_loss_pct: float
    ) -> float:
        """Calculate position size based on risk management parameters"""
        try:
            max_loss_amount = account_balance * risk_per_trade
            position_size = max_loss_amount / stop_loss_pct
            return round(position_size, 8)
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> float:
        """Calculate risk/reward ratio for a trade"""
        try:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            return round(reward / risk, 2) if risk != 0 else 0
        except Exception as e:
            logger.error(f"Error calculating R/R ratio: {e}")
            return 0.0

class TimeUtils:
    def __init__(self):
        self.current_time = datetime(2025, 3, 7, 0, 6, 15)
        self.current_user = 'dhineshk6'
    
    def get_timestamp_ms(self) -> int:
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)
    
    def convert_to_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """Convert OHLCV data to different timeframe"""
        try:
            rules = {
                '1m': '1min', '3m': '3min', '5m': '5min',
                '15m': '15min', '30m': '30min', '1h': '1H',
                '2h': '2H', '4h': '4H', '6h': '6H',
                '12h': '12H', '1d': 'D', '1w': 'W'
            }
            
            if timeframe not in rules:
                raise ValueError(f"Invalid timeframe: {timeframe}")
            
            resampled = df.resample(rules[timeframe], on='timestamp').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            return resampled.dropna()
            
        except Exception as e:
            logger.error(f"Error converting timeframe: {e}")
            return pd.DataFrame()
    
    def get_time_windows(
        self,
        start_time: datetime,
        end_time: datetime,
        window_size: str
    ) -> List[tuple]:
        """Split time range into windows"""
        try:
            windows = []
            current = start_time
            
            while current < end_time:
                window_end = min(
                    current + pd.Timedelta(window_size),
                    end_time
                )
                windows.append((current, window_end))
                current = window_end
            
            return windows
            
        except Exception as e:
            logger.error(f"Error creating time windows: {e}")
            return []

class SecurityUtils:
    def __init__(self):
        self.current_time = datetime(2025, 3, 7, 0, 6, 15)
        self.current_user = 'dhineshk6'
    
    def generate_signature(
        self,
        secret: str,
        message: str,
        algorithm: str = 'sha256'
    ) -> str:
        """Generate HMAC signature"""
        try:
            message_bytes = message.encode('utf-8')
            secret_bytes = secret.encode('utf-8')
            
            if algorithm == 'sha256':
                signature = hmac.new(
                    secret_bytes,
                    message_bytes,
                    hashlib.sha256
                ).hexdigest()
            elif algorithm == 'sha512':
                signature = hmac.new(
                    secret_bytes,
                    message_bytes,
                    hashlib.sha512
                ).hexdigest()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            return signature
            
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            return ""
    
    def encrypt_credentials(self, data: Dict) -> str:
        """Basic encryption for credentials"""
        try:
            json_str = json.dumps(data)
            return base64.b64encode(json_str.encode()).decode()
        except Exception as e:
            logger.error(f"Error encrypting credentials: {e}")
            return ""
    
    def decrypt_credentials(self, encrypted_data: str) -> Dict:
        """Basic decryption for credentials"""
        try:
            json_str = base64.b64decode(encrypted_data).decode()
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error decrypting credentials: {e}")
            return {}

class DataUtils:
    def __init__(self):
        self.current_time = datetime(2025, 3, 7, 0, 6, 15)
        self.current_user = 'dhineshk6'
    
    def validate_ohlcv(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data"""
        try:
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Check columns
            if not all(col in df.columns for col in required_columns):
                return False
            
            # Check data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(np.issubdtype(df[col].dtype, np.number) for col in numeric_cols):
                return False
            
            # Validate OHLC relationships
            valid_ohlc = (
                (df['high'] >= df['low']).all() and
                (df['high'] >= df['open']).all() and
                (df['high'] >= df['close']).all() and
                (df['low'] <= df['open']).all() and
                (df['low'] <= df['close']).all()
            )
            
            return valid_ohlc
            
        except Exception as e:
            logger.error(f"Error validating OHLCV data: {e}")
            return False
    
    def clean_dataframe(
        self,
        df: pd.DataFrame,
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """Clean and prepare DataFrame"""
        try:
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Handle missing values
            if fill_method == 'ffill':
                df = df.fillna(method='ffill')
            elif fill_method == 'bfill':
                df = df.fillna(method='bfill')
            elif fill_method == 'interpolate':
                df = df.interpolate(method='linear')
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning DataFrame: {e}")
            return pd.DataFrame()

class FileUtils:
    def __init__(self):
        self.current_time = datetime(2025, 3, 7, 0, 6, 15)
        self.current_user = 'dhineshk6'
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        directory: str = 'data'
    ) -> bool:
        """Save DataFrame to CSV"""
        try:
            # Create directory if it doesn't exist
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Full path
            filepath = Path(directory) / filename
            
            # Save file
            df.to_csv(filepath, index=True)
            logger.info(f"Successfully saved data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            return False
    
    def load_from_csv(
        self,
        filename: str,
        directory: str = 'data'
    ) -> pd.DataFrame:
        """Load DataFrame from CSV"""
        try:
            filepath = Path(directory) / filename
            df = pd.read_csv(filepath, index_col=0)
            return df
            
        except Exception as e:
            logger.error(f"Error loading from CSV: {e}")
            return pd.DataFrame()

# Initialize utility instances
trading_utils = TradingUtils()
time_utils = TimeUtils()
security_utils = SecurityUtils()
data_utils = DataUtils()
file_utils = FileUtils()