import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict

from config.settings import (
    LOOKBACK_PERIOD,
    BATCH_SIZE,
    EPOCHS,
    FEATURES,
    TRAIN_TEST_SPLIT
)

class AIModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.lookback_period = LOOKBACK_PERIOD
        self.current_time = datetime(2025, 3, 6, 20, 26, 57)
        self.current_user = 'dhineshk6'
        
    def prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for AI model"""
        try:
            # Select features
            data = df[FEATURES].copy()
            
            # Scale data
            scaled_data = self.scaler.fit_transform(data)
            
            X, y = [], []
            for i in range(self.lookback_period, len(scaled_data)):
                X.append(scaled_data[i-self.lookback_period:i])
                # Use price direction as target
                price_change = df['close'].iloc[i] > df['close'].iloc[i-1]
                y.append(1 if price_change else 0)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            return np.array([]), np.array([])
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """Build LSTM model"""
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(100, return_sequences=True, input_shape=input_shape),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(50, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(25, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
        except Exception as e:
            print(f"Error building model: {e}")
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train the AI model"""
        try:
            # Prepare data
            X, y = self.prepare_data(df)
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Failed to prepare training data")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-TRAIN_TEST_SPLIT, shuffle=False
            )
            
            # Build model if not exists
            if self.model is None:
                self.build_model((X.shape[1], X.shape[2]))
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate model
            evaluation = self.model.evaluate(X_test, y_test, verbose=0)
            
            return {
                'accuracy': evaluation[1],
                'loss': evaluation[0],
                'train_history': history.history
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            return {}
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        try:
            # Prepare prediction data
            X, _ = self.prepare_data(df.tail(self.lookback_period + 1))
            
            if len(X) == 0:
                raise ValueError("Failed to prepare prediction data")
            
            # Generate predictions
            predictions = self.model.predict(X, verbose=0)
            return predictions.flatten()
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return np.array([])
    
    def get_trading_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals from predictions"""
        try:
            predictions = self.predict(df)
            
            if len(predictions) == 0:
                return {
                    'signal': 0,
                    'confidence': 0,
                    'timestamp': self.current_time
                }
            
            latest_pred = predictions[-1]
            
            # Convert prediction to signal
            signal = 1 if latest_pred > 0.6 else (-1 if latest_pred < 0.4 else 0)
            
            return {
                'signal': signal,
                'confidence': float(abs(latest_pred - 0.5) * 2),
                'timestamp': self.current_time
            }
            
        except Exception as e:
            print(f"Error generating trading signals: {e}")
            return {
                'signal': 0,
                'confidence': 0,
                'timestamp': self.current_time
            }
    
    def save_model(self, path: str) -> bool:
        """Save model to file"""
        try:
            self.model.save(path)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load model from file"""
        try:
            self.model = tf.keras.models.load_model(path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False