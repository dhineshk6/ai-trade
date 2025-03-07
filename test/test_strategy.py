import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from strategies.trading_strategy import TradingStrategy

class TestTradingStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.current_time = datetime(2025, 3, 7, 1, 5, 41)
        self.current_user = 'dhineshk6'
        
        self.strategy = TradingStrategy()
        self.data = self._create_sample_data()

    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample market data for testing"""
        dates = pd.date_range(
            start='2025-01-01',
            end='2025-03-07',
            freq='1H'
        )
        
        data = pd.DataFrame(index=dates)
        data['open'] = np.random.normal(100, 2, len(dates))
        data['high'] = data['open'] + abs(np.random.normal(0, 0.5, len(dates)))
        data['low'] = data['open'] - abs(np.random.normal(0, 0.5, len(dates)))
        data['close'] = np.random.normal(100, 2, len(dates))
        data['volume'] = np.random.normal(1000, 100, len(dates))
        
        return data

    def test_indicator_calculation(self):
        """Test technical indicator calculation"""
        df = self.strategy.calculate_indicators(self.data)
        
        # Test indicator presence
        required_indicators = [
            'rsi',
            'ema_short',
            'ema_long',
            'macd',
            'macd_signal',
            'macd_hist',
            'bb_upper',
            'bb_middle',
            'bb_lower'
        ]
        
        for indicator in required_indicators:
            self.assertIn(indicator, df.columns)
            self.assertTrue(df[indicator].notnull().any())

    def test_signal_generation(self):
        """Test trading signal generation"""
        signals = self.strategy.generate_signals(self.data)
        
        # Test signal structure
        required_fields = [
            'action',
            'direction',
            'strategy',
            'risk_per_trade',
            'stop_loss_pct',
            'take_profit_pct'
        ]
        
        for field in required_fields:
            self.assertIn(field, signals)

    def test_parameter_validation(self):
        """Test strategy parameter validation"""
        params = self.strategy.get_strategy_parameters()
        
        # Test parameter presence
        required_params = [
            'rsi_period',
            'rsi_overbought',
            'rsi_oversold',
            'ema_short',
            'ema_long'
        ]
        
        for param in required_params:
            self.assertIn(param, params)

if __name__ == '__main__':
    unittest.main()