import unittest
from datetime import datetime
from strategies.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.current_time = datetime(2025, 3, 7, 1, 5, 41)
        self.current_user = 'dhineshk6'
        
        self.risk_manager = RiskManager(
            max_position_size=0.1,
            max_risk_per_trade=0.02,
            max_trades=5,
            max_drawdown=0.1
        )

    def test_position_sizing(self):
        """Test position size calculation"""
        capital = 10000
        price = 100
        risk = 0.02
        
        # Test normal calculation
        position_size = self.risk_manager.calculate_position_size(
            capital,
            price,
            risk
        )
        self.assertGreater(position_size, 0)
        self.assertLess(position_size * price, capital)
        
        # Test maximum position size limit
        max_position = self.risk_manager.calculate_position_size(
            capital,
            price,
            0.5  # Excessive risk
        )
        self.assertLessEqual(
            max_position * price,
            capital * self.risk_manager.max_position_size
        )

    def test_stop_loss_calculation(self):
        """Test stop loss calculation"""
        price = 100
        
        # Test long position
        long_stop = self.risk_manager.calculate_stop_loss(
            price,
            'long',
            0.02
        )
        self.assertLess(long_stop, price)
        
        # Test short position
        short_stop = self.risk_manager.calculate_stop_loss(
            price,
            'short',
            0.02
        )
        self.assertGreater(short_stop, price)

    def test_take_profit_calculation(self):
        """Test take profit calculation"""
        price = 100
        
        # Test long position
        long_tp = self.risk_manager.calculate_take_profit(
            price,
            'long',
            0.04
        )
        self.assertGreater(long_tp, price)
        
        # Test short position
        short_tp = self.risk_manager.calculate_take_profit(
            price,
            'short',
            0.04
        )
        self.assertLess(short_tp, price)

    def test_risk_limits(self):
        """Test risk limit checks"""
        # Test normal conditions
        self.assertTrue(
            self.risk_manager.check_risk_limits(
                capital=10000,
                drawdown=0.05,
                open_positions=2
            )
        )
        
        # Test maximum drawdown
        self.assertFalse(
            self.risk_manager.check_risk_limits(
                capital=10000,
                drawdown=0.15,  # Exceeds max_drawdown
                open_positions=2
            )
        )
        
        # Test maximum positions
        self.assertFalse(
            self.risk_manager.check_risk_limits(
                capital=10000,
                drawdown=0.05,
                open_positions=6  # Exceeds max_trades
            )
        )

    def test_risk_metrics(self):
        """Test risk metrics calculation"""
        metrics = self.risk_manager.get_risk_metrics(10000)
        
        required_metrics = [
            'current_drawdown',
            'peak_capital',
            'open_risk',
            'total_exposure',
            'position_count'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)

if __name__ == '__main__':
    unittest.main()