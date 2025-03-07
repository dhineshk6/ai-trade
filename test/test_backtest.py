import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from strategies.backtest_engine import BacktestEngine
from strategies.trading_strategy import TradingStrategy
from strategies.risk_manager import RiskManager

class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        self.current_time = datetime(2025, 3, 7, 1, 5, 41)
        self.current_user = 'dhineshk6'
        
        # Initialize components
        self.backtest = BacktestEngine(
            initial_capital=10000,
            trading_fee=0.001,
            slippage=0.001
        )
        self.strategy = TradingStrategy()
        self.risk_manager = RiskManager()
        
        # Create sample data
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

    def test_initialization(self):
        """Test backtest engine initialization"""
        self.assertEqual(self.backtest.initial_capital, 10000)
        self.assertEqual(self.backtest.current_capital, 10000)
        self.assertEqual(len(self.backtest.positions), 0)
        self.assertEqual(len(self.backtest.trades), 0)

    def test_data_preparation(self):
        """Test data preparation and validation"""
        # Test with valid data
        prepared_data = self.backtest._prepare_data(
            self.data,
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 3, 7)
        )
        self.assertFalse(prepared_data.empty)
        
        # Test with missing columns
        invalid_data = self.data.drop(['volume'], axis=1)
        with self.assertRaises(ValueError):
            self.backtest._prepare_data(invalid_data, None, None)

    def test_position_management(self):
        """Test position opening and closing"""
        # Create sample position
        timestamp = self.current_time
        candle = self.data.iloc[0]
        
        signals = {
            'action': 'enter',
            'direction': 'long',
            'risk_per_trade': 0.02,
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'strategy': 'test'
        }
        
        # Test position opening
        self.backtest._execute_signals(
            timestamp,
            candle,
            signals,
            self.risk_manager
        )
        self.assertEqual(len(self.backtest.positions), 1)
        
        # Test position closing
        position = self.backtest.positions[0]
        self.backtest._close_position(
            position,
            timestamp + timedelta(hours=1),
            candle['close'] * 1.05,
            'test'
        )
        self.assertEqual(len(self.backtest.positions), 0)
        self.assertEqual(len(self.backtest.trades), 1)

    def test_risk_management(self):
        """Test risk management integration"""
        # Test position sizing
        capital = 10000
        price = 100
        risk = 0.02
        
        position_size = self.risk_manager.calculate_position_size(
            capital,
            price,
            risk
        )
        self.assertGreater(position_size, 0)
        self.assertLess(position_size * price, capital)
        
        # Test stop loss calculation
        stop_loss = self.risk_manager.calculate_stop_loss(
            price,
            'long',
            0.02
        )
        self.assertLess(stop_loss, price)

    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Run backtest
        results = self.backtest.run(
            self.data,
            self.strategy,
            self.risk_manager
        )
        
        # Test metrics existence
        required_metrics = [
            'total_return',
            'sharpe_ratio',
            'max_drawdown',
            'win_rate'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, results)
            self.assertIsNotNone(results[metric])
        
        # Test metric values
        self.assertGreaterEqual(results['win_rate'], 0)
        self.assertLessEqual(results['win_rate'], 1)
        self.assertGreaterEqual(results['max_drawdown'], 0)

    def test_result_plotting(self):
        """Test results plotting functionality"""
        # Run backtest
        results = self.backtest.run(
            self.data,
            self.strategy,
            self.risk_manager
        )
        
        # Test plot generation
        try:
            self.backtest.plot_results(
                results,
                save_path='test_results.png'
            )
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Plot generation failed: {e}")

    def test_results_saving_loading(self):
        """Test results saving and loading"""
        # Run backtest
        original_results = self.backtest.run(
            self.data,
            self.strategy,
            self.risk_manager
        )
        
        # Save results
        save_path = 'test_results.json'
        self.backtest.save_results(original_results, save_path)
        
        # Load results
        loaded_results = self.backtest.load_results(save_path)
        
        # Compare results
        self.assertEqual(
            original_results['total_return'],
            loaded_results['total_return']
        )
        self.assertEqual(
            original_results['total_trades'],
            loaded_results['total_trades']
        )

    def test_threading(self):
        """Test threading functionality"""
        # Test with threading enabled
        results_threaded = self.backtest.run(
            self.data,
            self.strategy,
            self.risk_manager
        )
        
        # Test with threading disabled
        self.backtest.enable_threading = False
        results_non_threaded = self.backtest.run(
            self.data,
            self.strategy,
            self.risk_manager
        )
        
        # Compare results
        self.assertAlmostEqual(
            results_threaded['total_return'],
            results_non_threaded['total_return'],
            places=4
        )

    def test_error_handling(self):
        """Test error handling"""
        # Test with invalid data
        invalid_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.backtest.run(
                invalid_data,
                self.strategy,
                self.risk_manager
            )
        
        # Test with invalid strategy
        with self.assertRaises(AttributeError):
            self.backtest.run(
                self.data,
                None,
                self.risk_manager
            )

if __name__ == '__main__':
    unittest.main()