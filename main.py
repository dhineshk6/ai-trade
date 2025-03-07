#!/usr/bin/env python3
"""
Trading System Main Application
-----------------------------
Created: 2025-03-07 01:15:26
Author: dhineshk6
"""

import logging
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import yaml
import pandas as pd

# Update imports to use the new package structure
from ai_trade.strategies.trading_strategy import TradingStrategy
from ai_trade.strategies.risk_manager import RiskManager
from ai_trade.strategies.backtest_engine import BacktestEngine
from ai_trade.data.fetcher import DataFetcher
from ai_trade.utils.helpers import setup_logging

# Initialize logging
logger = setup_logging()

class TradingApplication:
    def __init__(self):
        self.current_time = datetime(2025, 3, 7, 1, 15, 26)
        self.current_user = 'dhineshk6'
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.strategy = TradingStrategy(
            rsi_period=self.config['strategy']['rsi_period'],
            rsi_overbought=self.config['strategy']['rsi_overbought'],
            rsi_oversold=self.config['strategy']['rsi_oversold'],
            ema_short=self.config['strategy']['ema_short'],
            ema_long=self.config['strategy']['ema_long']
        )
        
        self.risk_manager = RiskManager(
            max_position_size=self.config['risk']['max_position_size'],
            max_risk_per_trade=self.config['risk']['max_risk_per_trade'],
            max_trades=self.config['risk']['max_trades'],
            max_drawdown=self.config['risk']['max_drawdown']
        )
        
        self.data_fetcher = DataFetcher()
        self.backtest_engine = BacktestEngine(
            initial_capital=self.config['backtest']['initial_capital'],
            trading_fee=self.config['backtest']['trading_fee'],
            slippage=self.config['backtest']['slippage']
        )

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            config_path = Path('config/config.yml')
            if not config_path.exists():
                logger.error("Configuration file not found")
                self._create_default_config(config_path)
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)

    def _create_default_config(self, config_path: Path):
        """Create default configuration file"""
        default_config = {
            'strategy': {
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'ema_short': 9,
                'ema_long': 21
            },
            'risk': {
                'max_position_size': 0.1,
                'max_risk_per_trade': 0.02,
                'max_trades': 5,
                'max_drawdown': 0.1
            },
            'backtest': {
                'initial_capital': 10000,
                'trading_fee': 0.001,
                'slippage': 0.001
            },
            'data': {
                'timeframe': '1h',
                'symbols': ['BTC/USDT', 'ETH/USDT']
            }
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        
        logger.info("Created default configuration file")

    def run_backtest(self,
                    symbol: str,
                    start_date: datetime,
                    end_date: datetime,
                    timeframe: str = '1h') -> Dict:
        """Run backtest for specified parameters"""
        try:
            logger.info(f"Starting backtest for {symbol}")
            
            # Fetch historical data
            data = self.data_fetcher.fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            if data.empty:
                raise ValueError("No data available for backtest")
            
            # Run backtest
            results = self.backtest_engine.run(
                data=data,
                strategy=self.strategy,
                risk_manager=self.risk_manager
            )
            
            # Save results
            self._save_results(results, symbol, timeframe)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}

    def _save_results(self,
                     results: Dict,
                     symbol: str,
                     timeframe: str):
        """Save backtest results"""
        try:
            # Create results directory
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # Create timestamp string
            timestamp = self.current_time.strftime('%Y%m%d_%H%M%S')
            
            # Save results
            results_path = results_dir / f'backtest_{symbol}_{timeframe}_{timestamp}.json'
            self.backtest_engine.save_results(results, str(results_path))
            
            # Generate and save plot
            plot_path = results_dir / f'backtest_{symbol}_{timeframe}_{timestamp}.png'
            self.backtest_engine.plot_results(results, str(plot_path))
            
            logger.info(f"Results saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Trading System')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Data timeframe')
    
    args = parser.parse_args()
    
    try:
        # Initialize application
        app = TradingApplication()
        
        # Parse dates
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
        
        # Run backtest
        results = app.run_backtest(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=args.timeframe
        )
        
        # Print summary
        if results:
            print("\nBacktest Summary:")
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            print(f"Win Rate: {results['win_rate']*100:.2f}%")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
