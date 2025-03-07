#!/usr/bin/env python3
"""
Backtest Runner Script
--------------------
Created: 2025-03-07 01:15:26
Author: dhineshk6
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yaml
import concurrent.futures

from strategies.trading_strategy import TradingStrategy
from strategies.risk_manager import RiskManager
from strategies.backtest_engine import BacktestEngine
from data.fetcher import DataFetcher
from utils.helpers import setup_logging

# Initialize logging
logger = setup_logging()

class BacktestRunner:
    def __init__(self):
        self.current_time = datetime(2025, 3, 7, 1, 15, 26)
        self.current_user = 'dhineshk6'
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.strategies = self._initialize_strategies()
        self.risk_manager = self._initialize_risk_manager()
        self.backtest_engine = self._initialize_backtest_engine()

    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            config_path = Path('config/backtest_config.yml')
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _initialize_strategies(self) -> List[TradingStrategy]:
        """Initialize trading strategies"""
        strategies = []
        for strategy_config in self.config['strategies']:
            strategy = TradingStrategy(
                rsi_period=strategy_config.get('rsi_period', 14),
                rsi_overbought=strategy_config.get('rsi_overbought', 70),
                rsi_oversold=strategy_config.get('rsi_oversold', 30),
                ema_short=strategy_config.get('ema_short', 9),
                ema_long=strategy_config.get('ema_long', 21)
            )
            strategies.append(strategy)
        return strategies

    def _initialize_risk_manager(self) -> RiskManager:
        """Initialize risk manager"""
        return RiskManager(
            max_position_size=self.config['risk']['max_position_size'],
            max_risk_per_trade=self.config['risk']['max_risk_per_trade'],
            max_trades=self.config['risk']['max_trades'],
            max_drawdown=self.config['risk']['max_drawdown']
        )

    def _initialize_backtest_engine(self) -> BacktestEngine:
        """Initialize backtest engine"""
        return BacktestEngine(
            initial_capital=self.config['backtest']['initial_capital'],
            trading_fee=self.config['backtest']['trading_fee'],
            slippage=self.config['backtest']['slippage']
        )

    def run_backtests(self):
        """Run multiple backtests in parallel"""
        try:
            logger.info("Starting backtest runs...")
            
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_params = {
                    executor.submit(
                        self._run_single_backtest,
                        symbol,
                        strategy,
                        self.config['backtest']['start_date'],
                        self.config['backtest']['end_date'],
                        self.config['backtest']['timeframe']
                    ): (symbol, strategy.__class__.__name__)
                    for symbol in self.config['symbols']
                    for strategy in self.strategies
                }
                
                for future in concurrent.futures.as_completed(future_to_params):
                    symbol, strategy_name = future_to_params[future]
                    try:
                        result = future.result()
                        results.append({
                            'symbol': symbol,
                            'strategy': strategy_name,
                            'results': result
                        })
                        logger.info(f"Completed backtest for {symbol} using {strategy_name}")
                    except Exception as e:
                        logger.error(f"Backtest failed for {symbol} using {strategy_name}: {e}")
            
            # Save and analyze results
            self._process_results(results)
            
        except Exception as e:
            logger.error(f"Error running backtests: {e}")
            raise

    def _run_single_backtest(self,
                           symbol: str,
                           strategy: TradingStrategy,
                           start_date: datetime,
                           end_date: datetime,
                           timeframe: str) -> Dict:
        """Run single backtest"""
        try:
            # Fetch data
            data = self.data_fetcher.fetch_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Run backtest
            results = self.backtest_engine.run(
                data=data,
                strategy=strategy,
                risk_manager=self.risk_manager
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest run: {e}")
            raise

    def _process_results(self, results: List[Dict]):
        """Process and save backtest results"""
        try:
            # Create results directory
            results_dir = Path('results')
            results_dir.mkdir(exist_ok=True)
            
            # Create timestamp string
            timestamp = self.current_time.strftime('%Y%m%d_%H%M%S')
            
            # Save individual results
            for result in results:
                # Save detailed results
                result_path = results_dir / f"backtest_{result['symbol']}_{result['strategy']}_{timestamp}.json"
                self.backtest_engine.save_results(result['results'], str(result_path))
                
                # Generate plot
                plot_path = results_dir / f"backtest_{result['symbol']}_{result['strategy']}_{timestamp}.png"
                self.backtest_engine.plot_results(result['results'], str(plot_path))
            
            # Generate summary report
            self._generate_summary_report(results, results_dir, timestamp)
            
            logger.info(f"All results processed and saved to {results_dir}")
            
        except Exception as e:
            logger.error(f"Error processing results: {e}")

    def _generate_summary_report(self,
                              results: List[Dict],
                              results_dir: Path,
                              timestamp: str):
        """Generate summary report of all backtests"""
        summary_data = []
        
        for result in results:
            summary_data.append({
                'Symbol': result['symbol'],
                'Strategy': result['strategy'],
                'Total Return (%)': result['results']['total_return'],
                'Sharpe Ratio': result['results']['sharpe_ratio'],
                'Max Drawdown (%)': result['results']['max_drawdown'],
                'Win Rate (%)': result['results']['win_rate'] * 100,
                'Total Trades': result['results']['total_trades']
            })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = results_dir / f"backtest_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Print summary
        print("\nBacktest Summary:")
        print(summary_df.to_string())

def main():
    """Main entry point"""
    try:
        runner = BacktestRunner()
        runner.run_backtests()
        
    except Exception as e:
        logger.error(f"Runner error: {e}")
        raise

if __name__ == "__main__":
    main()