import os
from datetime import datetime
import logging
from pathlib import Path
import ccxt
import pandas as pd

from config.settings import (
    TRADING_SYMBOLS,
    TRADING_MODE,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL
)
from data.fetcher import DataFetcher
from data.database import DatabaseManager
from models.indicators import TechnicalIndicators
from models.ai_model import AIModel
from strategies.trading_strategies import TradingStrategy
from strategies.risk_management import RiskManager

# Setup logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.current_time = datetime(2025, 3, 6, 23, 33, 41)
        self.current_user = 'dhineshk6'
        
        logger.info(f"Initializing Trading Bot for user {self.current_user}")
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.db_manager = DatabaseManager()
        self.trading_strategy = TradingStrategy()
        self.ai_model = AIModel()
        
        # Initialize account balance
        self.account_balance = self._get_account_balance()
        self.risk_manager = RiskManager(self.account_balance)
        
    def _get_account_balance(self) -> float:
        """Get account balance"""
        try:
            balance = self.data_fetcher.fetch_balance()
            if balance and 'total' in balance:
                return float(balance['total'].get('USDT', 0))
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
        return 10000  # Default paper trading balance
    
    def run(self):
        """Main trading loop"""
        logger.info(f"Starting trading bot in {TRADING_MODE} mode")
        
        try:
            for symbol in TRADING_SYMBOLS:
                logger.info(f"Processing symbol: {symbol}")
                
                # Fetch market data
                df = self.data_fetcher.fetch_historical_data(symbol)
                if df.empty:
                    logger.error(f"No data available for {symbol}")
                    continue
                
                # Generate trading signals
                signals = self.trading_strategy.generate_signals(df)
                
                # Log signals
                logger.info(f"Generated signals for {symbol}: {signals['direction']} "
                           f"(strength: {signals['strength']:.2f}, "
                           f"confidence: {signals['confidence']:.2f})")
                
                # Execute trades if conditions are met
                self._execute_trades(symbol, signals, df)
                
        except Exception as e:
            logger.error(f"Error in main trading loop: {e}")
    
    def _execute_trades(self, symbol: str, signals: dict, df: pd.DataFrame):
        """Execute trades based on signals"""
        try:
            if signals['direction'] == 'neutral':
                return
            
            # Get current positions
            positions = self.db_manager.get_positions(symbol=symbol)
            
            # Check if we should enter a new position
            if not positions and signals['confidence'] > 0.7:
                self._enter_position(symbol, signals, df)
            
            # Check if we should exit existing positions
            for position in positions:
                self._manage_position(position, signals, df)
                
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
    
    def _enter_position(self, symbol: str, signals: dict, df: pd.DataFrame):
        """Enter a new trading position"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(current_price)
            
            if position_size == 0:
                return
            
            # Calculate stop loss and take profit
            stop_loss = self.risk_manager.calculate_stop_loss(
                current_price,
                'long' if signals['direction'] == 'buy' else 'short'
            )
            
            take_profit = self.risk_manager.calculate_take_profit(
                current_price,
                'long' if signals['direction'] == 'buy' else 'short'
            )
            
            # Validate trade
            is_valid, message = self.risk_manager.validate_trade(
                symbol,
                signals['direction'],
                current_price,
                position_size
            )
            
            if not is_valid:
                logger.warning(f"Trade validation failed: {message}")
                return
            
            # Store trade in database
            trade_data = {
                'symbol': symbol,
                'entry_time': self.current_time,
                'side': 'long' if signals['direction'] == 'buy' else 'short',
                'entry_price': current_price,
                'quantity': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': 'multi_strategy_v1'
            }
            
            trade_id = self.db_manager.store_trade(trade_data)
            logger.info(f"Entered new position: {trade_data}")
            
        except Exception as e:
            logger.error(f"Error entering position: {e}")
    
    def _manage_position(self, position: dict, signals: dict, df: pd.DataFrame):
        """Manage existing position"""
        try:
            current_price = df['close'].iloc[-1]
            
            # Check if we should close the position
            should_close, reason = self.risk_manager.should_close_position(
                position,
                current_price
            )
            
            if should_close:
                # Update position in database
                update_data = {
                    'exit_time': self.current_time,
                    'exit_price': current_price,
                    'status': 'closed'
                }
                
                self.db_manager.update_trade(position['id'], update_data)
                logger.info(f"Closed position {position['id']}: {reason}")
                
            else:
                # Update trailing stop if needed
                new_stop = self.risk_manager.update_trailing_stop(
                    position,
                    current_price
                )
                
                if new_stop:
                    self.db_manager.update_trade(
                        position['id'],
                        {'stop_loss': new_stop}
                    )
                    logger.info(f"Updated trailing stop for position {position['id']}")
                
        except Exception as e:
            logger.error(f"Error managing position: {e}")

def main():
    """Main entry point"""
    try:
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize and run trading bot
        bot = TradingBot()
        bot.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()