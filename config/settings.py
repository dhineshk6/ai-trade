import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY')
KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET')
KUCOIN_API_PASSPHRASE = os.getenv('KUCOIN_API_PASSPHRASE')

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')

# Trading Parameters
TRADING_SYMBOLS = [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'XRP/USDT:USDT',
    'SOL/USDT:USDT',
    'DOT/USDT:USDT'
]

# Timeframes
BASE_TIMEFRAME = '1h'
TIMEFRAMES = ['1h', '4h', '1d']

# Risk Management
MAX_POSITION_SIZE = 0.1  # Maximum position size as percentage of portfolio
RISK_PER_TRADE = 0.02   # Risk per trade as percentage of portfolio
MAX_LEVERAGE = 5        # Maximum leverage to use
STOP_LOSS_PCT = 0.02   # Default stop loss percentage
TAKE_PROFIT_PCT = 0.04 # Default take profit percentage
MAX_OPEN_POSITIONS = 3  # Maximum number of open positions
TRAILING_STOP_PCT = 0.01  # Trailing stop percentage

# Technical Analysis Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_SHORT = 9
EMA_LONG = 21
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
VOLUME_MA_PERIOD = 20

# AI Model Parameters
LOOKBACK_PERIOD = 60
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 32
EPOCHS = 50
FEATURES = [
    'close',
    'volume',
    'rsi',
    'macd_diff',
    'bb_width',
    'atr_ratio',
    'obv_change',
    'mfi',
    'adx',
    'cci'
]

# Performance Monitoring
UPDATE_INTERVAL = 60  # Seconds between updates
MIN_TRADE_INTERVAL = 300  # Minimum seconds between trades
PERFORMANCE_METRICS = [
    'total_trades',
    'win_rate',
    'profit_factor',
    'max_drawdown',
    'sharpe_ratio',
    'avg_trade_duration'
]

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'logs/trading.log'

# User Configuration
CURRENT_USER = 'dhineshk6'
TIMEZONE = 'UTC'

# System Configuration
MAX_RETRIES = 3
RETRY_DELAY = 1
API_RATE_LIMIT = 1.0  # Seconds between API calls

class TradingMode:
    LIVE = 'live'
    PAPER = 'paper'
    BACKTEST = 'backtest'

# Trading Mode Configuration
TRADING_MODE = TradingMode.PAPER  # Default to paper trading

def get_current_time():
    """Get current time in UTC"""
    return datetime.utcnow()

def validate_configuration():
    """Validate the configuration settings"""
    required_env_vars = [
        'KUCOIN_API_KEY',
        'KUCOIN_API_SECRET',
        'KUCOIN_API_PASSPHRASE'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    if MAX_POSITION_SIZE > 1 or MAX_POSITION_SIZE <= 0:
        raise ValueError("MAX_POSITION_SIZE must be between 0 and 1")
    
    if RISK_PER_TRADE > 0.05:  # Maximum 5% risk per trade
        raise ValueError("RISK_PER_TRADE cannot exceed 5%")
    
    if MAX_LEVERAGE > 20:  # Maximum leverage limit
        raise ValueError("MAX_LEVERAGE cannot exceed 20")

# Configuration for different trading modes
MODE_CONFIG = {
    TradingMode.LIVE: {
        'use_real_money': True,
        'enable_alerts': True,
        'enable_stop_loss': True,
        'enable_take_profit': True,
        'max_positions': MAX_OPEN_POSITIONS
    },
    TradingMode.PAPER: {
        'use_real_money': False,
        'enable_alerts': True,
        'enable_stop_loss': True,
        'enable_take_profit': True,
        'max_positions': MAX_OPEN_POSITIONS
    },
    TradingMode.BACKTEST: {
        'use_real_money': False,
        'enable_alerts': False,
        'enable_stop_loss': True,
        'enable_take_profit': True,
        'max_positions': MAX_OPEN_POSITIONS
    }
}

# Validate configuration on import
validate_configuration()