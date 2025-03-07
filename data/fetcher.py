import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

from config.settings import (
    KUCOIN_API_KEY,
    KUCOIN_API_SECRET,
    KUCOIN_API_PASSPHRASE,
    TRADING_SYMBOLS,
    BASE_TIMEFRAME,
    API_RATE_LIMIT
)

class DataFetcher:
    def __init__(self):
        """Initialize KuCoin API connection"""
        self.exchange = ccxt.kucoinfutures({
            'apiKey': KUCOIN_API_KEY,
            'secret': KUCOIN_API_SECRET,
            'password': KUCOIN_API_PASSPHRASE,
            'enableRateLimit': True
        })
        
        self.last_api_call = 0
    
    def _respect_rate_limit(self):
        """Ensure API rate limits are respected"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < API_RATE_LIMIT:
            time.sleep(API_RATE_LIMIT - time_since_last_call)
        
        self.last_api_call = time.time()

    def fetch_historical_data(
        self, 
        symbol: str, 
        timeframe: str = BASE_TIMEFRAME,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch historical market data"""
        try:
            self._respect_rate_limit()
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_current_price(self, symbol: str) -> float:
        """Fetch current price for a symbol"""
        try:
            self._respect_rate_limit()
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return None

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """Fetch order book data"""
        try:
            self._respect_rate_limit()
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return {
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': order_book['timestamp'],
                'datetime': order_book['datetime']
            }
        except Exception as e:
            print(f"Error fetching order book for {symbol}: {e}")
            return None

    def fetch_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Fetch recent trades"""
        try:
            self._respect_rate_limit()
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            return trades
        except Exception as e:
            print(f"Error fetching recent trades for {symbol}: {e}")
            return []

    def fetch_market_info(self, symbol: str) -> Dict:
        """Fetch market information"""
        try:
            self._respect_rate_limit()
            market = self.exchange.market(symbol)
            return {
                'symbol': market['symbol'],
                'base': market['base'],
                'quote': market['quote'],
                'active': market['active'],
                'precision': market['precision'],
                'limits': market['limits']
            }
        except Exception as e:
            print(f"Error fetching market info for {symbol}: {e}")
            return None

    def fetch_balance(self) -> Dict:
        """Fetch account balance"""
        try:
            self._respect_rate_limit()
            balance = self.exchange.fetch_balance()
            return {
                'total': balance['total'],
                'used': balance['used'],
                'free': balance['free']
            }
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return None