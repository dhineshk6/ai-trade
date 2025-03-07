from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics data class"""
    current_drawdown: float
    peak_capital: float
    open_risk: float
    total_exposure: float
    position_count: int
    daily_loss: float
    weekly_loss: float

class RiskManager:
    """Risk management system for trading strategies"""
    
    def __init__(self,
                 max_position_size: float = 0.1,
                 max_risk_per_trade: float = 0.02,
                 max_trades: int = 5,
                 max_drawdown: float = 0.1,
                 max_daily_loss: float = 0.05,
                 max_weekly_loss: float = 0.1,
                 max_correlation: float = 0.7,
                 position_sizing_method: str = 'risk_based'):
        """
        Initialize risk manager
        
        Args:
            max_position_size (float): Maximum position size as % of capital
            max_risk_per_trade (float): Maximum risk per trade as % of capital
            max_trades (int): Maximum number of concurrent trades
            max_drawdown (float): Maximum allowed drawdown as % of capital
            max_daily_loss (float): Maximum daily loss as % of capital
            max_weekly_loss (float): Maximum weekly loss as % of capital
            max_correlation (float): Maximum correlation between positions
            position_sizing_method (str): Method for position sizing ('risk_based' or 'fixed')
        """
        self.current_time = datetime(2025, 3, 7, 0, 47, 28)
        self.current_user = 'dhineshk6'
        
        # Risk limits
        self.max_position_size = max_position_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_trades = max_trades
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_weekly_loss = max_weekly_loss
        self.max_correlation = max_correlation
        self.position_sizing_method = position_sizing_method
        
        # Risk tracking
        self.initial_capital = 0.0
        self.peak_capital = 0.0
        self.daily_trades = []
        self.weekly_trades = []
        self.open_positions = []

    def calculate_position_size(self,
                              capital: float,
                              price: float,
                              risk_per_trade: float,
                              volatility: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            capital (float): Available capital
            price (float): Current asset price
            risk_per_trade (float): Risk per trade as %
            volatility (float, optional): Asset volatility
            
        Returns:
            float: Position size in base currency
        """
        try:
            # Update peak capital
            self.peak_capital = max(self.peak_capital, capital)
            
            # Calculate risk amount
            risk_amount = min(
                capital * risk_per_trade,
                capital * self.max_risk_per_trade
            )
            
            if self.position_sizing_method == 'risk_based':
                # Adjust for volatility if provided
                if volatility:
                    risk_amount = risk_amount * (1 / volatility)
                position_size = risk_amount / price
            else:
                # Fixed percentage of capital
                position_size = (capital * self.max_position_size) / price
            
            # Apply position size limits
            max_size = capital * self.max_position_size / price
            position_size = min(position_size, max_size)
            
            return round(position_size, 8)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def calculate_stop_loss(self,
                          price: float,
                          direction: str,
                          stop_loss_pct: float,
                          atr: Optional[float] = None) -> float:
        """
        Calculate stop loss price
        
        Args:
            price (float): Entry price
            direction (str): Trade direction ('long' or 'short')
            stop_loss_pct (float): Stop loss percentage
            atr (float, optional): Average True Range for dynamic stops
            
        Returns:
            float: Stop loss price
        """
        try:
            # Calculate base stop loss
            if direction == 'long':
                stop_loss = price * (1 - stop_loss_pct)
                if atr:
                    # Dynamic stop using ATR
                    atr_stop = price - (atr * 2)
                    stop_loss = max(stop_loss, atr_stop)
            else:  # short
                stop_loss = price * (1 + stop_loss_pct)
                if atr:
                    # Dynamic stop using ATR
                    atr_stop = price + (atr * 2)
                    stop_loss = min(stop_loss, atr_stop)
            
            return round(stop_loss, 8)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return price

    def calculate_take_profit(self,
                            price: float,
                            direction: str,
                            take_profit_pct: float,
                            risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit price
        
        Args:
            price (float): Entry price
            direction (str): Trade direction ('long' or 'short')
            take_profit_pct (float): Take profit percentage
            risk_reward_ratio (float): Desired risk/reward ratio
            
        Returns:
            float: Take profit price
        """
        try:
            if direction == 'long':
                take_profit = price * (1 + (take_profit_pct * risk_reward_ratio))
            else:  # short
                take_profit = price * (1 - (take_profit_pct * risk_reward_ratio))
            
            return round(take_profit, 8)
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return price

    def check_risk_limits(self,
                         capital: float,
                         drawdown: float,
                         open_positions: int,
                         new_position: Dict = None) -> bool:
        """
        Check if trade complies with risk limits
        
        Args:
            capital (float): Current capital
            drawdown (float): Current drawdown percentage
            open_positions (int): Number of open positions
            new_position (Dict, optional): New position to check
            
        Returns:
            bool: True if within risk limits, False otherwise
        """
        try:
            # Initialize risk metrics
            metrics = self._calculate_risk_metrics(capital)
            
            # Check drawdown limit
            if abs(drawdown) >= self.max_drawdown:
                logger.warning(f"Max drawdown limit reached: {drawdown:.2f}%")
                return False
            
            # Check maximum number of trades
            if open_positions >= self.max_trades:
                logger.warning(f"Max trades limit reached: {open_positions}")
                return False
            
            # Check daily loss limit
            if metrics.daily_loss >= self.max_daily_loss:
                logger.warning(f"Daily loss limit reached: {metrics.daily_loss:.2f}%")
                return False
            
            # Check weekly loss limit
            if metrics.weekly_loss >= self.max_weekly_loss:
                logger.warning(f"Weekly loss limit reached: {metrics.weekly_loss:.2f}%")
                return False
            
            # Check correlation if new position provided
            if new_position and self.open_positions:
                if self._check_correlation(new_position):
                    logger.warning("Position correlation limit reached")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False

    def _calculate_risk_metrics(self, current_capital: float) -> RiskMetrics:
        """Calculate current risk metrics"""
        try:
            # Calculate drawdown
            peak_capital = max(self.peak_capital, current_capital)
            current_drawdown = (peak_capital - current_capital) / peak_capital
            
            # Calculate open risk
            open_risk = sum(pos['risk_amount'] for pos in self.open_positions)
            
            # Calculate total exposure
            total_exposure = sum(pos['position_size'] * pos['entry_price'] 
                               for pos in self.open_positions)
            
            # Calculate daily and weekly losses
            daily_loss = self._calculate_period_loss(self.daily_trades)
            weekly_loss = self._calculate_period_loss(self.weekly_trades)
            
            return RiskMetrics(
                current_drawdown=current_drawdown,
                peak_capital=peak_capital,
                open_risk=open_risk,
                total_exposure=total_exposure,
                position_count=len(self.open_positions),
                daily_loss=daily_loss,
                weekly_loss=weekly_loss
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)

    def _calculate_period_loss(self, trades: List[Dict]) -> float:
        """Calculate loss for a given period"""
        try:
            if not trades:
                return 0.0
            
            total_pnl = sum(trade['pnl'] for trade in trades)
            return abs(min(0, total_pnl)) / self.peak_capital
            
        except Exception as e:
            logger.error(f"Error calculating period loss: {e}")
            return 0.0

    def _check_correlation(self, new_position: Dict) -> bool:
        """Check correlation with existing positions"""
        try:
            for position in self.open_positions:
                if position['symbol'] == new_position['symbol']:
                    continue
                
                correlation = self._calculate_correlation(
                    position['price_history'],
                    new_position['price_history']
                )
                
                if abs(correlation) > self.max_correlation:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking correlation: {e}")
            return True

    def _calculate_correlation(self, 
                             price_history1: pd.Series, 
                             price_history2: pd.Series) -> float:
        """Calculate correlation between two price series"""
        try:
            return price_history1.corr(price_history2)
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 1.0

    def update_trade_history(self, trade: Dict):
        """Update trade history for risk tracking"""
        try:
            # Update daily trades
            current_date = self.current_time.date()
            self.daily_trades = [t for t in self.daily_trades 
                               if t['exit_time'].date() == current_date]
            self.daily_trades.append(trade)
            
            # Update weekly trades
            week_start = current_date - pd.Timedelta(days=current_date.weekday())
            self.weekly_trades = [t for t in self.weekly_trades 
                                if t['exit_time'].date() >= week_start]
            self.weekly_trades.append(trade)
            
        except Exception as e:
            logger.error(f"Error updating trade history: {e}")

    def get_risk_metrics(self, capital: float) -> Dict:
        """Get current risk metrics"""
        try:
            metrics = self._calculate_risk_metrics(capital)
            
            return {
                'current_drawdown': metrics.current_drawdown,
                'peak_capital': metrics.peak_capital,
                'open_risk': metrics.open_risk,
                'total_exposure': metrics.total_exposure,
                'position_count': metrics.position_count,
                'daily_loss': metrics.daily_loss,
                'weekly_loss': metrics.weekly_loss,
                'timestamp': self.current_time,
                'user': self.current_user
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {}