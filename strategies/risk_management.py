from datetime import datetime
import numpy as np
from typing import Dict, Tuple, Optional
from config.settings import (
    MAX_POSITION_SIZE,
    RISK_PER_TRADE,
    MAX_LEVERAGE,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT,
    CURRENT_USER
)

class RiskManager:
    def __init__(self, account_balance: float):
        self.account_balance = account_balance
        self.current_user = CURRENT_USER
        self.current_time = datetime(2025, 3, 6, 19, 52, 40)  # Updated current time
        
    def calculate_position_size(
        self, 
        entry_price: float,
        stop_loss: float = None,
        leverage: float = 1.0
    ) -> float:
        """Calculate safe position size based on risk parameters"""
        try:
            # If stop loss is provided, use it for risk calculation
            if stop_loss:
                risk_per_unit = abs(entry_price - stop_loss)
                max_risk_amount = self.account_balance * RISK_PER_TRADE
                position_size = (max_risk_amount / risk_per_unit) * leverage
            else:
                # Use default risk percentage
                position_size = (self.account_balance * MAX_POSITION_SIZE * leverage)
            
            # Ensure position size doesn't exceed maximum allowed
            max_allowed = self.account_balance * MAX_POSITION_SIZE * MAX_LEVERAGE
            return min(position_size, max_allowed)
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        side: str,
        atr: float = None
    ) -> float:
        """Calculate stop loss price"""
        try:
            if atr:
                # Use ATR for dynamic stop loss
                multiplier = 2.0  # Adjustable ATR multiplier
                stop_distance = atr * multiplier
            else:
                # Use default percentage
                stop_distance = entry_price * STOP_LOSS_PCT
            
            return entry_price - stop_distance if side == 'long' else entry_price + stop_distance
            
        except Exception as e:
            print(f"Error calculating stop loss: {e}")
            return entry_price * (1 - STOP_LOSS_PCT if side == 'long' else 1 + STOP_LOSS_PCT)
    
    def calculate_take_profit(
        self, 
        entry_price: float, 
        side: str,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """Calculate take profit price"""
        try:
            profit_distance = entry_price * TAKE_PROFIT_PCT * risk_reward_ratio
            return entry_price + profit_distance if side == 'long' else entry_price - profit_distance
            
        except Exception as e:
            print(f"Error calculating take profit: {e}")
            return entry_price * (1 + TAKE_PROFIT_PCT if side == 'long' else 1 - TAKE_PROFIT_PCT)
    
    def update_trailing_stop(
        self, 
        position: Dict,
        current_price: float
    ) -> Optional[float]:
        """Update trailing stop loss"""
        try:
            if position['side'] == 'long':
                # For long positions
                new_stop = current_price * (1 - TRAILING_STOP_PCT)
                if new_stop > position['stop_loss']:
                    return new_stop
            else:
                # For short positions
                new_stop = current_price * (1 + TRAILING_STOP_PCT)
                if new_stop < position['stop_loss']:
                    return new_stop
            
            return None
            
        except Exception as e:
            print(f"Error updating trailing stop: {e}")
            return None
    
    def should_close_position(
        self, 
        position: Dict,
        current_price: float
    ) -> Tuple[bool, str]:
        """Determine if position should be closed"""
        try:
            # Check stop loss
            if position['side'] == 'long':
                if current_price <= position['stop_loss']:
                    return True, 'stop_loss'
                if current_price >= position['take_profit']:
                    return True, 'take_profit'
            else:  # Short position
                if current_price >= position['stop_loss']:
                    return True, 'stop_loss'
                if current_price <= position['take_profit']:
                    return True, 'take_profit'
            
            return False, None
            
        except Exception as e:
            print(f"Error checking position closure: {e}")
            return False, None
    
    def validate_trade(
        self, 
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        leverage: float = 1.0
    ) -> Tuple[bool, str]:
        """Validate trade parameters"""
        try:
            # Check leverage
            if leverage > MAX_LEVERAGE:
                return False, f"Leverage {leverage} exceeds maximum allowed {MAX_LEVERAGE}"
            
            # Check position size
            position_value = entry_price * quantity
            max_position_value = self.account_balance * MAX_POSITION_SIZE * leverage
            if position_value > max_position_value:
                return False, f"Position size {position_value} exceeds maximum allowed {max_position_value}"
            
            # Check risk per trade
            risk_amount = position_value * STOP_LOSS_PCT
            max_risk = self.account_balance * RISK_PER_TRADE
            if risk_amount > max_risk:
                return False, f"Risk amount {risk_amount} exceeds maximum allowed {max_risk}"
            
            return True, "Trade validated successfully"
            
        except Exception as e:
            return False, f"Error validating trade: {e}"
    
    def calculate_position_risk(
        self, 
        positions: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Calculate current portfolio risk"""
        try:
            total_risk = 0
            position_risks = {}
            
            for symbol, position in positions.items():
                # Calculate individual position risk
                position_value = position['quantity'] * position['current_price']
                risk_amount = position_value * STOP_LOSS_PCT
                risk_percent = risk_amount / self.account_balance
                
                position_risks[symbol] = {
                    'risk_amount': risk_amount,
                    'risk_percent': risk_percent,
                    'position_value': position_value
                }
                
                total_risk += risk_percent
            
            return {
                'total_risk_percent': total_risk,
                'position_risks': position_risks,
                'available_risk': RISK_PER_TRADE - total_risk
            }
            
        except Exception as e:
            print(f"Error calculating position risk: {e}")
            return {
                'total_risk_percent': 0,
                'position_risks': {},
                'available_risk': RISK_PER_TRADE
            }
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'timestamp': self.current_time,
            'user': self.current_user,
            'account_balance': self.account_balance,
            'max_position_size': self.account_balance * MAX_POSITION_SIZE,
            'max_risk_per_trade': self.account_balance * RISK_PER_TRADE,
            'max_leverage': MAX_LEVERAGE,
            'risk_settings': {
                'stop_loss_pct': STOP_LOSS_PCT,
                'take_profit_pct': TAKE_PROFIT_PCT,
                'trailing_stop_pct': TRAILING_STOP_PCT
            }
        }