from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Trading position dataclass"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    strategy: str
    metadata: Dict = None

@dataclass
class Trade:
    """Completed trade dataclass"""
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    fees: float
    strategy: str
    stop_loss: float
    take_profit: float
    exit_reason: str
    metadata: Dict = None

class BacktestEngine:
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 trading_fee: float = 0.001,
                 slippage: float = 0.001,
                 enable_threading: bool = True):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital (float): Starting capital
            trading_fee (float): Trading fee as percentage
            slippage (float): Slippage as percentage
            enable_threading (bool): Enable multi-threading for calculations
        """
        self.current_time = datetime(2025, 3, 7, 0, 52, 20)
        self.current_user = 'dhineshk6'
        
        # Configuration
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trading_fee = trading_fee
        self.slippage = slippage
        self.enable_threading = enable_threading
        
        # State tracking
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.drawdown_curve: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        logger.info(
            f"Initialized BacktestEngine for {self.current_user} "
            f"with {initial_capital:.2f} initial capital"
        )

    def run(self,
           data: pd.DataFrame,
           strategy,
           risk_manager,
           start_date: Optional[datetime] = None,
           end_date: Optional[datetime] = None) -> Dict:
        """
        Run backtest simulation
        
        Args:
            data (pd.DataFrame): Historical market data
            strategy: Trading strategy instance
            risk_manager: Risk management instance
            start_date (datetime): Start date for backtest
            end_date (datetime): End date for backtest
            
        Returns:
            Dict: Backtest results and metrics
        """
        try:
            logger.info("Starting backtest simulation...")
            
            # Reset state
            self._reset_state()
            
            # Validate and prepare data
            data = self._prepare_data(data, start_date, end_date)
            
            # Run simulation
            with ThreadPoolExecutor() if self.enable_threading else NoThreading():
                for timestamp, candle in data.iterrows():
                    # Update positions
                    self._update_positions(timestamp, candle)
                    
                    # Generate signals
                    signals = strategy.generate_signals(
                        data.loc[:timestamp],
                        self.positions
                    )
                    
                    # Check risk limits
                    if risk_manager.check_risk_limits(
                        capital=self.current_capital,
                        drawdown=self._calculate_drawdown(),
                        open_positions=len(self.positions)
                    ):
                        # Execute trades
                        self._execute_signals(
                            timestamp,
                            candle,
                            signals,
                            risk_manager
                        )
                    
                    # Record metrics
                    self._record_metrics(timestamp, candle)
            
            # Close remaining positions
            self._close_all_positions(data.index[-1], data.iloc[-1])
            
            # Generate results
            results = self._generate_results()
            
            logger.info(
                f"Backtest completed. Final capital: {self.current_capital:.2f}, "
                f"Total return: {results['total_return']:.2f}%"
            )
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest simulation: {e}")
            raise

    def _reset_state(self):
        """Reset backtest state"""
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.trade_history = []

    def _prepare_data(self,
                     data: pd.DataFrame,
                     start_date: Optional[datetime],
                     end_date: Optional[datetime]) -> pd.DataFrame:
        """Prepare and validate data for backtesting"""
        try:
            # Filter date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # Validate data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Need: {required_columns}")
            
            if data.empty:
                raise ValueError("No data available for backtesting")
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def _update_positions(self, timestamp: datetime, candle: pd.Series):
        """Update open positions"""
        for position in self.positions[:]:
            # Apply slippage to prices
            high_price = candle['high'] * (1 + self.slippage)
            low_price = candle['low'] * (1 - self.slippage)
            
            # Check stop loss
            if position.direction == 'long':
                if low_price <= position.stop_loss:
                    self._close_position(
                        position,
                        timestamp,
                        position.stop_loss,
                        'stop_loss'
                    )
                    continue
            else:  # short position
                if high_price >= position.stop_loss:
                    self._close_position(
                        position,
                        timestamp,
                        position.stop_loss,
                        'stop_loss'
                    )
                    continue
            
            # Check take profit
            if position.direction == 'long':
                if high_price >= position.take_profit:
                    self._close_position(
                        position,
                        timestamp,
                        position.take_profit,
                        'take_profit'
                    )
            else:
                if low_price <= position.take_profit:
                    self._close_position(
                        position,
                        timestamp,
                        position.take_profit,
                        'take_profit'
                    )

    def _execute_signals(self,
                        timestamp: datetime,
                        candle: pd.Series,
                        signals: Dict,
                        risk_manager) -> None:
        """Execute trading signals"""
        try:
            if not signals or signals.get('action') == 'hold':
                return
            
            # Apply slippage to entry price
            price = candle['close']
            if signals['direction'] == 'long':
                entry_price = price * (1 + self.slippage)
            else:
                entry_price = price * (1 - self.slippage)
            
            # Calculate position size
            position_size = risk_manager.calculate_position_size(
                self.current_capital,
                entry_price,
                signals.get('risk_per_trade', 0.02)
            )
            
            if position_size == 0:
                return
            
            # Calculate stop loss and take profit
            stop_loss = risk_manager.calculate_stop_loss(
                entry_price,
                signals['direction'],
                signals.get('stop_loss_pct', 0.02)
            )
            
            take_profit = risk_manager.calculate_take_profit(
                entry_price,
                signals['direction'],
                signals.get('take_profit_pct', 0.04)
            )
            
            # Create new position
            position = Position(
                symbol=candle.name,
                direction=signals['direction'],
                entry_price=entry_price,
                quantity=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time=timestamp,
                strategy=signals.get('strategy', 'default'),
                metadata={
                    'signal_metadata': signals.get('metadata', {}),
                    'reason': signals.get('reason', 'N/A')
                }
            )
            
            self.positions.append(position)
            
            logger.info(
                f"Opened {position.direction} position: "
                f"Price: {entry_price:.8f}, "
                f"Size: {position_size:.8f}, "
                f"Stop: {stop_loss:.8f}, "
                f"Target: {take_profit:.8f}"
            )
            
        except Exception as e:
            logger.error(f"Error executing signals: {e}")

    def _close_position(self,
                       position: Position,
                       timestamp: datetime,
                       exit_price: float,
                       exit_reason: str) -> None:
        """Close a trading position"""
        try:
            # Calculate PnL
            if position.direction == 'long':
                pnl = (exit_price - position.entry_price) * position.quantity
            else:  # short
                pnl = (position.entry_price - exit_price) * position.quantity
            
            # Calculate fees
            entry_fee = position.entry_price * position.quantity * self.trading_fee
            exit_fee = exit_price * position.quantity * self.trading_fee
            total_fees = entry_fee + exit_fee
            
            # Update capital
            self.current_capital += pnl - total_fees
            
            # Record trade
            trade = Trade(
                symbol=position.symbol,
                direction=position.direction,
                entry_time=position.entry_time,
                exit_time=timestamp,
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                pnl=pnl,
                pnl_percent=(pnl / (position.entry_price * position.quantity)) * 100,
                fees=total_fees,
                strategy=position.strategy,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                exit_reason=exit_reason,
                metadata=position.metadata
            )
            
            self.trades.append(trade)
            self.positions.remove(position)
            
            # Record trade history
            self.trade_history.append({
                'timestamp': timestamp,
                'type': 'exit',
                'direction': position.direction,
                'price': exit_price,
                'quantity': position.quantity,
                'pnl': pnl,
                'fees': total_fees,
                'reason': exit_reason
            })
            
            logger.info(
                f"Closed {position.direction} position: "
                f"Exit: {exit_price:.8f}, "
                f"PnL: {pnl:.2f} ({trade.pnl_percent:.2f}%), "
                f"Reason: {exit_reason}"
            )
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def _close_all_positions(self, timestamp: datetime, candle: pd.Series):
        """Close all open positions"""
        for position in self.positions[:]:
            self._close_position(
                position,
                timestamp,
                candle['close'],
                'end_of_simulation'
            )

    def _record_metrics(self, timestamp: datetime, candle: pd.Series):
        """Record performance metrics"""
        # Calculate unrealized PnL
        unrealized_pnl = sum(
            (candle['close'] - pos.entry_price) * pos.quantity
            if pos.direction == 'long'
            else (pos.entry_price - candle['close']) * pos.quantity
            for pos in self.positions
        )
        
        # Calculate equity
        equity = self.current_capital + unrealized_pnl
        
        # Record equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })
        
        # Calculate and record drawdown
        peak_equity = max(point['equity'] for point in self.equity_curve)
        drawdown = (peak_equity - equity) / peak_equity * 100
        
        self.drawdown_curve.append({
            'timestamp': timestamp,
            'drawdown': drawdown
        })

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        if not self.equity_curve:
            return 0.0
        
        peak = max(point['equity'] for point in self.equity_curve)
        current = self.equity_curve[-1]['equity']
        return (peak - current) / peak * 100

    def _generate_results(self) -> Dict:
        """Generate backtest results and metrics"""
        try:
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            returns = equity_df['equity'].pct_change()
            
            # Calculate metrics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.pnl > 0])
            losing_trades = total_trades - winning_trades
            
            results = {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_return': ((self.current_capital - self.initial_capital) 
                                / self.initial_capital * 100),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': (winning_trades / total_trades 
                            if total_trades > 0 else 0),
                'total_fees': sum(t.fees for t in self.trades),
                
                # Risk metrics
                'max_drawdown': self._calculate_max_drawdown(equity_df['equity']),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'sortino_ratio': self._calculate_sortino_ratio(returns),
                'profit_factor': self._calculate_profit_factor(),
                
                # Trade analysis
                'avg_trade_return': np.mean([t.pnl_percent for t in self.trades]),
                'std_trade_return': np.std([t.pnl_percent for t in self.trades]),
                'max_trade_return': max([t.pnl_percent for t in self.trades]),
                'min_trade_return': min([t.pnl_percent for t in self.trades]),
                'avg_trade_duration': self._calculate_avg_trade_duration(),
                
                                # Time series
                'equity_curve': equity_df['equity'],
                'returns': returns,
                'drawdown': self._calculate_drawdown(equity_df['equity']),
                
                # Trade details
                'trade_history': self.trade_history,
                
                # Metadata
                'timestamp': self.current_time,
                'user': self.current_user,
                'backtest_params': {
                    'trading_fee': self.trading_fee,
                    'slippage': self.slippage,
                    'enable_threading': self.enable_threading
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating results: {e}")
            raise

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity.expanding(min_periods=1).max()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min()) * 100

    def _calculate_drawdown(self, equity: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        peak = equity.expanding(min_periods=1).max()
        return (equity - peak) / peak * 100

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.empty:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if returns.empty:
            return 0.0
        negative_returns = returns[returns < 0]
        if negative_returns.empty:
            return float('inf')
        return np.sqrt(252) * returns.mean() / negative_returns.std()

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        winning_trades = sum(t.pnl for t in self.trades if t.pnl > 0)
        losing_trades = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return winning_trades / losing_trades if losing_trades != 0 else float('inf')

    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in hours"""
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 3600 
            for t in self.trades
        ]
        return np.mean(durations) if durations else 0

    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot backtest results"""
        try:
            plt.style.use('seaborn')
            fig = plt.figure(figsize=(15, 10))
            
            # Create grid
            gs = plt.GridSpec(3, 2, figure=fig)
            
            # Equity curve
            ax1 = fig.add_subplot(gs[0, :])
            results['equity_curve'].plot(ax=ax1)
            ax1.set_title('Equity Curve')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Equity')
            
            # Drawdown
            ax2 = fig.add_subplot(gs[1, :])
            results['drawdown'].plot(ax=ax2, color='red', alpha=0.5)
            ax2.fill_between(
                results['drawdown'].index,
                results['drawdown'].values,
                0,
                color='red',
                alpha=0.3
            )
            ax2.set_title('Drawdown')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown %')
            
            # Returns distribution
            ax3 = fig.add_subplot(gs[2, 0])
            returns_data = [t.pnl_percent for t in self.trades]
            sns.histplot(returns_data, kde=True, ax=ax3)
            ax3.set_title('Returns Distribution')
            ax3.set_xlabel('Return %')
            ax3.set_ylabel('Frequency')
            
            # Monthly returns heatmap
            ax4 = fig.add_subplot(gs[2, 1])
            monthly_returns = results['returns'].resample('M').sum() * 100
            monthly_returns = monthly_returns.to_frame()
            monthly_returns['Year'] = monthly_returns.index.year
            monthly_returns['Month'] = monthly_returns.index.month
            
            pivot_table = monthly_returns.pivot_table(
                index='Year',
                columns='Month',
                values='equity'
            )
            
            sns.heatmap(
                pivot_table,
                annot=True,
                fmt='.1f',
                center=0,
                cmap='RdYlGn',
                ax=ax4
            )
            ax4.set_title('Monthly Returns Heatmap')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Results plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting results: {e}")

    def save_results(self, results: Dict, save_path: str):
        """Save backtest results to file"""
        try:
            # Convert DataFrame and Series objects to lists
            serializable_results = results.copy()
            serializable_results['equity_curve'] = results['equity_curve'].tolist()
            serializable_results['returns'] = results['returns'].tolist()
            serializable_results['drawdown'] = results['drawdown'].tolist()
            
            # Save to JSON
            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=4, default=str)
            
            logger.info(f"Results saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def load_results(self, load_path: str) -> Dict:
        """Load backtest results from file"""
        try:
            with open(load_path, 'r') as f:
                results = json.load(f)
            
            # Convert lists back to Series
            results['equity_curve'] = pd.Series(results['equity_curve'])
            results['returns'] = pd.Series(results['returns'])
            results['drawdown'] = pd.Series(results['drawdown'])
            
            logger.info(f"Results loaded from {load_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return {}

class NoThreading:
    """Context manager for disabling threading"""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
