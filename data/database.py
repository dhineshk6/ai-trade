import sqlite3
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import DATABASE_URL, CURRENT_USER

Base = declarative_base()

class Trade(Base):
    """Trade database model"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    side = Column(String)  # 'long' or 'short'
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Float)
    pnl = Column(Float)
    pnl_percent = Column(Float)
    status = Column(String)  # 'open' or 'closed'
    stop_loss = Column(Float)
    take_profit = Column(Float)
    strategy = Column(String)
    user = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Position(Base):
    """Position database model"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    side = Column(String)
    entry_price = Column(Float)
    current_price = Column(Float)
    quantity = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    pnl = Column(Float)
    pnl_percent = Column(Float)
    status = Column(String)
    user = Column(String)
    updated_at = Column(DateTime)

class MarketData(Base):
    """Market data database model"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    timestamp = Column(DateTime)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self):
        """Initialize database connection"""
        self.engine = create_engine(DATABASE_URL)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def store_trade(self, trade_data: dict):
        """Store trade information"""
        trade = Trade(
            symbol=trade_data['symbol'],
            entry_time=trade_data['entry_time'],
            exit_time=trade_data.get('exit_time'),
            side=trade_data['side'],
            entry_price=trade_data['entry_price'],
            exit_price=trade_data.get('exit_price'),
            quantity=trade_data['quantity'],
            pnl=trade_data.get('pnl', 0),
            pnl_percent=trade_data.get('pnl_percent', 0),
            status=trade_data.get('status', 'open'),
            stop_loss=trade_data.get('stop_loss'),
            take_profit=trade_data.get('take_profit'),
            strategy=trade_data.get('strategy'),
            user=CURRENT_USER
        )
        
        try:
            self.session.add(trade)
            self.session.commit()
            return trade.id
        except Exception as e:
            self.session.rollback()
            raise e

    def update_trade(self, trade_id: int, update_data: dict):
        """Update existing trade"""
        try:
            trade = self.session.query(Trade).filter_by(id=trade_id).first()
            if trade:
                for key, value in update_data.items():
                    setattr(trade, key, value)
                self.session.commit()
                return True
            return False
        except Exception as e:
            self.session.rollback()
            raise e

    def store_position(self, position_data: dict):
        """Store position information"""
        position = Position(
            symbol=position_data['symbol'],
            side=position_data['side'],
            entry_price=position_data['entry_price'],
            current_price=position_data['current_price'],
            quantity=position_data['quantity'],
            stop_loss=position_data.get('stop_loss'),
            take_profit=position_data.get('take_profit'),
            pnl=position_data.get('pnl', 0),
            pnl_percent=position_data.get('pnl_percent', 0),
            status=position_data.get('status', 'open'),
            user=CURRENT_USER,
            updated_at=datetime.utcnow()
        )
        
        try:
            self.session.add(position)
            self.session.commit()
            return position.id
        except Exception as e:
            self.session.rollback()
            raise e

    def store_market_data(self, market_data: dict):
        """Store market data"""
        data = MarketData(
            symbol=market_data['symbol'],
            timestamp=market_data['timestamp'],
            open=market_data['open'],
            high=market_data['high'],
            low=market_data['low'],
            close=market_data['close'],
            volume=market_data['volume']
        )
        
        try:
            self.session.add(data)
            self.session.commit()
            return data.id
        except Exception as e:
            self.session.rollback()
            raise e

    def get_trades(self, start_time=None, end_time=None, symbol=None, status=None):
        """Get trades with optional filters"""
        query = self.session.query(Trade)
        
        if start_time:
            query = query.filter(Trade.entry_time >= start_time)
        if end_time:
            query = query.filter(Trade.entry_time <= end_time)
        if symbol:
            query = query.filter(Trade.symbol == symbol)
        if status:
            query = query.filter(Trade.status == status)
            
        return query.all()

    def get_positions(self, symbol=None, status='open'):
        """Get positions with optional filters"""
        query = self.session.query(Position)
        
        if symbol:
            query = query.filter(Position.symbol == symbol)
        if status:
            query = query.filter(Position.status == status)
            
        return query.all()

    def get_market_data(self, symbol: str, start_time=None, end_time=None):
        """Get market data with optional time range"""
        query = self.session.query(MarketData).filter(MarketData.symbol == symbol)
        
        if start_time:
            query = query.filter(MarketData.timestamp >= start_time)
        if end_time:
            query = query.filter(MarketData.timestamp <= end_time)
            
        return pd.read_sql(query.statement, self.session.bind)

    def close(self):
        """Close database session"""
        self.session.close()