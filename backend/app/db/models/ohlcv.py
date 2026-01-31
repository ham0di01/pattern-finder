from sqlalchemy import Column, DateTime, Float, String, UniqueConstraint
from app.db.base_class import Base

class OHLCV(Base):
    __tablename__ = "ohlcv"

    time = Column(DateTime, primary_key=True)
    symbol = Column(String, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        UniqueConstraint('time', 'symbol', name='uix_time_symbol'),
    )
