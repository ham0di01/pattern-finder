from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional
from datetime import datetime
import pandas as pd

from app.db.models.ohlcv import OHLCV
from app.schemas.analysis import OHLCVBase

class OHLCVRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_symbol_and_timeframe(self, symbol: str, start_date: Optional[datetime] = None) -> pd.DataFrame:
        query = self.db.query(OHLCV.time, OHLCV.close, OHLCV.open, OHLCV.high, OHLCV.low, OHLCV.volume)\
            .filter(OHLCV.symbol == symbol)\
            .order_by(OHLCV.time.asc())
            
        if start_date:
            query = query.filter(OHLCV.time >= start_date)
            
        df = pd.read_sql(query.statement, self.db.bind)
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'])
        return df

    def get_all_symbols(self) -> List[str]:
        """Get list of all distinct symbols in database."""
        result = self.db.query(OHLCV.symbol).distinct().order_by(OHLCV.symbol.asc()).all()
        return [r[0] for r in result]

    def bulk_insert(self, data: List[dict]):
        if not data:
            return

        stmt = text("""
            INSERT OR IGNORE INTO ohlcv (time, symbol, open, high, low, close, volume)
            VALUES (:time, :symbol, :open, :high, :low, :close, :volume)
        """)

        try:
            self.db.execute(stmt, data)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise e
