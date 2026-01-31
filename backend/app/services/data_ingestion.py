import ccxt
from datetime import datetime
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from app.db.repositories.ohlcv import OHLCVRepository

class IngestionService:
    def __init__(self, db: Session):
        self.repository = OHLCVRepository(db)
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 1000, since: Optional[int] = None) -> Tuple[List[dict], Optional[int]]:
        print(f"Fetching {symbol} ({timeframe})...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
        except Exception as e:
            print(f"Error fetching data from exchange: {e}")
            return [], None
        
        data = []
        if not ohlcv:
            return [], None

        for candle in ohlcv:
            # timestamp, open, high, low, close, volume
            data.append({
                "time": datetime.fromtimestamp(candle[0] / 1000.0),
                "symbol": symbol,
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5]
            })
        
        last_ts = ohlcv[-1][0] if ohlcv else None
        return data, last_ts

    def run_ingestion(self, symbols: List[str], timeframe: str = '1h', start_date: Optional[str] = None):
        for symbol in symbols:
            if start_date:
                since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                while True:
                    data, last_ts = self.fetch_ohlcv(symbol, timeframe, since=since)
                    if not data:
                        break
                    
                    self.repository.bulk_insert(data)
                    print(f"Stored {len(data)} records for {symbol}")
                    
                    since = last_ts + 1
                    # Stop if we are within the last minute (to avoid infinite loops on near-realtime)
                    if since > (datetime.now().timestamp() * 1000) - 60000:
                        break
            else:
                data, _ = self.fetch_ohlcv(symbol, timeframe)
                self.repository.bulk_insert(data)
                print(f"Stored {len(data)} records for {symbol}")
