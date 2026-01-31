import argparse
import sys
import os

# Ensure app is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.data_ingestion import IngestionService
from app.db.session import SessionLocal

def main():
    parser = argparse.ArgumentParser(description="Ingest crypto data")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol to fetch")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    db = SessionLocal()
    try:
        service = IngestionService(db)
        service.run_ingestion([args.symbol], args.timeframe, start_date=args.start_date)
    finally:
        db.close()

if __name__ == "__main__":
    main()