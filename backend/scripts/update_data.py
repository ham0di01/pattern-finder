import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.data_ingestion import IngestionService
from app.db.session import SessionLocal
from app.db.repositories.ohlcv import OHLCVRepository

def get_latest_timestamp_for_symbol(db, symbol: str) -> datetime:
    """Get the latest timestamp for a given symbol"""
    from app.db.models.ohlcv import OHLCV
    from sqlalchemy import func

    latest = db.query(func.max(OHLCV.time)).filter(OHLCV.symbol == symbol).first()
    return latest[0] if latest and latest[0] else None

def main():
    db = SessionLocal()
    try:
        # Get all symbols in database
        repo = OHLCVRepository(db)
        symbols = repo.get_all_symbols()

        if not symbols:
            print("No symbols found in database. Run populate_top_coins.py first.")
            return

        print(f"Found {len(symbols)} symbols in database:")
        for symbol in symbols:
            latest = get_latest_timestamp_for_symbol(db, symbol)
            if latest:
                print(f"  - {symbol}: Last data point at {latest}")
            else:
                print(f"  - {symbol}: No data")

        print(f"\nUpdating data from {datetime.now().isoformat()}...")

        service = IngestionService(db)

        for symbol in symbols:
            latest = get_latest_timestamp_for_symbol(db, symbol)

            if latest:
                # Calculate 'since' parameter (start from the last timestamp + 1 hour)
                since = int(latest.timestamp() * 1000) + 3600000  # +1 hour in milliseconds
                print(f"\nFetching new data for {symbol} since {latest}...")

                # Fetch and insert new data (duplicates will be ignored)
                data, last_ts = service.fetch_ohlcv(symbol, timeframe='1h', since=since)
                if data:
                    repo.bulk_insert(data)
                    print(f"  ✓ Added {len(data)} new records for {symbol}")
                else:
                    print(f"  - No new data available for {symbol}")
            else:
                print(f"\n⚠ {symbol} has no historical data. Skipping...")

        print("\n✓ Update complete!")

    except Exception as e:
        print(f"\n✗ Error during update: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    main()
