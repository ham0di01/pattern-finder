import sys
import os
import ccxt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.data_ingestion import IngestionService
from app.db.session import SessionLocal

def get_top_20_coins():
    """Get top 20 coins by 24h volume from Binance"""
    exchange = ccxt.binance({'enableRateLimit': True})

    # Fetch all tickers at once (Binance returns all)
    tickers = exchange.fetch_tickers()

    # Filter for USDT pairs only and sort by volume
    usdt_tickers = {
        symbol: ticker for symbol, ticker in tickers.items()
        if symbol.endswith('/USDT')
    }

    # Sort by 24h volume and get top 20
    sorted_pairs = sorted(
        usdt_tickers.items(),
        key=lambda x: x[1].get('quoteVolume', 0),
        reverse=True
    )

    top_20 = [symbol for symbol, _ in sorted_pairs[:20]]
    return top_20

def main():
    print("Fetching top 20 coins by 24h volume...")
    top_20 = get_top_20_coins()

    print("\nTop 20 coins to ingest:")
    for i, coin in enumerate(top_20, 1):
        print(f"{i}. {coin}")

    print(f"\nStarting ingestion from 2017-01-01 to today...")

    db = SessionLocal()
    try:
        service = IngestionService(db)
        # Fetch historical data from 2017-01-01 to today
        service.run_ingestion(top_20, timeframe='1h', start_date='2017-01-01')
        print("\n✓ Ingestion complete!")
    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
