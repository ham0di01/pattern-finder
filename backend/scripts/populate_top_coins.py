import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.data_ingestion import IngestionService
from app.db.session import SessionLocal

def get_hardcoded_coins():
    """Hardcoded list of popular coins (excludes stablecoins)"""
    return [
        'BTC/USDT',   # Bitcoin
        'ETH/USDT',   # Ethereum
        'BNB/USDT',   # Binance Coin
        'XRP/USDT',   # Ripple
        'SOL/USDT',   # Solana
        'ADA/USDT',   # Cardano
        'DOGE/USDT',  # Dogecoin
        'AVAX/USDT',  # Avalanche
        'DOT/USDT',   # Polkadot
        'MATIC/USDT', # Polygon
        'LINK/USDT',  # Chainlink
        'UNI/USDT',   # Uniswap
        'LTC/USDT',   # Litecoin
        'ATOM/USDT',  # Cosmos
        'XLM/USDT',   # Stellar
        'ALGO/USDT',  # Algorand
        'VET/USDT',   # VeChain
        'FIL/USDT',   # Filecoin
        'ICP/USDT',   # Internet Computer
        'NEAR/USDT',  # NEAR Protocol
    ]

def main():
    coins = get_hardcoded_coins()

    print(f"\n{len(coins)} hardcoded coins to ingest:")
    for i, coin in enumerate(coins, 1):
        print(f"{i}. {coin}")

    print(f"\nStarting ingestion from 2017-01-01 to today...")

    db = SessionLocal()
    try:
        service = IngestionService(db)
        # Fetch historical data from 2017-01-01 to today
        service.run_ingestion(coins, timeframe='1h', start_date='2017-01-01')
        print("\n✓ Ingestion complete!")
    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
