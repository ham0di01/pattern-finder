# Pattern Finder

A web application for detecting and matching trading chart patterns using computer vision and historical market data. Upload a chart image to extract patterns and find similar historical occurrences across cryptocurrency markets.

## Features

- **Image Pattern Recognition**: Upload chart images and automatically extract price patterns using computer vision
- **Historical Pattern Matching**: Search through historical cryptocurrency data to find similar patterns
- **Interactive Charts**: Visualize extracted patterns and historical matches using Lightweight Charts
- **Multiple Search Modes**: Search globally across all symbols or select specific cryptocurrencies
- **Real-time Analysis**: Fast pattern matching with similarity scoring

## Prerequisites

- Python 3.9+
- Node.js 18+

## Quick Start

### 1. Start the Application

Run the provided startup script:

```bash
./start.sh
```

This will automatically:
- Set up the Python virtual environment
- Install all backend and frontend dependencies
- Start the backend server on port 8000
- Start the frontend development server on port 3000

Access the application at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000/api/v1/docs

Press `Ctrl+C` to stop both servers.

### 2. Populate Data (First Time Only)

Before using the application, you need to fetch historical cryptocurrency data. In a new terminal:

```bash
cd backend
./venv/bin/python scripts/fetch_data.py
```

This will fetch hourly OHLCV data from 2017 to present for 20 popular cryptocurrencies:
- BTC, ETH, BNB, XRP, SOL, ADA, DOGE, AVAX, DOT, MATIC
- LINK, UNI, LTC, ATOM, XLM, ALGO, VET, FIL, ICP, NEAR

### 3. Update Data (Optional)

To update the database with the latest data:

```bash
cd backend
./venv/bin/python scripts/update_data.py
```

## Usage

1. **Open the Application**: Navigate to http://localhost:3000

2. **Upload a Chart Image**:
   - Click the upload area or drag and drop a chart image
   - Supported formats: PNG, JPG, JPEG

3. **Select Search Mode**:
   - **Global Mode**: Search across all available symbols (default)
   - **Custom Mode**: Select specific cryptocurrencies to search

4. **Analyze**: Click the "Analyze Pattern" button to:
   - Extract the price pattern from your image
   - Search for similar patterns in historical data
   - Display matches ranked by similarity score

5. **Review Results**:
   - View the extracted pattern in the main chart
   - Browse historical matches with similarity scores
   - Click on any match to view the full chart visualization

## Project Structure

```
pattern-finder/
├── backend/
│   ├── app/
│   │   ├── api/          # API endpoints
│   │   ├── core/         # Configuration
│   │   ├── db/           # Database models and repositories
│   │   ├── schemas/      # Pydantic schemas
│   │   └── services/     # Business logic
│   ├── scripts/          # Data ingestion scripts
│   └── data/             # SQLite database
├── frontend/
│   ├── app/              # Next.js app directory
│   └── components/       # React components
└── start.sh              # Startup script
```

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /api/v1/symbols` - Get all available symbols
- `POST /api/v1/analyze-image` - Analyze uploaded chart image
- `GET /api/v1/docs` - Interactive API documentation (Swagger UI)

## Troubleshooting

### Port Already in Use

If ports 8000 or 3000 are already in use:

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Database Issues

If you encounter database issues, delete the database and re-populate:

```bash
rm backend/data/patterns.db
cd backend
./venv/bin/python scripts/fetch_data.py
```

### TA-Lib Installation Issues (macOS)

```bash
brew install ta-lib
```

Then re-run `./start.sh`

## License

This project is provided as-is for educational and research purposes.
