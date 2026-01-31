from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict, Any

# Shared properties
class OHLCVBase(BaseModel):
    time: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

# Properties to return to client
class OHLCV(OHLCVBase):
    class Config:
        from_attributes = True

class MatchResult(BaseModel):
    date: str
    score: float
    prices: List[float]
    symbol: Optional[str] = None

class AnalysisResponse(BaseModel):
    extracted_pattern: List[float]
    matches: List[MatchResult]
    debug_image: Optional[str] = None

# Advanced schemas for detailed analysis

class DetailedMetrics(BaseModel):
    """Detailed similarity metrics for a match."""
    euclidean: float
    pearson: float
    cosine: float
    dtw: float
    cross_correlation: float
    shape: float

class AdvancedMatchResult(BaseModel):
    """Enhanced match result with detailed metrics."""
    start_index: int
    confidence: float
    date: str
    prices: List[float]
    symbol: str
    ensemble_score: float
    detailed_metrics: Dict[str, float]

class PatternStatistics(BaseModel):
    """Statistical summary of pattern matches."""
    n_matches: int
    mean_confidence: float
    std_confidence: float
    min_confidence: float
    max_confidence: float
    metric_averages: Optional[Dict[str, Dict[str, float]]] = None

class AdvancedAnalysisResponse(BaseModel):
    """Response from advanced pattern analysis."""
    extracted_pattern: List[float]
    debug_image: Optional[str]
    matches: List[AdvancedMatchResult]
    statistics: Optional[PatternStatistics] = None

class PatternSearchRequest(BaseModel):
    """Request for pattern search without image."""
    pattern: List[float]
    target_symbol: str = "BTC/USDT"
    method: str = "ensemble"
    top_k: int = 5

class PatternSearchResponse(BaseModel):
    """Response from pattern search."""
    matches: List[Dict[str, Any]]
