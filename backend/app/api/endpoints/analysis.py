from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Query, Body
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from app.api import dependencies
from app.schemas.analysis import AnalysisResponse, MatchResult, AdvancedMatchResult
from app.services.vision_processing import VisionService
from app.services.pattern_matcher import PatternMatcherService
from app.db.repositories.ohlcv import OHLCVRepository


class PatternSearchRequest(BaseModel):
    """Request model for pattern search."""
    pattern: List[float]
    target_symbol: str = "BTC/USDT"
    method: str = "ensemble"
    top_k: int = 5


router = APIRouter()


@router.get("/symbols")
async def get_available_symbols(db: Session = Depends(dependencies.get_db)):
    """Get list of all available trading symbols in database."""
    repo = OHLCVRepository(db)
    symbols = repo.get_all_symbols()
    return {"symbols": symbols}


@router.post("/analyze-image", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    mode: str = Form("GLOBAL"),
    method: str = Form("ensemble"),
    top_k: int = Form(5),
    db: Session = Depends(dependencies.get_db)
):
    """
    Analyze uploaded chart image and find similar patterns.

    Args:
        file: Image file of chart pattern
        mode: Search mode - "GLOBAL" for all symbols, or comma-separated list like "BTC/USDT,ETH/USDT,SOL/USDT"
        method: Matching method - 'ensemble' (recommended), 'euclidean', 'dtw', 'pearson', 'cosine'
        top_k: Number of results to return

    Returns:
        AnalysisResponse with extracted pattern and matches
    """
    try:
        contents = await file.read()
        extracted_pattern, debug_image = VisionService.process_image(contents)

        # Determine symbols to search
        if mode == "GLOBAL":
            repo = OHLCVRepository(db)
            target_symbols = repo.get_all_symbols()
        else:
            target_symbols = [s.strip() for s in mode.split(',')]

        matcher = PatternMatcherService(db, use_advanced=(method == 'ensemble'))

        # Search all symbols and aggregate matches
        all_matches = []
        for symbol in target_symbols:
            matches = matcher.find_matches(
                target_symbol=symbol,
                user_pattern=extracted_pattern,
                top_k=top_k,
                method=method
            )
            for start_idx, score, segment in matches:
                all_matches.append((score, symbol, start_idx, segment))

        # Sort by score and take top_k overall
        all_matches.sort(key=lambda x: x[0])
        all_matches = all_matches[:top_k]

        results = []
        for score, symbol, start_idx, segment in all_matches:
            results.append(MatchResult(
                date=str(segment['time'].iloc[0]),
                score=score,
                prices=segment['close'].tolist(),
                symbol=symbol
            ))

        return AnalysisResponse(
            extracted_pattern=extracted_pattern,
            matches=results,
            debug_image=debug_image
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error processing analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/analyze-pattern-advanced")
async def analyze_pattern_advanced(
    file: UploadFile = File(...),
    target_symbol: str = Form("BTC/USDT"),
    top_k: int = Form(5),
    min_quality: float = Form(0.0),
    db: Session = Depends(dependencies.get_db)
):
    """
    Advanced pattern analysis with detailed metrics and explanations.

    Uses the ensemble pattern recognition module with multiple similarity metrics
    and feature engineering for maximum accuracy.

    Returns:
        Detailed match results with confidence scores and metric breakdowns
    """
    try:
        contents = await file.read()
        extracted_pattern, debug_image = VisionService.process_image(contents)

        # Use advanced pattern matcher
        matcher = PatternMatcherService(db, use_advanced=True)
        result = matcher.get_matches_with_details(
            target_symbol=target_symbol,
            user_pattern=extracted_pattern,
            top_k=top_k
        )

        # Format response
        matches = result.get('matches', [])
        statistics = result.get('statistics', {})

        formatted_matches = []
        for match in matches:
            formatted_matches.append({
                'start_index': match['start_index'],
                'confidence': match['confidence'],
                'date': str(match['segment']['time'].iloc[0]),
                'prices': match['segment']['close'].tolist(),
                'symbol': match['symbol'],
                'ensemble_score': match['ensemble_score'],
                'detailed_metrics': match['detailed_metrics']
            })

        return {
            'extracted_pattern': extracted_pattern,
            'debug_image': debug_image,
            'matches': formatted_matches,
            'statistics': statistics
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error in advanced analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/find-patterns")
async def find_patterns(
    request: PatternSearchRequest,
    db: Session = Depends(dependencies.get_db)
):
    """
    Find patterns similar to the provided numeric pattern array.

    Args:
        request: PatternSearchRequest containing pattern and search parameters

    Returns:
        List of matching patterns with scores
    """
    try:
        matcher = PatternMatcherService(db, use_advanced=(request.method == 'ensemble'))

        matches = matcher.find_matches(
            target_symbol=request.target_symbol,
            user_pattern=request.pattern,
            top_k=request.top_k,
            method=request.method
        )

        results = []
        for start_idx, score, segment in matches:
            results.append({
                'start_index': start_idx,
                'score': score,
                'date': str(segment['time'].iloc[0]),
                'prices': segment['close'].tolist(),
                'symbol': request.target_symbol
            })

        return {'matches': results}

    except Exception as e:
        print(f"Error finding patterns: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
