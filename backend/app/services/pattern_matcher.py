import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.db.repositories.ohlcv import OHLCVRepository
from app.core.config import settings
from app.services.pattern_recognition_module import PatternRecognitionModule


class PatternMatcherService:
    """
    Enhanced Pattern Matching Service with multiple matching algorithms.
    Provides backward compatibility while using advanced pattern recognition.
    """

    def __init__(self, db: Session, use_advanced: bool = True):
        """
        Initialize pattern matcher service.

        Args:
            db: Database session
            use_advanced: Whether to use the advanced pattern recognition module
        """
        self.repository = OHLCVRepository(db)
        self.use_advanced = use_advanced

        if use_advanced:
            # Initialize advanced pattern recognition module
            config = {
                'use_caching': True,
                'smooth_pattern': True,
                'initial_sample_rate': 1,
                'initial_threshold': 2.0,
                'metric_weights': {
                    'euclidean': 0.15,
                    'pearson': 0.25,
                    'cosine': 0.15,
                    'dtw': 0.25,
                    'cross_correlation': 0.10,
                    'shape': 0.10
                }
            }
            self.advanced_module = PatternRecognitionModule(db, config=config)

    def normalize(self, window: np.ndarray) -> np.ndarray:
        """Normalize window to zero mean and unit variance."""
        std = np.std(window)
        if std == 0:
            return np.zeros_like(window)
        return (window - np.mean(window)) / (std + 1e-8)

    def find_matches(self,
                     target_symbol: str,
                     user_pattern: List[float],
                     top_k: int = 5,
                     method: str = 'ensemble') -> List[Tuple[int, float, pd.DataFrame]]:
        """
        Find pattern matches using specified method.

        Args:
            target_symbol: Symbol to search in
            user_pattern: Pattern to match
            top_k: Number of results to return
            method: 'ensemble' (advanced), 'euclidean', 'dtw', 'pearson', 'cosine'

        Returns:
            List of (start_index, score, dataframe) tuples
        """
        if method == 'ensemble' and self.use_advanced:
            return self._find_matches_advanced(target_symbol, user_pattern, top_k)
        else:
            return self._find_matches_basic(target_symbol, user_pattern, top_k, method)

    def _find_matches_advanced(self,
                               target_symbol: str,
                               user_pattern: List[float],
                               top_k: int = 5) -> List[Tuple[int, float, pd.DataFrame]]:
        """Use advanced pattern recognition module."""
        matches = self.advanced_module.find_similar_patterns(
            target_symbol=target_symbol,
            query_pattern=user_pattern,
            top_k=top_k,
            min_quality=0.0
        )

        # Convert to legacy format
        results = []
        for match in matches:
            results.append((
                match['start_index'],
                match['ensemble_score'],
                match['segment']
            ))

        return results

    def _find_matches_basic(self,
                            target_symbol: str,
                            user_pattern: List[float],
                            top_k: int = 5,
                            method: str = 'euclidean') -> List[Tuple[int, float, pd.DataFrame]]:
        """
        Basic pattern matching using specified distance metric.
        Provides fallback compatibility and single-metric options.
        """
        df = self.repository.get_by_symbol_and_timeframe(target_symbol)
        if df.empty:
            return []

        history_values = df['close'].values
        query_pattern = np.array(user_pattern)
        query_norm = self.normalize(query_pattern)

        n = len(query_pattern)
        future_candles = settings.FUTURE_CANDLES

        distances = []
        indices = []

        # Sliding window search
        limit = len(history_values) - n - future_candles

        for i in range(limit):
            window = history_values[i:i+n]

            # Skip flat lines (no variance)
            if np.std(window) == 0:
                continue

            window_norm = self.normalize(window)

            # Compute distance based on method
            if method == 'euclidean':
                dist = np.linalg.norm(query_norm - window_norm)
            elif method == 'cosine':
                from scipy.spatial.distance import cosine
                dist = cosine(query_norm, window_norm)
            elif method == 'pearson':
                from scipy import stats
                correlation, _ = stats.pearsonr(query_norm, window_norm)
                dist = 1 - abs(correlation)
            elif method == 'dtw':
                from fastdtw import fastdtw
                dist, _ = fastdtw(query_norm.reshape(-1, 1), window_norm.reshape(-1, 1))
            else:
                # Default to Euclidean
                dist = np.linalg.norm(query_norm - window_norm)

            distances.append(dist)
            indices.append(i)

        if not distances:
            return []

        distances = np.array(distances)
        indices = np.array(indices)

        # Sort by distance (ascending)
        sorted_idx = np.argsort(distances)[:top_k]

        results = []
        for idx in indices[sorted_idx]:
            pos = np.where(indices == idx)[0][0]
            dist_score = distances[pos]

            start_idx = int(idx)
            end_idx = min(start_idx + n + future_candles, len(df))

            segment = df.iloc[start_idx:end_idx]
            results.append((start_idx, float(dist_score), segment))

        return results

    def get_matches_with_details(self,
                                  target_symbol: str,
                                  user_pattern: List[float],
                                  top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get matches with detailed metrics and explanations.

        Args:
            target_symbol: Symbol to search in
            user_pattern: Pattern to match
            top_k: Number of results

        Returns:
            List of detailed match dictionaries
        """
        if not self.use_advanced:
            raise ValueError("Detailed matches require advanced module. Initialize with use_advanced=True")

        matches = self.advanced_module.find_similar_patterns(
            target_symbol=target_symbol,
            query_pattern=user_pattern,
            top_k=top_k,
            min_quality=0.0
        )

        # Add statistics
        stats = self.advanced_module.get_pattern_statistics(matches)

        return {
            'matches': matches,
            'statistics': stats
        }

    def set_metric_weights(self, weights: Dict[str, float]) -> None:
        """
        Customize metric weights for ensemble matching.

        Args:
            weights: Dictionary of metric name -> weight
        """
        if not self.use_advanced:
            raise ValueError("Metric weights require advanced module")

        self.advanced_module.set_metric_weights(weights)
