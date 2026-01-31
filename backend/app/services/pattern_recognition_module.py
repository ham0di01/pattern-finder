"""
Production-grade pattern matching system with ensemble methods,
multiple distance metrics, and feature engineering for maximum accuracy.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional, Any
from sqlalchemy.orm import Session
from scipy import stats, integrate
from scipy.spatial.distance import cosine
from scipy.signal import correlate, find_peaks
from scipy.fft import fft, fftfreq
from fastdtw import fastdtw
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from numpy.lib.stride_tricks import sliding_window_view
import warnings
warnings.filterwarnings('ignore')

from app.db.repositories.ohlcv import OHLCVRepository
from app.core.config import settings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for time-series patterns.
    Extracts meaningful features to improve pattern matching accuracy.
    """

    @staticmethod
    def extract_technical_indicators(prices: np.ndarray) -> Dict[str, float]:
        """Extract technical indicators from price series."""
        features = {}

        # Basic statistics
        features['mean'] = np.mean(prices)
        features['std'] = np.std(prices)
        features['min'] = np.min(prices)
        features['max'] = np.max(prices)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(prices)

        # Returns-based features
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        features['return_mean'] = np.mean(returns)
        features['return_std'] = np.std(returns)
        features['return_skew'] = stats.skew(returns)
        features['return_kurtosis'] = stats.kurtosis(returns)

        # Trend indicators
        features['trend_slope'] = np.polyfit(range(len(prices)), prices, 1)[0]
        features['momentum'] = prices[-1] - prices[0]

        # Volatility
        features['volatility'] = np.std(returns) * np.sqrt(len(prices))

        # Price acceleration
        if len(prices) >= 3:
            acceleration = np.diff(returns)
            features['acceleration'] = np.mean(acceleration)

        # Relative strength
        if len(prices) > 10:
            sma_short = np.mean(prices[-5:])
            sma_long = np.mean(prices)
            features['rs_ratio'] = sma_short / (sma_long + 1e-8)

        return features

    @staticmethod
    def extract_shape_features(prices: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features describing pattern geometry."""
        features = {}

        # Normalize to [0, 1] for shape analysis
        normalized = (prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8)

        # Shape descriptors - use scipy.integrate.trapezoid instead of np.trapz
        features['area_under_curve'] = float(integrate.trapezoid(normalized))
        features['convexity'] = np.sum(np.diff(normalized, 2) > 0) / len(normalized)

        # Peak and trough analysis
        peaks, _ = find_peaks(normalized, height=0.5)
        troughs, _ = find_peaks(-normalized, height=0.5)
        features['n_peaks'] = len(peaks)
        features['n_troughs'] = len(troughs)
        features['peak_trough_ratio'] = len(peaks) / (len(troughs) + 1e-8)

        # Direction changes
        direction_changes = np.sum(np.diff(np.sign(np.diff(normalized))) != 0)
        features['direction_changes'] = direction_changes

        # Linearity (R² of linear fit)
        slope, intercept = np.polyfit(range(len(normalized)), normalized, 1)
        predicted = slope * np.arange(len(normalized)) + intercept
        ss_res = np.sum((normalized - predicted) ** 2)
        ss_tot = np.sum((normalized - np.mean(normalized)) ** 2)
        features['linearity_r2'] = 1 - (ss_res / (ss_tot + 1e-8))

        return features

    @staticmethod
    def extract_frequency_features(prices: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain features using FFT."""
        # Detrend the signal
        detrended = prices - np.polyfit(range(len(prices)), prices, 1)[0] * np.arange(len(prices))

        # Apply FFT
        fft_vals = fft(detrended)
        fft_freq = fftfreq(len(prices))

        # Power spectrum
        power = np.abs(fft_vals) ** 2

        # Dominant frequency
        pos_freq_idx = fft_freq > 0
        dominant_freq_idx = np.argmax(power[pos_freq_idx])
        features = {
            'dominant_frequency': float(fft_freq[pos_freq_idx][dominant_freq_idx]),
            'dominant_power': float(power[pos_freq_idx][dominant_freq_idx]),
            'total_power': float(np.sum(power[pos_freq_idx])),
            'spectral_centroid': float(np.average(fft_freq[pos_freq_idx], weights=power[pos_freq_idx]))
        }

        return features


class SimilarityMetrics:
    """
    Comprehensive similarity metrics for pattern matching.
    Uses multiple algorithms to capture different aspects of similarity.
    """

    @staticmethod
    def euclidean_distance(query: np.ndarray, candidate: np.ndarray) -> float:
        """Standard Euclidean distance (L2 norm)."""
        query_norm = (query - np.mean(query)) / (np.std(query) + 1e-8)
        candidate_norm = (candidate - np.mean(candidate)) / (np.std(candidate) + 1e-8)
        return float(np.linalg.norm(query_norm - candidate_norm))

    @staticmethod
    def pearson_correlation(query: np.ndarray, candidate: np.ndarray) -> float:
        """
        Pearson correlation coefficient.
        Returns 1 - correlation so lower = more similar.
        """
        correlation, _ = stats.pearsonr(query, candidate)
        return float(1 - abs(correlation))  # Absolute because inverse pattern also useful

    @staticmethod
    def cosine_similarity(query: np.ndarray, candidate: np.ndarray) -> float:
        """Cosine similarity (1 - cosine distance)."""
        return float(cosine(query, candidate))

    @staticmethod
    def dynamic_time_warping(query: np.ndarray, candidate: np.ndarray) -> float:
        """
        Dynamic Time Warping distance.
        Excellent for time series with phase differences or temporal distortions.
        """
        query_norm = (query - np.mean(query)) / (np.std(query) + 1e-8)
        candidate_norm = (candidate - np.mean(candidate)) / (np.std(candidate) + 1e-8)
        distance, _ = fastdtw(query_norm.reshape(-1, 1), candidate_norm.reshape(-1, 1))
        return float(distance)

    @staticmethod
    def cross_correlation(query: np.ndarray, candidate: np.ndarray) -> float:
        """
        Maximum cross-correlation.
        Finds best alignment and measures similarity.
        """
        correlation = correlate(query - np.mean(query), candidate - np.mean(candidate), mode='valid')
        max_corr = np.max(correlation)
        # Normalize
        norm_query = np.linalg.norm(query - np.mean(query))
        norm_candidate = np.linalg.norm(candidate - np.mean(candidate))
        if norm_query * norm_candidate == 0:
            return 1.0
        return float(1 - abs(max_corr / (norm_query * norm_candidate)))

    @staticmethod
    def shape_distance(query: np.ndarray, candidate: np.ndarray) -> float:
        """
        Shape-based distance focusing on relative movements.
        Uses first-order differences (momentum pattern).
        """
        query_diff = np.diff(query)
        candidate_diff = np.diff(candidate)
        query_diff_norm = (query_diff - np.mean(query_diff)) / (np.std(query_diff) + 1e-8)
        candidate_diff_norm = (candidate_diff - np.mean(candidate_diff)) / (np.std(candidate_diff) + 1e-8)
        return float(np.linalg.norm(query_diff_norm - candidate_diff_norm))

    @staticmethod
    def compute_all_metrics(query: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
        """Compute all similarity metrics between two patterns."""
        return {
            'euclidean': SimilarityMetrics.euclidean_distance(query, candidate),
            'pearson': SimilarityMetrics.pearson_correlation(query, candidate),
            'cosine': SimilarityMetrics.cosine_similarity(query, candidate),
            'dtw': SimilarityMetrics.dynamic_time_warping(query, candidate),
            'cross_correlation': SimilarityMetrics.cross_correlation(query, candidate),
            'shape': SimilarityMetrics.shape_distance(query, candidate)
        }


class EnsembleMatcher:
    """
    Ensemble-based pattern matching combining multiple metrics and features.
    Uses weighted voting and feature similarity for superior accuracy.
    """

    # Default weights for different metrics (can be tuned with validation data)
    DEFAULT_WEIGHTS = {
        'euclidean': 0.15,
        'pearson': 0.25,
        'cosine': 0.15,
        'dtw': 0.25,
        'cross_correlation': 0.10,
        'shape': 0.10
    }

    # Weights for feature similarity
    FEATURE_WEIGHT = 0.15
    PATTERN_WEIGHT = 0.85

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble matcher.

        Args:
            custom_weights: Optional custom weights for metrics
        """
        self.weights = custom_weights or self.DEFAULT_WEIGHTS
        self.feature_engineer = FeatureEngineer()

    def normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range."""
        min_val = min(scores.values())
        max_val = max(scores.values())
        if max_val - min_val == 0:
            return {k: 0.5 for k in scores}
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    def compute_feature_similarity(self, query_features: Dict[str, float],
                                  candidate_features: Dict[str, float]) -> float:
        """Compute similarity between feature dictionaries."""
        common_keys = set(query_features.keys()) & set(candidate_features.keys())
        if not common_keys:
            return 0.5

        similarities = []
        for key in common_keys:
            q_val = query_features[key]
            c_val = candidate_features[key]

            # Normalize by range
            max_val = max(abs(q_val), abs(c_val)) + 1e-8
            sim = 1 - abs(q_val - c_val) / max_val
            similarities.append(sim)

        return float(np.mean(similarities))

    def ensemble_score(self, query: np.ndarray, candidate: np.ndarray,
                      query_features: Dict[str, float],
                      candidate_features: Dict[str, float]) -> float:
        """
        Compute ensemble score combining pattern and feature similarity.

        Lower score = more similar (0 = perfect match, 1 = very different)
        """
        # Get all pattern metrics
        pattern_metrics = SimilarityMetrics.compute_all_metrics(query, candidate)

        # Normalize pattern scores
        normalized_metrics = self.normalize_scores(pattern_metrics)

        # Compute weighted pattern score
        weighted_pattern_score = sum(
            normalized_metrics[metric] * weight
            for metric, weight in self.weights.items()
        )

        # Compute feature similarity
        feature_sim = self.compute_feature_similarity(query_features, candidate_features)

        # Combine pattern and feature scores
        ensemble_score = (
            self.PATTERN_WEIGHT * weighted_pattern_score +
            self.FEATURE_WEIGHT * (1 - feature_sim)
        )

        return float(ensemble_score)


class PatternRecognitionModule:
    """
    Main Pattern Recognition Module with advanced algorithms.
    Provides highly accurate pattern matching for financial time series.
    """

    def __init__(self, db: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Pattern Recognition Module.

        Args:
            db: Database session
            config: Optional configuration dictionary
        """
        self.repository = OHLCVRepository(db)
        self.config = config or {}
        self.feature_engineer = FeatureEngineer()
        self.ensemble_matcher = EnsembleMatcher(
            custom_weights=self.config.get('metric_weights')
        )

        # Performance optimization: precompute indices
        self._use_caching = self.config.get('use_caching', True)
        self._pattern_cache = {}

    def preprocess_pattern(self, pattern: List[float]) -> np.ndarray:
        """
        Preprocess pattern for matching.

        Args:
            pattern: Raw pattern values

        Returns:
            Preprocessed numpy array
        """
        pattern_arr = np.array(pattern)

        # Remove NaN/Inf values
        pattern_arr = pattern_arr[~np.isnan(pattern_arr)]
        pattern_arr = pattern_arr[~np.isinf(pattern_arr)]

        # Smooth the pattern (optional, reduces noise)
        if self.config.get('smooth_pattern', True):
            window_size = min(5, len(pattern_arr) // 4)
            if window_size >= 2:
                kernel = np.ones(window_size) / window_size
                pattern_arr = np.convolve(pattern_arr, kernel, mode='same')

        return pattern_arr

    def extract_pattern_features(self, pattern: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive features from pattern.

        Args:
            pattern: Pattern array

        Returns:
            Feature dictionary
        """
        features = {}
        features.update(self.feature_engineer.extract_technical_indicators(pattern))
        features.update(self.feature_engineer.extract_shape_features(pattern))
        features.update(self.feature_engineer.extract_frequency_features(pattern))
        return features

    def find_similar_patterns(self,
                             target_symbol: str,
                             query_pattern: List[float],
                             top_k: int = 5,
                             min_quality: float = 0.0) -> List[Dict[str, Any]]:
        """
        Find patterns similar to the query pattern using ensemble matching.

        Args:
            target_symbol: Symbol to search in (e.g., 'BTC/USDT')
            query_pattern: User-provided pattern to match
            top_k: Number of top results to return
            min_quality: Minimum quality threshold (0-1)

        Returns:
            List of matches with detailed scores
        """
        # Preprocess query pattern
        query_arr = self.preprocess_pattern(query_pattern)
        query_features = self.extract_pattern_features(query_arr)

        # Fetch historical data
        df = self.repository.get_by_symbol_and_timeframe(target_symbol)
        if df.empty:
            return []

        history_values = df['close'].values
        n = len(query_arr)
        future_candles = settings.FUTURE_CANDLES

        # Vectorized sliding window for initial filtering
        candidates = self._vectorized_search(history_values, query_arr, n, future_candles)

        if not candidates:
            return []

        # Detailed scoring for candidates
        results = []
        for idx, window in candidates:
            try:
                candidate_features = self.extract_pattern_features(window)

                # Compute ensemble score
                ensemble_score = self.ensemble_matcher.ensemble_score(
                    query_arr, window, query_features, candidate_features
                )

                # Get individual metrics for transparency
                metrics = SimilarityMetrics.compute_all_metrics(query_arr, window)

                results.append({
                    'start_index': int(idx),
                    'ensemble_score': ensemble_score,
                    'metrics': metrics,
                    'window': window
                })

            except Exception as e:
                # Log error and skip problematic windows
                logger.warning(f"Error processing window at index {idx}: {e}")
                continue

        # Sort by ensemble score (ascending - lower is better)
        results.sort(key=lambda x: x['ensemble_score'])

        # Filter by quality and return top_k
        quality_threshold = min_quality
        top_results = [r for r in results if r['ensemble_score'] <= (1 - quality_threshold)][:top_k]

        # Format results for API
        formatted_results = []
        for result in top_results:
            start_idx = result['start_index']
            end_idx = min(start_idx + n + future_candles, len(df))
            segment = df.iloc[start_idx:end_idx]

            # Convert score to confidence (1 - score, inverted so higher = better)
            confidence = max(0, 1 - result['ensemble_score'])

            formatted_results.append({
                'start_index': start_idx,
                'confidence': float(confidence),
                'ensemble_score': float(result['ensemble_score']),
                'detailed_metrics': {k: float(v) for k, v in result['metrics'].items()},
                'segment': segment,
                'symbol': target_symbol
            })

        return formatted_results

    def _vectorized_search(self, history: np.ndarray, query: np.ndarray,
                          pattern_length: int, future_candles: int) -> List[Tuple[int, np.ndarray]]:
        """
        Fast vectorized search to find candidate matches.
        Uses efficient sliding window with early filtering.

        Args:
            history: Historical price data
            query: Query pattern
            pattern_length: Length of pattern
            future_candles: Number of future candles to include

        Returns:
            List of (index, window) tuples for candidates
        """
        n = len(query)
        limit = len(history) - n - future_candles

        if limit <= 0:
            return []

        # Normalize query
        query_norm = (query - np.mean(query)) / (np.std(query) + 1e-8)

        # Create sliding windows view (no memory copy)
        # shape: (len(history) - n + 1, n)
        all_windows = sliding_window_view(history, window_shape=n)
        
        # Truncate to limit (to account for future candles)
        windows = all_windows[:limit]

        # Calculate standard deviation for all windows
        # axis=1 computes std across the window dimension
        stds = np.std(windows, axis=1, keepdims=True)
        
        # Filter out flat windows (std ~ 0) to avoid division by zero
        # We'll use a mask but proceed with calculation to keep vectorization simple
        valid_mask = (stds > 1e-8).flatten()
        
        # Normalize all windows in one go: (window - mean) / std
        means = np.mean(windows, axis=1, keepdims=True)
        
        # Add epsilon to stds to prevent division by zero for invalid windows
        windows_norm = (windows - means) / (stds + 1e-8)
        
        # Calculate Euclidean distance for all windows against query
        # broadcasting query_norm across all windows
        # axis=1 sums the squared differences for each window
        distances = np.linalg.norm(windows_norm - query_norm, axis=1)
        
        # Apply threshold and valid mask
        threshold = self.config.get('initial_threshold', 2.0)
        
        # Indices where distance is low enough AND window is not flat
        candidate_indices = np.where((distances <= threshold) & valid_mask)[0]
        
        # Apply sample rate if needed (post-filter is cheaper now since calculation was vectorized)
        sample_rate = self.config.get('initial_sample_rate', 1)
        if sample_rate > 1:
            candidate_indices = candidate_indices[::sample_rate]
            
        # Construct result list
        candidates = []
        for idx in candidate_indices:
            candidates.append((int(idx), windows[idx]))
            
        return candidates

    def get_pattern_statistics(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about matched patterns.

        Args:
            matches: List of pattern matches

        Returns:
            Statistical summary
        """
        if not matches:
            return {}

        confidences = [m['confidence'] for m in matches]
        metrics_keys = matches[0]['detailed_metrics'].keys()

        stats = {
            'n_matches': len(matches),
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'metric_averages': {}
        }

        for metric in metrics_keys:
            values = [m['detailed_metrics'][metric] for m in matches]
            stats['metric_averages'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        return stats

    def set_metric_weights(self, weights: Dict[str, float]) -> None:
        """
        Update metric weights for ensemble matching.

        Args:
            weights: Dictionary mapping metric names to weights
        """
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            normalized_weights = {k: v / total for k, v in weights.items()}
            self.ensemble_matcher.weights = normalized_weights

    def enable_caching(self, enabled: bool) -> None:
        """
        Enable or disable result caching.

        Args:
            enabled: Whether to enable caching
        """
        self._use_caching = enabled
        if not enabled:
            self._pattern_cache.clear()
