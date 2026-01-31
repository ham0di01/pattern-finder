"""
Centralized configuration for the Pattern Recognition Module.
Allows easy tuning of accuracy and performance parameters.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PatternRecognitionConfig:
    """
    Configuration for Pattern Recognition Module.

    Attributes:
        # Performance Settings
        use_caching: Enable result caching for faster repeated queries
        cache_ttl: Time-to-live for cache entries (not yet implemented)
        initial_sample_rate: Stride for initial search (1 = check every window)
        initial_threshold: Quick filter threshold (lower = faster but may miss matches)

        # Pattern Preprocessing
        smooth_pattern: Apply smoothing kernel to reduce noise
        smoothing_window: Size of smoothing kernel (adaptive if None)
        detrend_pattern: Remove linear trend before matching

        # Feature Engineering
        extract_technical_indicators: Include technical indicator features
        extract_shape_features: Include shape-based features
        extract_frequency_features: Include frequency-domain features

        # Similarity Metrics
        metric_weights: Weights for ensemble combination (must sum to 1.0)
        enable_dtw: Enable Dynamic Time Warping (slower but more accurate)
        enable_cross_correlation: Enable cross-correlation matching

        # Quality Control
        min_pattern_quality: Minimum quality score to return results
        max_results: Maximum number of results to return
        require_minimum_variance: Skip windows with very low variance
        minimum_variance_threshold: Minimum std dev for valid window

        # Advanced Options
        ensemble_mode: 'weighted', 'voting', or 'stacking'
        feature_weight: Weight for feature similarity (vs pattern similarity)
        normalize_patterns: Normalize patterns before comparison
        outlier_detection: Use Isolation Forest to filter outlier matches
    """

    # Performance Settings
    use_caching: bool = True
    cache_ttl: int = 3600
    initial_sample_rate: int = 1
    initial_threshold: float = 2.0

    # Pattern Preprocessing
    smooth_pattern: bool = True
    smoothing_window: Optional[int] = None
    detrend_pattern: bool = False

    # Feature Engineering
    extract_technical_indicators: bool = True
    extract_shape_features: bool = True
    extract_frequency_features: bool = True

    # Similarity Metrics
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        'euclidean': 0.15,
        'pearson': 0.25,
        'cosine': 0.15,
        'dtw': 0.25,
        'cross_correlation': 0.10,
        'shape': 0.10
    })
    enable_dtw: bool = True
    enable_cross_correlation: bool = True

    # Quality Control
    min_pattern_quality: float = 0.0
    max_results: int = 10
    require_minimum_variance: bool = True
    minimum_variance_threshold: float = 1e-8

    # Advanced Options
    ensemble_mode: str = 'weighted'
    feature_weight: float = 0.15
    normalize_patterns: bool = True
    outlier_detection: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate metric weights sum to 1
        total_weight = sum(self.metric_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.metric_weights = {
                k: v / total_weight
                for k, v in self.metric_weights.items()
            }

        # Validate ensemble mode
        valid_modes = ['weighted', 'voting', 'stacking']
        if self.ensemble_mode not in valid_modes:
            raise ValueError(f"ensemble_mode must be one of {valid_modes}")

        # Validate feature weight
        if not 0 <= self.feature_weight <= 1:
            raise ValueError("feature_weight must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'use_caching': self.use_caching,
            'initial_sample_rate': self.initial_sample_rate,
            'initial_threshold': self.initial_threshold,
            'smooth_pattern': self.smooth_pattern,
            'detrend_pattern': self.detrend_pattern,
            'extract_technical_indicators': self.extract_technical_indicators,
            'extract_shape_features': self.extract_shape_features,
            'extract_frequency_features': self.extract_frequency_features,
            'metric_weights': self.metric_weights,
            'enable_dtw': self.enable_dtw,
            'enable_cross_correlation': self.enable_cross_correlation,
            'min_pattern_quality': self.min_pattern_quality,
            'max_results': self.max_results,
            'require_minimum_variance': self.require_minimum_variance,
            'minimum_variance_threshold': self.minimum_variance_threshold,
            'ensemble_mode': self.ensemble_mode,
            'feature_weight': self.feature_weight,
            'normalize_patterns': self.normalize_patterns,
            'outlier_detection': self.outlier_detection
        }


# Preset configurations for different use cases

def get_high_accuracy_config() -> PatternRecognitionConfig:
    """
    Configuration optimized for maximum accuracy.
    Slower but finds the best matches.
    """
    return PatternRecognitionConfig(
        use_caching=True,
        initial_sample_rate=1,
        initial_threshold=3.0,
        smooth_pattern=True,
        extract_technical_indicators=True,
        extract_shape_features=True,
        extract_frequency_features=True,
        metric_weights={
            'euclidean': 0.10,
            'pearson': 0.25,
            'cosine': 0.15,
            'dtw': 0.30,  # Higher weight on DTW for accuracy
            'cross_correlation': 0.10,
            'shape': 0.10
        },
        enable_dtw=True,
        enable_cross_correlation=True,
        feature_weight=0.20,  # Higher feature weight
        min_pattern_quality=0.0
    )


def get_fast_config() -> PatternRecognitionConfig:
    """
    Configuration optimized for speed.
    Faster but may miss some subtle matches.
    """
    return PatternRecognitionConfig(
        use_caching=True,
        initial_sample_rate=2,  # Skip every other window
        initial_threshold=1.5,  # Stricter filter
        smooth_pattern=False,  # Skip preprocessing
        extract_technical_indicators=True,
        extract_shape_features=True,
        extract_frequency_features=False,  # Skip expensive FFT
        metric_weights={
            'euclidean': 0.30,
            'pearson': 0.30,
            'cosine': 0.20,
            'dtw': 0.0,  # Disable DTW for speed
            'cross_correlation': 0.10,
            'shape': 0.10
        },
        enable_dtw=False,
        enable_cross_correlation=True,
        feature_weight=0.10,
        min_pattern_quality=0.1
    )


def get_balanced_config() -> PatternRecognitionConfig:
    """
    Balanced configuration between accuracy and speed.
    Good default for most use cases.
    """
    return PatternRecognitionConfig()  # Use defaults


def get_trend_following_config() -> PatternRecognitionConfig:
    """
    Configuration optimized for finding trend-following patterns.
    Weighs trend-related features more heavily.
    """
    return PatternRecognitionConfig(
        metric_weights={
            'euclidean': 0.10,
            'pearson': 0.30,  # Higher weight on correlation
            'cosine': 0.15,
            'dtw': 0.25,
            'cross_correlation': 0.15,
            'shape': 0.05
        },
        extract_technical_indicators=True,
        extract_shape_features=True,
        extract_frequency_features=False,
        feature_weight=0.25
    )


def get_reversal_config() -> PatternRecognitionConfig:
    """
    Configuration optimized for finding reversal patterns.
    Focuses on shape and turning points.
    """
    return PatternRecognitionConfig(
        metric_weights={
            'euclidean': 0.15,
            'pearson': 0.15,
            'cosine': 0.15,
            'dtw': 0.20,
            'cross_correlation': 0.10,
            'shape': 0.25  # Higher weight on shape
        },
        extract_technical_indicators=True,
        extract_shape_features=True,
        extract_frequency_features=True,
        feature_weight=0.20,
        smooth_pattern=True
    )
