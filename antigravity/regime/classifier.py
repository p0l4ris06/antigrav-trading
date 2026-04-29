"""
Regime Classifier — PCA → GMM Unsupervised State Extraction.

Detects latent market regimes (trend, mean-reversion, volatility expansion)
via Gaussian Mixture Model with BIC-optimal component selection.

Autonomous Refitting:
    Tracks out-of-sample log-likelihood. If current LL drops below the
    5th percentile of historical distribution, triggers autonomous refit
    on trailing N periods.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import structlog
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from antigravity.config import settings

logger = structlog.get_logger(__name__)


class RegimeClassifier:
    """
    PCA-orthogonalized GMM regime classifier with autonomous refitting.

    Pipeline:
        1. StandardScaler → normalize feature scales
        2. PCA → retain 95% variance, orthogonalize input space
        3. GMM → soft cluster assignments via EM algorithm
        4. BIC grid search over [min_components, max_components]
    """

    def __init__(
        self,
        min_components: int = 2,
        max_components: int = 6,
        pca_variance: float = 0.95,
        ll_history_size: int = 500,
    ) -> None:
        self.min_components = min_components
        self.max_components = max_components
        self._pca_variance = pca_variance

        self._scaler = StandardScaler()
        self._pca = PCA(n_components=pca_variance)
        self._gmm: GaussianMixture | None = None
        self._optimal_n: int = 3
        self._is_fitted: bool = False

        # Log-likelihood tracking for autonomous refit decisions
        self._ll_history: deque[float] = deque(maxlen=ll_history_size)
        self._refit_percentile = settings.overseer.refit_percentile

        # Regime labels (for logging/dashboard)
        self._regime_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_components(self) -> int:
        return self._optimal_n

    def fit(self, feature_matrix: np.ndarray) -> dict[str, any]:
        """
        Fit the full pipeline: Scaler → PCA → GMM with BIC search.

        Args:
            feature_matrix: shape (n_samples, n_features)

        Returns:
            dict with fit statistics (bic_scores, optimal_n, converged, etc.)
        """
        if feature_matrix.shape[0] < 100:
            logger.warning("regime.insufficient_data", n_samples=feature_matrix.shape[0])
            return {"error": "insufficient_data"}

        # Clean input
        X = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 1: Standardize
        X_scaled = self._scaler.fit_transform(X)

        # Step 2: PCA
        X_pca = self._pca.fit_transform(X_scaled)
        n_pca = X_pca.shape[1]

        # Step 3: BIC grid search for optimal n_components
        bic_scores: dict[int, float] = {}
        for n in range(self.min_components, self.max_components + 1):
            if n > X_pca.shape[0]:
                break
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="full",
                n_init=3,
                max_iter=200,
                random_state=42,
            )
            gmm.fit(X_pca)
            bic_scores[n] = gmm.bic(X_pca)

        self._optimal_n = min(bic_scores, key=bic_scores.get)

        # Step 4: Fit final GMM with optimal n
        self._gmm = GaussianMixture(
            n_components=self._optimal_n,
            covariance_type="full",
            n_init=5,
            max_iter=300,
            random_state=42,
        )
        self._gmm.fit(X_pca)
        self._is_fitted = True

        # Generate regime labels
        self._regime_names = [f"regime_{i}" for i in range(self._optimal_n)]

        stats = {
            "optimal_n": self._optimal_n,
            "bic_scores": bic_scores,
            "converged": self._gmm.converged_,
            "n_iter": self._gmm.n_iter_,
            "pca_components": n_pca,
            "pca_explained_variance": float(np.sum(self._pca.explained_variance_ratio_)),
        }

        logger.info("regime.fitted", **stats)
        return stats

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Soft regime assignments (probability vector).

        Args:
            features: shape (n_samples, n_features) or (n_features,)

        Returns:
            shape (n_samples, n_components) probability matrix
        """
        if not self._is_fitted or self._gmm is None:
            return np.zeros(self._optimal_n)

        X = features.reshape(1, -1) if features.ndim == 1 else features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self._scaler.transform(X)
        X_pca = self._pca.transform(X_scaled)
        return self._gmm.predict_proba(X_pca)

    def predict(self, features: np.ndarray) -> int:
        """Hard regime assignment (argmax of probabilities)."""
        proba = self.predict_proba(features)
        if proba.ndim == 1:
            return int(np.argmax(proba))
        return int(np.argmax(proba[0]))

    def should_refit(self, new_data: np.ndarray) -> bool:
        """
        Check if autonomous refit is needed based on OOS log-likelihood.

        Returns True if current LL < 5th percentile of historical LL distribution.
        """
        if not self._is_fitted or self._gmm is None:
            return True

        X = np.nan_to_num(new_data, nan=0.0, posinf=0.0, neginf=0.0)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        try:
            X_scaled = self._scaler.transform(X)
            X_pca = self._pca.transform(X_scaled)
            ll = self._gmm.score(X_pca)
        except Exception:
            return True

        self._ll_history.append(ll)

        if len(self._ll_history) < 50:
            return False

        p_threshold = np.percentile(
            list(self._ll_history), self._refit_percentile
        )
        should = ll < p_threshold

        if should:
            logger.warning(
                "regime.refit_triggered",
                current_ll=round(ll, 4),
                threshold_ll=round(p_threshold, 4),
                percentile=self._refit_percentile,
            )

        return should

    def get_regime_name(self, regime_id: int) -> str:
        """Get human-readable regime name."""
        if regime_id < len(self._regime_names):
            return self._regime_names[regime_id]
        return f"regime_{regime_id}"
