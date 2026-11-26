"""
Machine Learning Signal Quality Filter (Section 14).
Uses chronological XGBoost classification to estimate the probability that a
candidate setup reaches its target profit before its stop-loss (net of costs).
Acts strictly as an optional secondary filter; cannot override risk limits.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from scripts.engine.regime_detector import MarketRegime
from scripts.engine.storage import EngineStorage


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "user_data" / "models"

REGIME_ENCODING = {
    MarketRegime.BULLISH_TREND.value: 1.0,
    MarketRegime.BEARISH_TREND.value: -1.0,
    MarketRegime.SIDEWAYS.value: 0.0,
    MarketRegime.HIGH_VOLATILITY.value: 2.0,
    MarketRegime.LOW_VOLATILITY.value: -2.0,
    MarketRegime.UNCERTAIN.value: 0.0,
}

FEATURE_COLUMNS = [
    "dist_ema20_pct",
    "dist_ema50_pct",
    "ema_20_50_spread_pct",
    "ema_20_slope",
    "ema_50_slope",
    "rsi_14",
    "macd_hist_pct",
    "adx_14",
    "atr_pct",
    "bb_width",
    "relative_volume",
    "ret_6h_pct",
    "ret_24h_pct",
    "hist_volatility_20",
    "regime_code",
    "supertrend_direction",
]


@dataclass
class MLConfig:
    """Configuration for ML labeling, training, and filtering."""
    enabled: bool = False                     # Optional secondary filter
    min_probability: float = 0.50             # Minimum predicted probability to pass filter
    horizon_bars: int = 48                    # 48h forward labeling window
    tp_atr_mult: float = 3.0                  # Target profit in ATR multiples
    sl_atr_mult: float = 2.0                  # Stop-loss in ATR multiples
    roundtrip_cost_pct: float = 0.003         # 0.3% fees + slippage buffer
    train_ratio: float = 0.75                 # Chronological 75% train / 25% test


@dataclass
class MLValidationMetrics:
    version: str
    trained_at_utc: str
    model_type: str
    model_path: str
    train_samples: int
    test_samples: int
    precision_score: float
    recall_score: float
    brier_score: float
    roc_auc: float
    baseline_accuracy: float
    feature_names: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "trained_at_utc": self.trained_at_utc,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "precision_score": round(self.precision_score, 4),
            "recall_score": round(self.recall_score, 4),
            "brier_score": round(self.brier_score, 4),
            "roc_auc": round(self.roc_auc, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "feature_names": self.feature_names,
        }


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the standardized ML feature matrix from an indicator-enriched dataframe."""
    feat = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    ema20 = df["ema_20"].astype(float)
    ema50 = df["ema_50"].astype(float)

    feat["dist_ema20_pct"] = np.where(ema20 > 0, (close - ema20) / ema20 * 100.0, 0.0)
    feat["dist_ema50_pct"] = np.where(ema50 > 0, (close - ema50) / ema50 * 100.0, 0.0)
    feat["ema_20_50_spread_pct"] = np.where(ema50 > 0, (ema20 - ema50) / ema50 * 100.0, 0.0)
    feat["ema_20_slope"] = df.get("ema_20_slope", 0.0)
    feat["ema_50_slope"] = df.get("ema_50_slope", 0.0)
    feat["rsi_14"] = df.get("rsi_14", 50.0)
    feat["macd_hist_pct"] = np.where(close > 0, df.get("macd_hist", 0.0) / close * 100.0, 0.0)
    feat["adx_14"] = df.get("adx_14", 20.0)
    feat["atr_pct"] = df.get("atr_pct", 1.0)
    feat["bb_width"] = df.get("bb_width", 0.05)
    feat["relative_volume"] = df.get("relative_volume", 1.0)
    feat["ret_6h_pct"] = close.pct_change(6) * 100.0
    feat["ret_24h_pct"] = close.pct_change(24) * 100.0
    feat["hist_volatility_20"] = df.get("hist_volatility_20", 0.0)

    if "market_regime" in df.columns:
        feat["regime_code"] = df["market_regime"].map(REGIME_ENCODING).fillna(0.0)
    else:
        feat["regime_code"] = 0.0

    feat["supertrend_direction"] = df.get("supertrend_direction", 0)
    return feat[FEATURE_COLUMNS]


def generate_triple_barrier_labels(df: pd.DataFrame, cfg: MLConfig) -> pd.Series:
    """
    Labels each bar i as 1 if future price reaches TP (net of costs) strictly before SL
    within cfg.horizon_bars, else 0. Last horizon_bars rows are NaN (cannot be labeled).
    """
    n = len(df)
    labels = np.full(n, np.nan)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr_14"].values if "atr_14" in df.columns else closes * 0.015

    for i in range(n - cfg.horizon_bars):
        entry = closes[i] * (1.0 + cfg.roundtrip_cost_pct / 2.0)
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0 or entry <= 0:
            continue
        tp_price = entry * (1.0 + cfg.roundtrip_cost_pct / 2.0) + (atr * cfg.tp_atr_mult)
        sl_price = entry - (atr * cfg.sl_atr_mult)

        outcome = 0.0
        for j in range(i + 1, i + 1 + cfg.horizon_bars):
            # Conservative: check SL first
            if lows[j] <= sl_price:
                outcome = 0.0
                break
            if highs[j] >= tp_price:
                outcome = 1.0
                break
        labels[i] = outcome

    return pd.Series(labels, index=df.index, name="target")


class MLSignalFilter:
    """
    Optional XGBoost-based probability filter for candidate swing signals.
    """

    def __init__(self, config: Optional[MLConfig] = None, storage: Optional[EngineStorage] = None):
        self.config = config or MLConfig()
        self.storage = storage
        self.model = None
        self.active_version: Optional[str] = None

    def train_chronological(
        self,
        df_enriched: pd.DataFrame,
        version: Optional[str] = None,
        save_model: bool = True
    ) -> MLValidationMetrics:
        """
        Trains an XGBoost classifier using a strict chronological split with a purge gap
        to prevent label-overlap leakage.
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.metrics import precision_score, recall_score, brier_score_loss, roc_auc_score
        except ImportError as exc:
            raise RuntimeError("xgboost and scikit-learn are required to train MLSignalFilter") from exc

        X = extract_features(df_enriched)
        y = generate_triple_barrier_labels(df_enriched, self.config)

        dataset = X.copy()
        dataset["target"] = y
        dataset = dataset.dropna().reset_index(drop=True)

        if len(dataset) < 100:
            raise ValueError(f"Insufficient labeled samples for training ({len(dataset)} < 100)")

        split_idx = int(len(dataset) * self.config.train_ratio)
        # Purge gap equal to horizon_bars to prevent overlapping label leakage across split
        purge_end = min(len(dataset) - 20, split_idx + self.config.horizon_bars)
        train_df = dataset.iloc[:split_idx]
        test_df = dataset.iloc[purge_end:]

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target"].astype(int)
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target"].astype(int)

        model = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= self.config.min_probability).astype(int)

        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        brier = float(brier_score_loss(y_test, probs))
        auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.5
        baseline_acc = float(max(y_test.mean(), 1.0 - y_test.mean()))

        ver = version or f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = str(MODEL_DIR / f"xgb_swing_{ver}.joblib")
        if save_model:
            joblib.dump(model, model_path)

        self.model = model
        self.active_version = ver

        metrics = MLValidationMetrics(
            version=ver,
            trained_at_utc=datetime.now(timezone.utc).isoformat(),
            model_type="XGBClassifier",
            model_path=model_path,
            train_samples=len(train_df),
            test_samples=len(test_df),
            precision_score=prec,
            recall_score=rec,
            brier_score=brier,
            roc_auc=auc,
            baseline_accuracy=baseline_acc,
        )
        if self.storage is not None and save_model:
            self.storage.save_ml_model_metadata(metrics.to_dict())
        return metrics

    def load_latest(self) -> bool:
        """Loads the active versioned model from EngineStorage if available."""
        if self.storage is None:
            return False
        meta = self.storage.get_active_ml_model()
        if not meta:
            return False
        path = Path(meta["model_path"])
        if not path.exists():
            return False
        self.model = joblib.load(path)
        self.active_version = meta["version"]
        return True

    def predict_success_probability(self, df_enriched: pd.DataFrame, candle_idx: int = -1) -> Optional[float]:
        """Returns probability [0.0, 1.0] for the given candle, or None if model unavailable."""
        if self.model is None or df_enriched is None or len(df_enriched) == 0:
            return None
        # Compute features with sufficient lookback for pct_change before selecting candle_idx
        feats_all = extract_features(df_enriched)
        feats = feats_all.iloc[[candle_idx]]
        if feats.isna().any(axis=None):
            return None
        prob = float(self.model.predict_proba(feats)[:, 1][0])
        return prob

    def passes_filter(self, df_enriched: pd.DataFrame, candle_idx: int = -1) -> Tuple[bool, Optional[float], str]:
        """
        Evaluates whether a candidate signal passes the ML quality filter.
        Always returns True if ML filtering is disabled or no model is loaded.
        """
        if not self.config.enabled:
            return True, None, "ML filter disabled"
        prob = self.predict_success_probability(df_enriched, candle_idx=candle_idx)
        if prob is None:
            return True, None, "ML model unavailable (bypassed)"
        if prob >= self.config.min_probability:
            return True, prob, f"ML approved (p={prob:.3f} >= {self.config.min_probability:.2f})"
        return False, prob, f"ML rejected (p={prob:.3f} < {self.config.min_probability:.2f})"
