"""
Data Validation and Normalization Engine for BTC Swing Trading.
Enforces data cleanliness, missing candle detection, duplicate prevention,
OHLCV consistency, and data freshness checks across 1D, 4H, and 1H timeframes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


TIMEFRAME_DELTAS = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "1w": timedelta(weeks=1),
}


@dataclass
class ValidationResult:
    """Detailed result of data quality inspection."""
    is_valid: bool
    timeframe: str
    total_candles: int
    missing_gaps: List[Tuple[str, str, int]] = field(default_factory=list)  # (start_iso, end_iso, missing_count)
    duplicate_count: int = 0
    invalid_ohlcv_count: int = 0
    is_stale: bool = False
    latest_candle_time: Optional[str] = None
    freshness_lag_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_df: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "timeframe": self.timeframe,
            "total_candles": self.total_candles,
            "missing_gaps_count": len(self.missing_gaps),
            "missing_gaps": self.missing_gaps[:10],  # cap at 10 for display
            "duplicate_count": self.duplicate_count,
            "invalid_ohlcv_count": self.invalid_ohlcv_count,
            "is_stale": self.is_stale,
            "latest_candle_time": self.latest_candle_time,
            "freshness_lag_seconds": round(self.freshness_lag_seconds, 1),
            "errors": self.errors,
            "warnings": self.warnings,
        }


class DataValidator:
    """
    Validates, normalizes, and filters market data candles before strategy calculation.
    Guarantees that decisions only use valid, strictly closed, and non-stale data.
    """

    def __init__(self, max_staleness_multiplier: float = 2.5):
        self.max_staleness_multiplier = max_staleness_multiplier

    @staticmethod
    def get_timeframe_delta(timeframe: str) -> timedelta:
        tf_lower = timeframe.lower()
        if tf_lower not in TIMEFRAME_DELTAS:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {list(TIMEFRAME_DELTAS.keys())}")
        return TIMEFRAME_DELTAS[tf_lower]

    @staticmethod
    def _normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
        """Ensures 'date' column is UTC datetime and index is sorted."""
        df_copy = df.copy()
        if 'date' not in df_copy.columns:
            if 'timestamp' in df_copy.columns:
                df_copy['date'] = df_copy['timestamp']
            elif isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy['date'] = df_copy.index
            else:
                raise ValueError("DataFrame must contain a 'date' or 'timestamp' column or DatetimeIndex")

        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            # Could be millisecond epoch or string
            if pd.api.types.is_numeric_dtype(df_copy['date']):
                # Test whether milliseconds or seconds
                sample = df_copy['date'].iloc[0] if len(df_copy) > 0 else 0
                unit = 'ms' if sample > 1e11 else 's'
                df_copy['date'] = pd.to_datetime(df_copy['date'], unit=unit, utc=True)
            else:
                df_copy['date'] = pd.to_datetime(df_copy['date'], utc=True)
        elif df_copy['date'].dt.tz is None:
            df_copy['date'] = df_copy['date'].dt.tz_localize(timezone.utc)
        else:
            df_copy['date'] = df_copy['date'].dt.tz_convert(timezone.utc)

        df_copy = df_copy.sort_values('date').reset_index(drop=True)
        return df_copy

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        timeframe: str = "1h",
        check_freshness: bool = True,
        current_time: Optional[datetime] = None,
        drop_incomplete_candle: bool = True
    ) -> ValidationResult:
        """
        Comprehensive data validation of an OHLCV dataframe.
        Checks:
        1. Required columns (date, open, high, low, close, volume)
        2. Non-empty data
        3. Valid numeric ranges and OHLC logic (high >= low, high >= open, etc.)
        4. Duplicate timestamps
        5. Missing candle gaps
        6. Incomplete candle at head/tail
        7. Freshness check against current time
        """
        errors = []
        warnings = []
        tf_delta = self.get_timeframe_delta(timeframe)

        if df is None or len(df) == 0:
            return ValidationResult(
                is_valid=False,
                timeframe=timeframe,
                total_candles=0,
                errors=["DataFrame is empty or None"]
            )

        # 1. Column verification
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            return ValidationResult(
                is_valid=False,
                timeframe=timeframe,
                total_candles=len(df),
                errors=[f"Missing required OHLCV columns: {missing_cols}"]
            )

        try:
            norm_df = self._normalize_date_column(df)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                timeframe=timeframe,
                total_candles=len(df),
                errors=[f"Failed to normalize date column: {str(e)}"]
            )

        # 2. Check and remove duplicate timestamps
        duplicate_mask = norm_df.duplicated(subset=['date'], keep='last')
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count > 0:
            warnings.append(f"Found and deduplicated {duplicate_count} duplicate timestamp candles")
            norm_df = norm_df[~duplicate_mask].reset_index(drop=True)

        # 3. Check for invalid OHLCV logic
        # High must be >= Low, High >= Open, High >= Close
        # Low must be <= Open, Low <= Close
        # Prices must be strictly positive
        # Volume must be >= 0
        invalid_mask = (
            (norm_df['high'] < norm_df['low']) |
            (norm_df['high'] < norm_df['open']) |
            (norm_df['high'] < norm_df['close']) |
            (norm_df['low'] > norm_df['open']) |
            (norm_df['low'] > norm_df['close']) |
            (norm_df['open'] <= 0) |
            (norm_df['high'] <= 0) |
            (norm_df['low'] <= 0) |
            (norm_df['close'] <= 0) |
            (norm_df['volume'] < 0) |
            norm_df[['open', 'high', 'low', 'close', 'volume']].isna().any(axis=1) |
            np.isinf(norm_df[['open', 'high', 'low', 'close', 'volume']]).any(axis=1)
        )
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            errors.append(f"Found {invalid_count} candles with invalid/corrupt OHLCV values (dropped)")
            norm_df = norm_df[~invalid_mask].reset_index(drop=True)

        if len(norm_df) == 0:
            return ValidationResult(
                is_valid=False,
                timeframe=timeframe,
                total_candles=0,
                duplicate_count=duplicate_count,
                invalid_ohlcv_count=invalid_count,
                errors=errors + ["All candles were invalid after filtering"]
            )

        # 4. Check for incomplete ongoing candle
        now_utc = current_time or datetime.now(timezone.utc)
        if drop_incomplete_candle and len(norm_df) > 0:
            latest_candle_time = norm_df['date'].iloc[-1].to_pydatetime()
            # If the current candle close time is in the future relative to now,
            # it is incomplete and should not be used for closed-candle decisions
            candle_close_time = latest_candle_time + tf_delta
            if candle_close_time > now_utc:
                warnings.append(f"Dropped incomplete live candle starting at {latest_candle_time.isoformat()}")
                norm_df = norm_df.iloc[:-1].reset_index(drop=True)

        if len(norm_df) == 0:
            return ValidationResult(
                is_valid=False,
                timeframe=timeframe,
                total_candles=0,
                errors=["No closed candles remaining after dropping incomplete candle"]
            )

        # 5. Missing candle / Gap detection
        missing_gaps = []
        if len(norm_df) > 1:
            diffs = norm_df['date'].diff().iloc[1:]
            expected_seconds = tf_delta.total_seconds()
            
            # Gaps occur when diff is greater than 1.5x expected step
            gap_indices = np.where(diffs.dt.total_seconds() > expected_seconds * 1.5)[0]
            for idx in gap_indices:
                # idx in diffs corresponds to row idx + 1
                row_idx = idx + 1
                gap_start = norm_df['date'].iloc[row_idx - 1].to_pydatetime()
                gap_end = norm_df['date'].iloc[row_idx].to_pydatetime()
                gap_seconds = (gap_end - gap_start).total_seconds()
                missing_count = int(round(gap_seconds / expected_seconds)) - 1
                missing_gaps.append((
                    gap_start.isoformat(),
                    gap_end.isoformat(),
                    missing_count
                ))
            if missing_gaps:
                warnings.append(f"Detected {len(missing_gaps)} time gap(s) with missing candles")

        # 6. Freshness check
        latest_candle = norm_df['date'].iloc[-1].to_pydatetime()
        latest_close = latest_candle + tf_delta
        freshness_lag = (now_utc - latest_close).total_seconds()
        
        # Stale if lag exceeds max_staleness_multiplier * timeframe duration
        max_allowed_lag = tf_delta.total_seconds() * self.max_staleness_multiplier
        is_stale = freshness_lag > max_allowed_lag if check_freshness else False
        if is_stale and check_freshness:
            warnings.append(
                f"Data is STALE: latest closed candle ({latest_candle.isoformat()}) "
                f"lag is {freshness_lag:.1f}s (max allowed {max_allowed_lag:.1f}s)"
            )

        is_valid = len(errors) == 0 and len(norm_df) > 0

        return ValidationResult(
            is_valid=is_valid,
            timeframe=timeframe,
            total_candles=len(norm_df),
            missing_gaps=missing_gaps,
            duplicate_count=duplicate_count,
            invalid_ohlcv_count=invalid_count,
            is_stale=is_stale,
            latest_candle_time=latest_candle.isoformat() if len(norm_df) > 0 else None,
            freshness_lag_seconds=max(0.0, freshness_lag),
            errors=errors,
            warnings=warnings,
            cleaned_df=norm_df
        )

    def clean_and_normalize(
        self,
        df: pd.DataFrame,
        timeframe: str = "1h",
        drop_incomplete: bool = True
    ) -> pd.DataFrame:
        """Convenience method returning sanitized, validated dataframe directly."""
        result = self.validate_dataframe(
            df,
            timeframe=timeframe,
            check_freshness=False,
            drop_incomplete_candle=drop_incomplete
        )
        if not result.is_valid or result.cleaned_df is None:
            raise ValueError(f"DataFrame validation failed: {result.errors}")
        return result.cleaned_df

    def validate_multi_timeframe(
        self,
        dfs: Dict[str, pd.DataFrame],
        required_timeframes: Optional[List[str]] = None,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Validates multiple timeframes (e.g. 1d, 4h, 1h) together.
        Ensures all required timeframes are present, valid, and synchronized.
        """
        required = required_timeframes or ["1d", "4h", "1h"]
        results = {}
        all_valid = True
        overall_errors = []
        overall_warnings = []

        now_utc = current_time or datetime.now(timezone.utc)

        for tf in required:
            if tf not in dfs or dfs[tf] is None:
                all_valid = False
                overall_errors.append(f"Missing required timeframe: '{tf}'")
                continue
            
            res = self.validate_dataframe(
                dfs[tf],
                timeframe=tf,
                check_freshness=True,
                current_time=now_utc,
                drop_incomplete_candle=True
            )
            results[tf] = res
            if not res.is_valid:
                all_valid = False
                overall_errors.extend([f"[{tf}] {err}" for err in res.errors])
            if res.is_stale:
                overall_warnings.append(f"[{tf}] Data is flagged as stale")
            if res.warnings:
                overall_warnings.extend([f"[{tf}] {warn}" for warn in res.warnings])

        # Cross-timeframe timestamp overlap check
        if all_valid and len(results) >= 2:
            latest_times = {
                tf: datetime.fromisoformat(res.latest_candle_time)
                for tf, res in results.items() if res.latest_candle_time
            }
            # All timeframes should have recent coverage ending close to now
            max_end = max(latest_times.values())
            min_end = min(latest_times.values())
            # For 1d vs 1h, difference between latest closed candle could be up to ~24h
            if (max_end - min_end).total_seconds() > 86400 * 2:
                overall_warnings.append(
                    f"Timeframe end dates desynchronized by more than 48h: {latest_times}"
                )

        return {
            "all_valid": all_valid,
            "overall_errors": overall_errors,
            "overall_warnings": overall_warnings,
            "timeframes": {tf: res.to_dict() for tf, res in results.items()}
        }
