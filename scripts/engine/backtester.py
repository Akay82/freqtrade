"""
Historical Backtesting Engine for BTC Swing Trading (Section 12).
Features:
- Multi-timeframe zero-lookahead alignment (uses strictly closed 1D, 4H, 1H candles)
- Realistic execution simulation (next-bar open + slippage, fees, conservative SL-first intrabar check)
- Full 12-metric quantitative evaluation (Sharpe, Sortino, Profit Factor, Expectancy, etc.)
- Benchmark comparisons (BTC Buy-and-Hold & 50/200 EMA Crossover baseline)
- Regime-segmented performance breakdown
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import pandas as pd

from scripts.engine.data_validator import DataValidator
from scripts.engine.indicators import IndicatorEngine
from scripts.engine.price_action import PriceActionDetector
from scripts.engine.regime_detector import RegimeDetector, MarketRegime
from scripts.engine.signal_scorer import SignalScorer, ScoringConfig
from scripts.engine.risk_manager import RiskManager, RiskConfig
from scripts.engine.strategies import (
    BaseSwingStrategy,
    StrategyA_TrendFollowing,
    StrategyB_Breakout,
    StrategyC_Pullback,
    StrategyD_RangeBound,
)


@dataclass
class BacktestConfig:
    """Configuration for a historical backtest run."""
    initial_capital: float = 1000.0
    fee_rate: float = 0.001
    slippage_bps: float = 5.0
    risk_per_trade_pct: float = 0.01
    min_signal_score: float = 65.0
    warmup_bars: int = 200
    strategy_filter: Optional[str] = None     # None = all eligible strategies, or specific strategy name
    enable_trailing_stop: bool = True
    trailing_activation_rr: float = 1.5
    trailing_atr_mult: float = 1.5


@dataclass
class BacktestTrade:
    """Record of a single simulated backtest trade."""
    trade_id: int
    strategy: str
    regime: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    signal_score: float
    fees: float
    slippage_cost: float
    net_pnl: float
    return_pct: float
    r_multiple: float
    hold_hours: float
    exit_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "strategy": self.strategy,
            "regime": self.regime,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "quantity": round(self.quantity, 6),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit": round(self.take_profit, 2),
            "signal_score": round(self.signal_score, 1),
            "fees": round(self.fees, 4),
            "slippage_cost": round(self.slippage_cost, 4),
            "net_pnl": round(self.net_pnl, 2),
            "return_pct": round(self.return_pct, 2),
            "r_multiple": round(self.r_multiple, 2),
            "hold_hours": round(self.hold_hours, 1),
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestMetrics:
    """All 12 required backtesting performance metrics (Section 12)."""
    total_return_pct: float
    net_profit: float
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    avg_reward_risk: float
    sharpe_ratio: float
    sortino_ratio: float
    avg_trade_return_pct: float
    expectancy: float
    total_trades: int
    max_losing_streak: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return_pct": round(self.total_return_pct, 2),
            "net_profit": round(self.net_profit, 2),
            "win_rate_pct": round(self.win_rate_pct, 1),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "avg_reward_risk": round(self.avg_reward_risk, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "avg_trade_return_pct": round(self.avg_trade_return_pct, 2),
            "expectancy": round(self.expectancy, 2),
            "total_trades": self.total_trades,
            "max_losing_streak": self.max_losing_streak,
        }


@dataclass
class BacktestResult:
    """Complete output of a backtest run including metrics, equity curve, and benchmarks."""
    run_id: str
    strategy_name: str
    start_date: str
    end_date: str
    config: BacktestConfig
    metrics: BacktestMetrics
    benchmarks: Dict[str, float]
    regime_breakdown: Dict[str, Dict[str, Any]]
    trades: List[BacktestTrade]
    equity_curve: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "metrics": self.metrics.to_dict(),
            "benchmarks": {k: round(v, 2) for k, v in self.benchmarks.items()},
            "regime_breakdown": self.regime_breakdown,
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": self.equity_curve,
        }


def compute_metrics(
    trades: List[BacktestTrade],
    equity_series: List[float],
    initial_capital: float
) -> BacktestMetrics:
    """Computes all 12 quantitative metrics deterministically from trade list & equity curve."""
    total_trades = len(trades)
    final_equity = equity_series[-1] if equity_series else initial_capital
    net_profit = final_equity - initial_capital
    total_return_pct = (net_profit / initial_capital * 100.0) if initial_capital > 0 else 0.0

    # Max Drawdown from equity curve
    peak = initial_capital
    max_dd = 0.0
    for eq in equity_series:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

    if total_trades == 0:
        return BacktestMetrics(
            total_return_pct=total_return_pct,
            net_profit=net_profit,
            win_rate_pct=0.0,
            profit_factor=0.0,
            max_drawdown_pct=max_dd,
            avg_reward_risk=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            avg_trade_return_pct=0.0,
            expectancy=0.0,
            total_trades=0,
            max_losing_streak=0,
        )

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]
    win_rate = (len(wins) / total_trades) * 100.0

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    avg_rr = sum(t.r_multiple for t in trades) / total_trades
    avg_trade_ret = sum(t.return_pct for t in trades) / total_trades
    expectancy = net_profit / total_trades

    # Max losing streak
    max_streak = 0
    cur_streak = 0
    for t in trades:
        if t.net_pnl <= 0:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    # Sharpe & Sortino on hourly equity returns (annualized by sqrt(8760))
    sharpe = 0.0
    sortino = 0.0
    if len(equity_series) > 2:
        eq_arr = np.array(equity_series, dtype=float)
        rets = np.diff(eq_arr) / np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1.0)
        mean_ret = float(np.mean(rets))
        std_ret = float(np.std(rets))
        if std_ret > 1e-12:
            sharpe = (mean_ret / std_ret) * math.sqrt(8760.0)
        downside = rets[rets < 0]
        downside_std = float(np.std(downside)) if len(downside) > 1 else 0.0
        if downside_std > 1e-12:
            sortino = (mean_ret / downside_std) * math.sqrt(8760.0)

    return BacktestMetrics(
        total_return_pct=total_return_pct,
        net_profit=net_profit,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd,
        avg_reward_risk=avg_rr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        avg_trade_return_pct=avg_trade_ret,
        expectancy=expectancy,
        total_trades=total_trades,
        max_losing_streak=max_streak,
    )


class SwingBacktester:
    """
    Zero-lookahead multi-timeframe backtester for the BTC swing trading engine.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.validator = DataValidator()
        self.ie = IndicatorEngine()
        self.pa = PriceActionDetector()
        self.regime_detector = RegimeDetector()

    def _prepare_aligned_data(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Computes indicators and aligns 1D/4H closed candles to 1H bar indices
        using close_time = open_time + timeframe_delta to guarantee zero lookahead.
        """
        c_1d = self.pa.compute_all_price_action(
            self.ie.compute_all_indicators(self.validator.clean_and_normalize(df_1d, "1d", drop_incomplete=False))
        )
        c_4h = self.pa.compute_all_price_action(
            self.ie.compute_all_indicators(self.validator.clean_and_normalize(df_4h, "4h", drop_incomplete=False))
        )
        c_1h = self.pa.compute_all_price_action(
            self.ie.compute_all_indicators(self.validator.clean_and_normalize(df_1h, "1h", drop_incomplete=False))
        )

        c_4h = self.regime_detector.classify_series(c_4h)

        # Compute candle close times
        c_1d['close_time'] = c_1d['date'] + timedelta(days=1)
        c_4h['close_time'] = c_4h['date'] + timedelta(hours=4)
        c_1h['close_time'] = c_1h['date'] + timedelta(hours=1)

        c_1d['idx_1d'] = np.arange(len(c_1d))
        c_4h['idx_4h'] = np.arange(len(c_4h))

        # Map each 1h candle's close_time to the latest CLOSED 4h and 1d candle index
        map_4h = pd.merge_asof(
            c_1h[['close_time']].sort_values('close_time'),
            c_4h[['close_time', 'idx_4h']].sort_values('close_time'),
            on='close_time',
            direction='backward'
        )
        map_1d = pd.merge_asof(
            c_1h[['close_time']].sort_values('close_time'),
            c_1d[['close_time', 'idx_1d']].sort_values('close_time'),
            on='close_time',
            direction='backward'
        )

        c_1h['mapped_4h_idx'] = map_4h['idx_4h'].values
        c_1h['mapped_1d_idx'] = map_1d['idx_1d'].values
        return c_1d, c_4h, c_1h

    def _compute_benchmarks(self, c_1h: pd.DataFrame, start_idx: int) -> Dict[str, float]:
        """Computes BTC Buy-and-Hold and simple 50/200 EMA crossover baseline returns."""
        if len(c_1h) <= start_idx + 1:
            return {"btc_buy_hold_pct": 0.0, "ema_crossover_pct": 0.0}

        sub = c_1h.iloc[start_idx:].reset_index(drop=True)
        entry_p = float(sub['close'].iloc[0])
        exit_p = float(sub['close'].iloc[-1])
        fee = self.config.fee_rate
        buy_hold = (((exit_p * (1.0 - fee)) / (entry_p * (1.0 + fee))) - 1.0) * 100.0

        # Simple 50/200 EMA crossover baseline
        cash = 1000.0
        btc_qty = 0.0
        for i in range(1, len(sub)):
            ema50 = float(sub['ema_50'].iloc[i])
            ema200 = float(sub['ema_200'].iloc[i])
            price = float(sub['close'].iloc[i])
            if np.isnan(ema50) or np.isnan(ema200):
                continue
            if btc_qty == 0.0 and ema50 > ema200:
                btc_qty = (cash * (1.0 - fee)) / price
                cash = 0.0
            elif btc_qty > 0.0 and ema50 < ema200:
                cash = btc_qty * price * (1.0 - fee)
                btc_qty = 0.0
        final_ema = cash + btc_qty * exit_p
        ema_ret = (final_ema - 1000.0) / 1000.0 * 100.0

        return {
            "btc_buy_hold_pct": buy_hold,
            "ema_crossover_pct": ema_ret,
        }

    def run(
        self,
        df_1d: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        run_id: Optional[str] = None,
        strategies: Optional[List[BaseSwingStrategy]] = None
    ) -> BacktestResult:
        """Executes bar-by-bar historical backtest with strict zero-lookahead."""
        c_1d, c_4h, c_1h = self._prepare_aligned_data(df_1d, df_4h, df_1h)

        strat_list = strategies or [
            StrategyA_TrendFollowing(),
            StrategyB_Breakout(),
            StrategyC_Pullback(),
            StrategyD_RangeBound(is_enabled=False),
        ]
        if self.config.strategy_filter:
            strat_list = [s for s in strat_list if s.name == self.config.strategy_filter]

        scorer = SignalScorer(ScoringConfig(moderate_threshold=self.config.min_signal_score))
        risk_mgr = RiskManager(
            initial_equity=self.config.initial_capital,
            config=RiskConfig(
                risk_per_trade_pct=self.config.risk_per_trade_pct,
                max_open_positions=1,
                fee_rate=self.config.fee_rate,
                slippage_bps=self.config.slippage_bps,
            )
        )

        cash = float(self.config.initial_capital)
        slip_rate = self.config.slippage_bps / 10000.0
        fee_rate = self.config.fee_rate

        open_pos: Optional[Dict[str, Any]] = None
        trades: List[BacktestTrade] = []
        equity_series: List[float] = []
        equity_curve: List[Dict[str, Any]] = []

        start_idx = min(self.config.warmup_bars, max(30, len(c_1h) // 4))
        trade_counter = 0

        for i in range(start_idx, len(c_1h) - 1):
            bar = c_1h.iloc[i]
            next_bar = c_1h.iloc[i + 1]
            bar_time = bar['date'].to_pydatetime()
            risk_mgr.check_day_rollover(bar_time)

            # 1. Manage Open Position on current bar
            if open_pos is not None:
                low_p = float(bar['low'])
                high_p = float(bar['high'])
                close_p = float(bar['close'])
                atr_val = float(bar.get('atr_14', close_p * 0.015))

                exit_price = None
                exit_reason = None

                # Conservative: check stop-loss first
                if low_p <= open_pos['stop_loss']:
                    exit_price = open_pos['stop_loss'] * (1.0 - slip_rate)
                    exit_reason = "STOP_LOSS"
                elif high_p >= open_pos['take_profit']:
                    exit_price = open_pos['take_profit'] * (1.0 - slip_rate)
                    exit_reason = "TAKE_PROFIT"
                elif self.config.enable_trailing_stop and not np.isnan(atr_val):
                    risk_dist = open_pos['entry_price'] - open_pos['initial_sl']
                    if risk_dist > 0 and (high_p - open_pos['entry_price']) >= risk_dist * self.config.trailing_activation_rr:
                        new_sl = high_p - (atr_val * self.config.trailing_atr_mult)
                        if new_sl > open_pos['stop_loss']:
                            open_pos['stop_loss'] = new_sl

                if exit_price is not None:
                    qty = open_pos['quantity']
                    gross_exit = qty * exit_price
                    exit_fee = gross_exit * fee_rate
                    net_proceeds = gross_exit - exit_fee
                    cash += net_proceeds

                    total_fees = open_pos['entry_fee'] + exit_fee
                    slip_cost = (open_pos['entry_price'] * slip_rate * qty) + (open_pos['stop_loss'] * slip_rate * qty)
                    net_pnl = net_proceeds - open_pos['entry_cost']
                    ret_pct = (net_pnl / open_pos['entry_cost'] * 100.0) if open_pos['entry_cost'] > 0 else 0.0
                    init_risk = (open_pos['entry_price'] - open_pos['initial_sl']) * qty
                    r_mult = (net_pnl / init_risk) if init_risk > 0 else 0.0
                    hold_hrs = (bar_time - open_pos['entry_dt']).total_seconds() / 3600.0

                    trade_counter += 1
                    trades.append(BacktestTrade(
                        trade_id=trade_counter,
                        strategy=open_pos['strategy'],
                        regime=open_pos['regime'],
                        entry_time=open_pos['entry_dt'].isoformat(),
                        exit_time=bar_time.isoformat(),
                        entry_price=open_pos['entry_price'],
                        exit_price=exit_price,
                        quantity=qty,
                        stop_loss=open_pos['initial_sl'],
                        take_profit=open_pos['take_profit'],
                        signal_score=open_pos['signal_score'],
                        fees=total_fees,
                        slippage_cost=slip_cost,
                        net_pnl=net_pnl,
                        return_pct=ret_pct,
                        r_multiple=r_mult,
                        hold_hours=hold_hrs,
                        exit_reason=exit_reason,
                    ))
                    risk_mgr.update_equity(cash, realized_pnl=net_pnl, now_utc=bar_time)
                    open_pos = None

            # 2. Evaluate new BUY entry on closed bar i (filled at next_bar open)
            if open_pos is None:
                m4 = bar['mapped_4h_idx']
                m1d = bar['mapped_1d_idx']
                if not np.isnan(m4) and not np.isnan(m1d):
                    idx_4h = int(m4)
                    idx_1d = int(m1d)
                    if idx_4h >= 30 and idx_1d >= 30:
                        regime_val = str(c_4h['market_regime'].iloc[idx_4h])
                        try:
                            active_regime = MarketRegime(regime_val)
                        except ValueError:
                            active_regime = MarketRegime.UNCERTAIN

                        # Only evaluate strategies compatible with active regime
                        best_sig = None
                        best_score = None
                        for strat in strat_list:
                            if not strat.is_eligible(active_regime):
                                continue
                            # Slice views up to idx to prevent any accidental future access
                            sig = strat.evaluate_setup(
                                c_1d.iloc[:idx_1d + 1],
                                c_4h.iloc[:idx_4h + 1],
                                c_1h.iloc[:i + 1],
                                candle_idx=-1
                            )
                            if sig is None or sig.action != "BUY":
                                continue
                            sc = scorer.score_signal(
                                c_1d.iloc[:idx_1d + 1],
                                c_4h.iloc[:idx_4h + 1],
                                c_1h.iloc[:i + 1],
                                strategy_name=strat.name,
                                candle_idx=-1
                            )
                            if sc.is_approved:
                                if best_score is None or sc.total_score > best_score.total_score:
                                    best_sig = sig
                                    best_score = sc

                        if best_sig is not None and best_score is not None:
                            # Simulate entry at next bar open + slippage
                            fill_price = float(next_bar['open']) * (1.0 + slip_rate)
                            atr_val = float(bar.get('atr_14', fill_price * 0.015))
                            sl, tp, _ = risk_mgr.calculate_atr_stops(
                                entry_price=fill_price,
                                atr=atr_val,
                                swing_low=float(bar.get('last_swing_low', np.nan)),
                                resistance=float(bar.get('resistance_level', np.nan)),
                            )
                            sizing = risk_mgr.evaluate_trade_risk(
                                entry_price=fill_price,
                                stop_loss_price=sl,
                                take_profit_price=tp,
                                available_balance=cash,
                                open_positions_count=0,
                                signal_key=f"bt-{i}-{best_sig.strategy_name}",
                            )
                            if sizing.approved and sizing.quantity > 0:
                                entry_fee = sizing.notional_value * fee_rate
                                total_cost = sizing.notional_value + entry_fee
                                if total_cost <= cash:
                                    cash -= total_cost
                                    open_pos = {
                                        "strategy": best_sig.strategy_name,
                                        "regime": active_regime.value,
                                        "entry_dt": next_bar['date'].to_pydatetime(),
                                        "entry_price": fill_price,
                                        "quantity": sizing.quantity,
                                        "entry_fee": entry_fee,
                                        "entry_cost": total_cost,
                                        "initial_sl": sl,
                                        "stop_loss": sl,
                                        "take_profit": tp,
                                        "signal_score": best_score.total_score,
                                    }

            # Mark-to-market equity
            mtm_eq = cash + (open_pos['quantity'] * float(bar['close']) if open_pos else 0.0)
            equity_series.append(mtm_eq)
            # Sample equity curve every 24h (daily) for clean dashboard plotting
            if (i - start_idx) % 24 == 0 or i == len(c_1h) - 2:
                equity_curve.append({
                    "timestamp": bar_time.isoformat(),
                    "equity": round(mtm_eq, 2),
                })

        # Close any open position at the final bar
        if open_pos is not None:
            final_bar = c_1h.iloc[-1]
            final_time = final_bar['date'].to_pydatetime()
            exit_price = float(final_bar['close']) * (1.0 - slip_rate)
            qty = open_pos['quantity']
            gross_exit = qty * exit_price
            exit_fee = gross_exit * fee_rate
            net_proceeds = gross_exit - exit_fee
            cash += net_proceeds
            net_pnl = net_proceeds - open_pos['entry_cost']
            init_risk = (open_pos['entry_price'] - open_pos['initial_sl']) * qty
            trade_counter += 1
            trades.append(BacktestTrade(
                trade_id=trade_counter,
                strategy=open_pos['strategy'],
                regime=open_pos['regime'],
                entry_time=open_pos['entry_dt'].isoformat(),
                exit_time=final_time.isoformat(),
                entry_price=open_pos['entry_price'],
                exit_price=exit_price,
                quantity=qty,
                stop_loss=open_pos['initial_sl'],
                take_profit=open_pos['take_profit'],
                signal_score=open_pos['signal_score'],
                fees=open_pos['entry_fee'] + exit_fee,
                slippage_cost=(open_pos['entry_price'] + exit_price) * slip_rate * qty,
                net_pnl=net_pnl,
                return_pct=(net_pnl / open_pos['entry_cost'] * 100.0) if open_pos['entry_cost'] > 0 else 0.0,
                r_multiple=(net_pnl / init_risk) if init_risk > 0 else 0.0,
                hold_hours=(final_time - open_pos['entry_dt']).total_seconds() / 3600.0,
                exit_reason="END_OF_BACKTEST",
            ))
            equity_series[-1] = cash

        metrics = compute_metrics(trades, equity_series, self.config.initial_capital)
        benchmarks = self._compute_benchmarks(c_1h, start_idx)

        # Regime breakdown
        regime_breakdown: Dict[str, Dict[str, Any]] = {}
        for t in trades:
            rb = regime_breakdown.setdefault(t.regime, {"trades": 0, "wins": 0, "net_pnl": 0.0})
            rb["trades"] += 1
            if t.net_pnl > 0:
                rb["wins"] += 1
            rb["net_pnl"] += t.net_pnl
        for r_key, rb in regime_breakdown.items():
            rb["win_rate_pct"] = round(rb["wins"] / rb["trades"] * 100.0, 1) if rb["trades"] > 0 else 0.0
            rb["net_pnl"] = round(rb["net_pnl"], 2)

        return BacktestResult(
            run_id=run_id or f"bt-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            strategy_name=self.config.strategy_filter or "ALL_REGIME_SELECTED",
            start_date=c_1h['date'].iloc[start_idx].isoformat() if len(c_1h) > start_idx else "",
            end_date=c_1h['date'].iloc[-1].isoformat() if len(c_1h) > 0 else "",
            config=self.config,
            metrics=metrics,
            benchmarks=benchmarks,
            regime_breakdown=regime_breakdown,
            trades=trades,
            equity_curve=equity_curve,
        )
