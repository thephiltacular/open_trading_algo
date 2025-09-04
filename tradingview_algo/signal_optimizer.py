"""
SignalOptimizer: Compute indicators, generate/backtest signals, and optimize signal combinations for all trade types.

Features:
- Compute indicators for all tickers and timeframes
- Generate trading signals
- Backtest signals on historical data
- Find best signal combinations for long, short, swing, etc.
- Extensible for new indicators/strategies
"""
import itertools
from typing import Any, Callable, Dict, List

import pandas as pd


class SignalOptimizer:
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        indicators: Dict[str, Callable],
        signal_generators: Dict[str, Callable],
    ):
        """
        data: Dict[ticker, DataFrame] - historical price data for each ticker
        indicators: Dict[name, func] - indicator functions (func(df) -> pd.Series)
        signal_generators: Dict[name, func] - signal functions (func(df, indicators) -> pd.Series)
        """
        self.data = data
        self.indicators = indicators
        self.signal_generators = signal_generators
        self.results = {}

    def compute_indicators(self):
        """Compute all indicators for all tickers."""
        self.indicator_results = {}
        for ticker, df in self.data.items():
            self.indicator_results[ticker] = {}
            for name, func in self.indicators.items():
                self.indicator_results[ticker][name] = func(df)

    def generate_signals(self):
        """Generate all signals for all tickers using computed indicators."""
        self.signal_results = {}
        for ticker, df in self.data.items():
            self.signal_results[ticker] = {}
            for name, func in self.signal_generators.items():
                self.signal_results[ticker][name] = func(df, self.indicator_results[ticker])

    def backtest_signals(self, trade_type: str = "long", signal_names: List[str] = None):
        """Backtest selected signals for all tickers and return performance metrics."""
        if signal_names is None:
            signal_names = list(self.signal_generators.keys())
        self.results[trade_type] = {}
        for ticker in self.data:
            # Combine signals (AND logic)
            combined_signal = pd.Series(True, index=self.data[ticker].index)
            for name in signal_names:
                combined_signal &= self.signal_results[ticker][name]
            # Simple backtest: buy when signal, hold, sell when not
            perf = self._simple_backtest(self.data[ticker], combined_signal, trade_type)
            self.results[trade_type][ticker] = perf
        return self.results[trade_type]

    def optimize_signals(self, trade_type: str = "long", max_signals: int = 3):
        """Find best combination of signals for given trade type."""
        best_combo = None
        best_perf = -float("inf")
        signal_names = list(self.signal_generators.keys())
        for combo in itertools.combinations(signal_names, max_signals):
            perf = self.backtest_signals(trade_type, list(combo))
            avg_perf = sum([v["total_return"] for v in perf.values()]) / len(perf)
            if avg_perf > best_perf:
                best_perf = avg_perf
                best_combo = combo
        return best_combo, best_perf

    @staticmethod
    def _simple_backtest(df: pd.DataFrame, signal: pd.Series, trade_type: str):
        # Example: buy when signal is True, sell when False
        # For simplicity, assume buy at open, sell at close
        returns = []
        in_trade = False
        entry_price = 0
        for i in range(len(df)):
            if signal.iloc[i] and not in_trade:
                entry_price = df["open"].iloc[i]
                in_trade = True
            elif not signal.iloc[i] and in_trade:
                exit_price = df["close"].iloc[i]
                returns.append(
                    (exit_price - entry_price)
                    if trade_type == "long"
                    else (entry_price - exit_price)
                )
                in_trade = False
        total_return = sum(returns)
        return {"total_return": total_return, "trades": len(returns)}

    def walk_forward_backtest(
        self,
        trade_type: str = "long",
        signal_names: List[str] = None,
        window: int = 252,
        step: int = 21,
    ):
        """
        Walk-forward optimization: rolling out-of-sample backtest.
        window: lookback period (e.g., 252 trading days)
        step: step size for rolling window (e.g., 21 trading days)
        Returns performance metrics for each window.
        """
        if signal_names is None:
            signal_names = list(self.signal_generators.keys())
        results = []
        for ticker in self.data:
            df = self.data[ticker]
            signals = self.signal_results[ticker]
            for start in range(0, len(df) - window, step):
                end = start + window
                combined_signal = pd.Series(True, index=df.index[start:end])
                for name in signal_names:
                    combined_signal &= signals[name].iloc[start:end]
                perf = self._simple_backtest(df.iloc[start:end], combined_signal, trade_type)
                results.append({"ticker": ticker, "start": start, "end": end, **perf})
        return results

    def monte_carlo_backtest(
        self, trade_type: str = "long", signal_names: List[str] = None, n_sim: int = 1000
    ):
        """
        Monte Carlo simulation: randomize trade sequences to estimate risk/return distribution.
        n_sim: number of simulations
        Returns distribution of total returns.
        """
        import numpy as np

        if signal_names is None:
            signal_names = list(self.signal_generators.keys())
        results = []
        for ticker in self.data:
            df = self.data[ticker]
            signals = self.signal_results[ticker]
            combined_signal = pd.Series(True, index=df.index)
            for name in signal_names:
                combined_signal &= signals[name]
            returns = []
            in_trade = False
            entry_price = 0
            trade_returns = []
            for i in range(len(df)):
                if combined_signal.iloc[i] and not in_trade:
                    entry_price = df["open"].iloc[i]
                    in_trade = True
                elif not combined_signal.iloc[i] and in_trade:
                    exit_price = df["close"].iloc[i]
                    trade_returns.append(
                        (exit_price - entry_price)
                        if trade_type == "long"
                        else (entry_price - exit_price)
                    )
                    in_trade = False
            trade_returns = np.array(trade_returns)
            sim_results = []
            for _ in range(n_sim):
                np.random.shuffle(trade_returns)
                sim_results.append(trade_returns.sum())
            results.append(
                {
                    "ticker": ticker,
                    "mean_return": np.mean(sim_results),
                    "std_return": np.std(sim_results),
                    "min_return": np.min(sim_results),
                    "max_return": np.max(sim_results),
                }
            )
        return results

    def risk_parity_backtest(self, trade_type: str = "long", signal_names: List[str] = None):
        """
        Multi-factor portfolio backtest using risk parity allocation.
        Returns portfolio performance metrics.
        """
        import numpy as np

        if signal_names is None:
            signal_names = list(self.signal_generators.keys())
        returns_dict = {}
        for ticker in self.data:
            df = self.data[ticker]
            signals = self.signal_results[ticker]
            combined_signal = pd.Series(True, index=df.index)
            for name in signal_names:
                combined_signal &= signals[name]
            perf = self._simple_backtest(df, combined_signal, trade_type)
            returns_dict[ticker] = perf["total_return"]
        # Risk parity allocation
        returns = np.array(list(returns_dict.values()))
        risk = np.std(returns)
        weights = 1 / (risk + 1e-8)
        weights /= weights.sum()
        portfolio_return = (returns * weights).sum()
        return {
            "portfolio_return": portfolio_return,
            "weights": dict(zip(returns_dict.keys(), weights)),
        }

    def regime_switching_backtest(
        self, trade_type: str = "long", signal_names: List[str] = None, n_states: int = 2
    ):
        """
        Regime-switching model (e.g., Markov regime switching) for signal backtesting.
        n_states: number of market regimes
        Returns performance by regime.
        """
        from sklearn.mixture import GaussianMixture

        if signal_names is None:
            signal_names = list(self.signal_generators.keys())
        results = []
        for ticker in self.data:
            df = self.data[ticker]
            returns = df["close"].pct_change().dropna().values.reshape(-1, 1)
            gm = GaussianMixture(n_components=n_states, random_state=42).fit(returns)
            regimes = gm.predict(returns)
            signals = self.signal_results[ticker]
            for regime in range(n_states):
                idx = regimes == regime
                combined_signal = pd.Series(True, index=df.index[1:][idx])
                for name in signal_names:
                    combined_signal &= signals[name].iloc[1:][idx]
                perf = self._simple_backtest(df.iloc[1:][idx], combined_signal, trade_type)
                results.append({"ticker": ticker, "regime": regime, **perf})
        return results

    def ml_ensemble_backtest(
        self,
        trade_type: str = "long",
        signal_names: List[str] = None,
        model_type: str = "RandomForest",
    ):
        """
        Machine learning ensemble (RandomForest, GradientBoosting) for signal generation and backtesting.
        Returns out-of-sample performance.
        """
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

        if signal_names is None:
            signal_names = list(self.signal_generators.keys())
        results = []
        for ticker in self.data:
            df = self.data[ticker]
            X = pd.DataFrame(
                {name: self.signal_results[ticker][name].astype(int) for name in signal_names}
            )
            y = (df["close"].shift(-1) > df["close"]).astype(int)  # next-day up move
            X = X.iloc[:-1]
            y = y.iloc[:-1]
            if model_type == "RandomForest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            preds = model.predict(X)
            perf = self._simple_backtest(
                df.iloc[:-1], pd.Series(preds.astype(bool), index=X.index), trade_type
            )
            results.append({"ticker": ticker, "model": model_type, **perf})
        return results

    def transaction_cost_backtest(
        self,
        trade_type: str = "long",
        signal_names: List[str] = None,
        cost_per_trade: float = 0.0005,
    ):
        """
        Backtest with transaction cost and slippage modeling.
        cost_per_trade: fraction of trade value lost to costs
        Returns net performance.
        """
        if signal_names is None:
            signal_names = list(self.signal_generators.keys())
        results = []
        for ticker in self.data:
            df = self.data[ticker]
            signals = self.signal_results[ticker]
            combined_signal = pd.Series(True, index=df.index)
            for name in signal_names:
                combined_signal &= signals[name]
            returns = []
            in_trade = False
            entry_price = 0
            for i in range(len(df)):
                if combined_signal.iloc[i] and not in_trade:
                    entry_price = df["open"].iloc[i]
                    in_trade = True
                elif not combined_signal.iloc[i] and in_trade:
                    exit_price = df["close"].iloc[i]
                    trade_return = (
                        (exit_price - entry_price)
                        if trade_type == "long"
                        else (entry_price - exit_price)
                    )
                    trade_return -= cost_per_trade * entry_price  # subtract cost
                    returns.append(trade_return)
                    in_trade = False
            total_return = sum(returns)
            results.append({"ticker": ticker, "net_return": total_return, "trades": len(returns)})
        return results
