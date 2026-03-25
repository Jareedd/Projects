import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def load_price_data(path: str) -> pd.DataFrame:

    standard = pd.read_csv(path)
    if {"Date", "Close"}.issubset(standard.columns):
        df = standard.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        return df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    multi = pd.read_csv(path, header=[0, 1])
    multi.columns = [col[0] if isinstance(col, tuple) else col for col in multi.columns]
    multi = multi.rename(columns={"Price": "Date"})

    if not {"Date", "Close"}.issubset(multi.columns):
        raise ValueError("CSV must contain Date and Close columns.")

    multi = multi[multi["Date"] != "Date"].copy()
    multi["Date"] = pd.to_datetime(multi["Date"], errors="coerce")
    multi["Close"] = pd.to_numeric(multi["Close"], errors="coerce")
    return multi.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_peak = equity_curve.cummax()
    drawdown = (equity_curve / rolling_peak) - 1.0
    return drawdown.min()


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    trading_days = 252
    excess = daily_returns - (risk_free_rate / trading_days)
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return np.sqrt(trading_days) * excess.mean() / vol


def run_backtest(price_df: pd.DataFrame, fast: int, slow: int, cost_bps: float = 5.0) -> pd.DataFrame:
    df = price_df.copy()
    df["Daily_Return"] = df["Close"].pct_change().fillna(0.0)
    df["SMA_Fast"] = df["Close"].rolling(window=fast, min_periods=fast).mean()
    df["SMA_Slow"] = df["Close"].rolling(window=slow, min_periods=slow).mean()

    df["Signal"] = np.where(df["SMA_Fast"] > df["SMA_Slow"], 1, -1)
    df.loc[df["SMA_Slow"].isna(), "Signal"] = 0
    df["Position"] = df["Signal"].shift(1).fillna(0)

    trade_cost = cost_bps / 10000
    df["Trade"] = (df["Position"] != df["Position"].shift(1)).astype(int)
    df["Strategy_Return"] = df["Position"] * df["Daily_Return"] - (df["Trade"] * trade_cost)
    df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()

    df["Benchmark_Return"] = df["Daily_Return"]
    df["Benchmark_Equity"] = (1 + df["Benchmark_Return"]).cumprod()
    return df


def build_ml_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df["Daily_Return"] = df["Close"].pct_change()

    df["ret_1"] = df["Daily_Return"]
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)
    df["vol_10"] = df["Daily_Return"].rolling(10).std()
    df["vol_20"] = df["Daily_Return"].rolling(20).std()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_30"] = df["Close"].rolling(30).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_30"] - 1.0
    df["mom_10"] = df["Close"] / df["Close"].shift(10) - 1.0

    df["Target"] = (df["Daily_Return"].shift(-1) > 0).astype(int)
    return df


def fit_ml_model(train_feat_df: pd.DataFrame, feature_cols: list, model_cfg: dict) -> Pipeline:
    X_train = train_feat_df[feature_cols]
    y_train = train_feat_df["Target"]

    if model_cfg["model"] == "logistic":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=model_cfg["c"], max_iter=2000, random_state=42)),
            ]
        )
    elif model_cfg["model"] == "random_forest":
        model = Pipeline(
            steps=[
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=model_cfg["n_estimators"],
                        max_depth=model_cfg["max_depth"],
                        min_samples_leaf=model_cfg["min_samples_leaf"],
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_cfg['model']}")

    model.fit(X_train, y_train)
    return model


def run_ml_backtest(
    feat_df: pd.DataFrame,
    model: Pipeline,
    feature_cols: list,
    prob_threshold: float = 0.55,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    df = feat_df.copy()
    probs = model.predict_proba(df[feature_cols])[:, 1]
    df["Prob_Up"] = probs

    df["Signal"] = 0
    df.loc[df["Prob_Up"] >= prob_threshold, "Signal"] = 1
    df.loc[df["Prob_Up"] <= (1 - prob_threshold), "Signal"] = -1
    df["Position"] = df["Signal"].shift(1).fillna(0)

    trade_cost = cost_bps / 10000
    df["Trade"] = (df["Position"] != df["Position"].shift(1)).astype(int)
    df["Strategy_Return"] = df["Position"] * df["Daily_Return"].fillna(0.0) - (df["Trade"] * trade_cost)
    df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()

    df["Benchmark_Return"] = df["Daily_Return"].fillna(0.0)
    df["Benchmark_Equity"] = (1 + df["Benchmark_Return"]).cumprod()
    return df


def tune_ml_threshold(train_feat_df: pd.DataFrame, feature_cols: list, thresholds: list, model_cfg: dict) -> float:
    #  first part for fitting, last part for validation.
    split_idx = int(len(train_feat_df) * 0.8)
    fit_df = train_feat_df.iloc[:split_idx].copy()
    val_df = train_feat_df.iloc[split_idx:].copy()

    if len(fit_df) == 0 or len(val_df) == 0:
        return 0.55

    model = fit_ml_model(fit_df, feature_cols, model_cfg=model_cfg)
    best_threshold = 0.55
    best_sharpe = -np.inf

    for threshold in thresholds:
        val_bt = run_ml_backtest(val_df, model, feature_cols, prob_threshold=threshold, cost_bps=5.0)
        val_metrics = compute_metrics(val_bt, "Strategy_Return", "Strategy_Equity")
        if val_metrics["sharpe"] > best_sharpe:
            best_sharpe = val_metrics["sharpe"]
            best_threshold = threshold

    return best_threshold


def walk_forward_score_params(
    dev_feat_df: pd.DataFrame,
    feature_cols: list,
    model_cfg: dict,
    threshold: float,
    train_size: int = 756,
    val_size: int = 126,
    step_size: int = 126,
) -> dict:
    sharpe_list = []
    drawdown_list = []
    turnover_list = []
    window_count = 0
    start = 0

    while start + train_size + val_size <= len(dev_feat_df):
        train_slice = dev_feat_df.iloc[start : start + train_size].dropna(subset=feature_cols + ["Target"]).copy()
        val_slice = dev_feat_df.iloc[start + train_size : start + train_size + val_size].dropna(
            subset=feature_cols + ["Target"]
        ).copy()

        if len(train_slice) < 200 or len(val_slice) < 50:
            start += step_size
            continue

        model = fit_ml_model(train_slice, feature_cols, model_cfg=model_cfg)
        val_bt = run_ml_backtest(val_slice, model, feature_cols, prob_threshold=threshold, cost_bps=5.0)
        metrics = compute_metrics(val_bt, "Strategy_Return", "Strategy_Equity")

        sharpe_list.append(metrics["sharpe"])
        drawdown_list.append(metrics["max_drawdown"])
        turnover_list.append(float(val_bt["Trade"].mean()))
        window_count += 1
        start += step_size

    if window_count == 0:
        return {
            "windows": 0,
            "avg_sharpe": -np.inf,
            "avg_drawdown": -1.0,
            "avg_turnover": 1.0,
            "score": -np.inf,
        }

    avg_sharpe = float(np.mean(sharpe_list))
    avg_drawdown = float(np.mean(drawdown_list))
    avg_turnover = float(np.mean(turnover_list))

    score = avg_sharpe - (0.40 * abs(avg_drawdown)) - (1.20 * avg_turnover)
    return {
        "windows": window_count,
        "avg_sharpe": avg_sharpe,
        "avg_drawdown": avg_drawdown,
        "avg_turnover": avg_turnover,
        "score": score,
    }


def build_model_grid() -> list:
    grid = []
    for c in [0.05, 0.10, 0.30, 1.0, 3.0, 10.0]:
        grid.append({"model": "logistic", "label": f"logistic_C_{c}", "c": c})

    for max_depth in [3, 5, None]:
        for min_samples_leaf in [1, 5, 10]:
            grid.append(
                {
                    "model": "random_forest",
                    "label": f"rf_depth_{max_depth}_leaf_{min_samples_leaf}",
                    "n_estimators": 300,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                }
            )
    return grid


def tune_ml_hyperparams_walk_forward(
    dev_feat_df: pd.DataFrame,
    feature_cols: list,
    model_grid: list,
    threshold_grid: list,
) -> tuple:
    rows = []
    best = {"model_cfg": None, "threshold": None, "score": -np.inf}

    for model_cfg in model_grid:
        for threshold in threshold_grid:
            result = walk_forward_score_params(dev_feat_df, feature_cols, model_cfg=model_cfg, threshold=threshold)
            row = {
                "model": model_cfg["model"],
                "model_label": model_cfg["label"],
                "threshold": threshold,
                "windows": result["windows"],
                "avg_sharpe": result["avg_sharpe"],
                "avg_drawdown": result["avg_drawdown"],
                "avg_turnover": result["avg_turnover"],
                "score": result["score"],
            }
            rows.append(row)
            if row["score"] > best["score"]:
                best = {"model_cfg": model_cfg, "threshold": threshold, "score": row["score"]}

    return best["model_cfg"], best["threshold"], pd.DataFrame(rows).sort_values("score", ascending=False)


def walk_forward_ml_backtest(
    feat_df: pd.DataFrame,
    feature_cols: list,
    train_size: int = 756,  # ~3 years of daily bars
    test_size: int = 126,  # ~6 months
    step_size: int = 126,  # roll every ~6 months
) -> tuple:
    windows = []
    oos_parts = []
    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]

    start = 0
    while start + train_size + test_size <= len(feat_df):
        train_slice = feat_df.iloc[start : start + train_size].dropna(subset=feature_cols + ["Target"]).copy()
        test_slice = feat_df.iloc[start + train_size : start + train_size + test_size].dropna(
            subset=feature_cols + ["Target"]
        ).copy()

        if len(train_slice) < 200 or len(test_slice) < 50:
            start += step_size
            continue

        default_model_cfg = {"model": "logistic", "label": "logistic_C_1.0", "c": 1.0}
        best_threshold = tune_ml_threshold(train_slice, feature_cols, thresholds, model_cfg=default_model_cfg)
        model = fit_ml_model(train_slice, feature_cols, model_cfg=default_model_cfg)
        test_bt = run_ml_backtest(test_slice, model, feature_cols, prob_threshold=best_threshold, cost_bps=5.0)

        strat_metrics = compute_metrics(test_bt, "Strategy_Return", "Strategy_Equity")
        bench_metrics = compute_metrics(test_bt, "Benchmark_Return", "Benchmark_Equity", trade_col="NoTrade")

        windows.append(
            {
                "train_start": train_slice["Date"].iloc[0],
                "train_end": train_slice["Date"].iloc[-1],
                "test_start": test_slice["Date"].iloc[0],
                "test_end": test_slice["Date"].iloc[-1],
                "threshold": best_threshold,
                "strategy_total_return": strat_metrics["total_return"],
                "strategy_sharpe": strat_metrics["sharpe"],
                "benchmark_total_return": bench_metrics["total_return"],
                "benchmark_sharpe": bench_metrics["sharpe"],
            }
        )

        oos_parts.append(test_bt[["Date", "Trade", "Strategy_Return", "Benchmark_Return"]].copy())
        start += step_size

    if not oos_parts:
        return pd.DataFrame(), pd.DataFrame(), {}, {}

    oos_df = pd.concat(oos_parts, ignore_index=True).sort_values("Date").drop_duplicates(subset=["Date"])
    oos_df["Strategy_Equity"] = (1 + oos_df["Strategy_Return"]).cumprod()
    oos_df["Benchmark_Equity"] = (1 + oos_df["Benchmark_Return"]).cumprod()

    summary_strategy = compute_metrics(oos_df, "Strategy_Return", "Strategy_Equity", trade_col="Trade")
    summary_benchmark = compute_metrics(oos_df, "Benchmark_Return", "Benchmark_Equity", trade_col="NoTrade")
    return pd.DataFrame(windows), oos_df, summary_strategy, summary_benchmark


def compute_metrics(df: pd.DataFrame, return_col: str, equity_col: str, trade_col: str = "Trade") -> dict:
    if len(df) == 0:
        return {
            "bars": 0,
            "trades": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    total_return = df[equity_col].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(df)) - 1
    annual_vol = df[return_col].std() * np.sqrt(252)

    return {
        "bars": len(df),
        "trades": int(df[trade_col].sum()) if trade_col in df.columns else 0,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe_ratio(df[return_col]),
        "max_drawdown": max_drawdown(df[equity_col]),
    }


def print_metrics(title: str, metrics: dict) -> None:
    print("-" * 60)
    print(title)
    print("-" * 60)
    print(f"Bars: {metrics['bars']}")
    print(f"Trades: {metrics['trades']}")
    print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
    print(f"Annual Return: {metrics['annual_return'] * 100:.2f}%")
    print(f"Annual Volatility: {metrics['annual_vol'] * 100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")


def optimize_on_train(price_df: pd.DataFrame, fast_grid: list, slow_grid: list) -> tuple:
    best = {"fast": None, "slow": None, "sharpe": -np.inf}
    for fast in fast_grid:
        for slow in slow_grid:
            if fast >= slow:
                continue
            bt = run_backtest(price_df, fast=fast, slow=slow, cost_bps=5.0)
            metrics = compute_metrics(bt, "Strategy_Return", "Strategy_Equity")
            if metrics["sharpe"] > best["sharpe"]:
                best = {"fast": fast, "slow": slow, "sharpe": metrics["sharpe"]}
    return best["fast"], best["slow"], best["sharpe"]


def main() -> None:
    df = load_price_data("AAPL.csv")
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    fast_grid = [10, 15, 20, 25, 30]
    slow_grid = [40, 50, 60, 80, 100]
    best_fast, best_slow, best_train_sharpe = optimize_on_train(train_df, fast_grid, slow_grid)

    train_bt = run_backtest(train_df, fast=best_fast, slow=best_slow, cost_bps=5.0)
    test_bt = run_backtest(test_df, fast=best_fast, slow=best_slow, cost_bps=5.0)

    train_strategy = compute_metrics(train_bt, "Strategy_Return", "Strategy_Equity")
    train_benchmark = compute_metrics(train_bt, "Benchmark_Return", "Benchmark_Equity", trade_col="NoTrade")
    test_strategy = compute_metrics(test_bt, "Strategy_Return", "Strategy_Equity")
    test_benchmark = compute_metrics(test_bt, "Benchmark_Return", "Benchmark_Equity", trade_col="NoTrade")

    print("=" * 60)
    print("SMA CROSSOVER RESEARCH PIPELINE (TRAIN/TEST)")
    print("=" * 60)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Best params from train: fast={best_fast}, slow={best_slow}, train_sharpe={best_train_sharpe:.2f}")

    print_metrics("TRAIN - Strategy", train_strategy)
    print_metrics("TRAIN - Buy and Hold", train_benchmark)
    print_metrics("TEST  - Strategy", test_strategy)
    print_metrics("TEST  - Buy and Hold", test_benchmark)

    print("\nLast 5 rows of TEST:")
    print(
        test_bt[
            ["Date", "Close", "Signal", "Position", "Strategy_Return", "Strategy_Equity", "Benchmark_Equity"]
        ].tail()
    )

    # ML model pipeline (leakage-safe train/test process)
    feat_df = build_ml_features(df)
    feature_cols = ["ret_1", "ret_5", "ret_10", "vol_10", "vol_20", "sma_ratio", "mom_10"]

    train_feat = feat_df.iloc[:split_idx].dropna(subset=feature_cols + ["Target"]).copy()
    test_feat = feat_df.iloc[split_idx:].dropna(subset=feature_cols + ["Target"]).copy()

    default_model_cfg = {"model": "logistic", "label": "logistic_C_1.0", "c": 1.0}
    tuned_threshold = tune_ml_threshold(
        train_feat,
        feature_cols,
        thresholds=[0.50, 0.52, 0.55, 0.58, 0.60],
        model_cfg=default_model_cfg,
    )
    ml_model = fit_ml_model(train_feat, feature_cols, model_cfg=default_model_cfg)
    train_ml_bt = run_ml_backtest(train_feat, ml_model, feature_cols, prob_threshold=tuned_threshold, cost_bps=5.0)
    test_ml_bt = run_ml_backtest(test_feat, ml_model, feature_cols, prob_threshold=tuned_threshold, cost_bps=5.0)

    train_ml_strategy = compute_metrics(train_ml_bt, "Strategy_Return", "Strategy_Equity")
    test_ml_strategy = compute_metrics(test_ml_bt, "Strategy_Return", "Strategy_Equity")
    test_ml_benchmark = compute_metrics(test_ml_bt, "Benchmark_Return", "Benchmark_Equity", trade_col="NoTrade")

    print("\n" + "=" * 60)
    print("ML LOGISTIC REGRESSION BACKTEST (TRAINED ON TRAIN ONLY)")
    print("=" * 60)
    print(f"Chosen threshold from train-validation: {tuned_threshold:.2f}")
    print_metrics("TRAIN - ML Strategy", train_ml_strategy)
    print_metrics("TEST  - ML Strategy", test_ml_strategy)
    print_metrics("TEST  - Buy and Hold", test_ml_benchmark)

    print("\nLast 5 rows of ML TEST:")
    print(
        test_ml_bt[
            ["Date", "Close", "Prob_Up", "Signal", "Position", "Strategy_Return", "Strategy_Equity", "Benchmark_Equity"]
        ].tail()
    )

    # Stronger evaluation protocol:
    # 1) Reserve an untouched final holdout.
    # 2) Tune C + threshold only on development data via walk-forward validation.
    # 3) Train once on development with best params, evaluate once on holdout.
    holdout_split_idx = int(len(feat_df) * 0.80)
    dev_feat = feat_df.iloc[:holdout_split_idx].dropna(subset=feature_cols + ["Target"]).copy()
    holdout_feat = feat_df.iloc[holdout_split_idx:].dropna(subset=feature_cols + ["Target"]).copy()

    model_grid = build_model_grid()
    threshold_grid = [0.50, 0.52, 0.55, 0.58, 0.60]
    best_model_cfg, best_threshold, tuning_table = tune_ml_hyperparams_walk_forward(
        dev_feat, feature_cols, model_grid=model_grid, threshold_grid=threshold_grid
    )

    final_model = fit_ml_model(dev_feat, feature_cols, model_cfg=best_model_cfg)
    holdout_bt = run_ml_backtest(
        holdout_feat, final_model, feature_cols, prob_threshold=best_threshold, cost_bps=5.0
    )
    holdout_strategy = compute_metrics(holdout_bt, "Strategy_Return", "Strategy_Equity")
    holdout_benchmark = compute_metrics(holdout_bt, "Benchmark_Return", "Benchmark_Equity", trade_col="NoTrade")

    holdout_bt["Pred_Direction"] = (holdout_bt["Prob_Up"] > 0.5).astype(int)
    holdout_bt["True_Direction"] = holdout_bt["Target"].astype(int)
    holdout_accuracy = accuracy_score(holdout_bt["True_Direction"], holdout_bt["Pred_Direction"])
    holdout_precision = precision_score(holdout_bt["True_Direction"], holdout_bt["Pred_Direction"], zero_division=0)
    holdout_recall = recall_score(holdout_bt["True_Direction"], holdout_bt["Pred_Direction"], zero_division=0)
    holdout_cm = confusion_matrix(holdout_bt["True_Direction"], holdout_bt["Pred_Direction"], labels=[0, 1])

    tuning_table.to_csv("model_tuning_results.csv", index=False)
    metrics_rows = [
        {
            "section": "holdout_strategy",
            "model_label": best_model_cfg["label"] if best_model_cfg else "none",
            "threshold": best_threshold,
            "bars": holdout_strategy["bars"],
            "trades": holdout_strategy["trades"],
            "total_return": holdout_strategy["total_return"],
            "annual_return": holdout_strategy["annual_return"],
            "annual_vol": holdout_strategy["annual_vol"],
            "sharpe": holdout_strategy["sharpe"],
            "max_drawdown": holdout_strategy["max_drawdown"],
            "accuracy": holdout_accuracy,
            "precision": holdout_precision,
            "recall": holdout_recall,
            "tn": int(holdout_cm[0, 0]),
            "fp": int(holdout_cm[0, 1]),
            "fn": int(holdout_cm[1, 0]),
            "tp": int(holdout_cm[1, 1]),
        },
        {
            "section": "holdout_benchmark",
            "model_label": "buy_and_hold",
            "threshold": np.nan,
            "bars": holdout_benchmark["bars"],
            "trades": holdout_benchmark["trades"],
            "total_return": holdout_benchmark["total_return"],
            "annual_return": holdout_benchmark["annual_return"],
            "annual_vol": holdout_benchmark["annual_vol"],
            "sharpe": holdout_benchmark["sharpe"],
            "max_drawdown": holdout_benchmark["max_drawdown"],
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tp": np.nan,
        },
    ]
    pd.DataFrame(metrics_rows).to_csv("holdout_metrics.csv", index=False)
    holdout_bt[
        [
            "Date",
            "Close",
            "Prob_Up",
            "True_Direction",
            "Pred_Direction",
            "Signal",
            "Position",
            "Strategy_Return",
            "Benchmark_Return",
            "Strategy_Equity",
            "Benchmark_Equity",
        ]
    ].to_csv("holdout_predictions.csv", index=False)

    print("\n" + "=" * 60)
    print("FINAL HOLDOUT TEST (UNTOUCHED DURING TUNING)")
    print("=" * 60)
    if best_model_cfg is None or best_threshold is None:
        print("Not enough data for walk-forward tuning.")
    else:
        print(
            f"Best params from walk-forward validation: model={best_model_cfg['label']}, "
            f"threshold={best_threshold:.2f}"
        )
        print_metrics("HOLDOUT - ML Strategy", holdout_strategy)
        print_metrics("HOLDOUT - Buy and Hold", holdout_benchmark)
        print("-" * 60)
        print("HOLDOUT CLASSIFICATION METRICS")
        print("-" * 60)
        print(f"Accuracy: {holdout_accuracy:.3f}")
        print(f"Precision: {holdout_precision:.3f}")
        print(f"Recall: {holdout_recall:.3f}")
        print("Confusion Matrix [rows=true 0/1, cols=pred 0/1]:")
        print(holdout_cm)
        print("Saved: model_tuning_results.csv, holdout_metrics.csv, holdout_predictions.csv")
        print("\nTop 8 tuning combos by validation score:")
        print(tuning_table.head(8).to_string(index=False))

    wf_windows, wf_oos, wf_strategy, wf_benchmark = walk_forward_ml_backtest(feat_df, feature_cols)

    print("\n" + "=" * 60)
    print("WALK-FORWARD ML SUMMARY (REPEATED OUT-OF-SAMPLE TESTS)")
    print("=" * 60)
    if len(wf_windows) == 0:
        print("Not enough data for walk-forward windows.")
    else:
        print(f"Walk-forward windows: {len(wf_windows)}")
        print_metrics("WALK-FORWARD OOS - ML Strategy", wf_strategy)
        print_metrics("WALK-FORWARD OOS - Buy and Hold", wf_benchmark)
        print("\nWindow-level results:")
        print(
            wf_windows[
                [
                    "test_start",
                    "test_end",
                    "threshold",
                    "strategy_total_return",
                    "strategy_sharpe",
                    "benchmark_total_return",
                    "benchmark_sharpe",
                ]
            ].to_string(index=False)
        )

    fig, axes = plt.subplots(6, 1, figsize=(12, 19), sharex=False)

    axes[0].plot(df["Date"], df["Close"], color="black", linewidth=1.5, label="Close")
    axes[0].axvline(df["Date"].iloc[split_idx], color="red", linestyle="--", label="Train/Test Split")
    axes[0].set_title("AAPL Close Price")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_bt["Date"], train_bt["Strategy_Equity"], label="Train Strategy", linewidth=1.8)
    axes[1].plot(train_bt["Date"], train_bt["Benchmark_Equity"], label="Train Buy&Hold", linewidth=1.8)
    axes[1].set_title(f"Train Equity (fast={best_fast}, slow={best_slow})")
    axes[1].set_ylabel("Equity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(test_bt["Date"], test_bt["Strategy_Equity"], label="Test Strategy", linewidth=1.8)
    axes[2].plot(test_bt["Date"], test_bt["Benchmark_Equity"], label="Test Buy&Hold", linewidth=1.8)
    axes[2].set_title("Test Equity (Unseen Data)")
    axes[2].set_ylabel("Equity")
    axes[2].set_xlabel("Date")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(test_ml_bt["Date"], test_ml_bt["Strategy_Equity"], label="Test ML Strategy", linewidth=1.8)
    axes[3].plot(test_ml_bt["Date"], test_ml_bt["Benchmark_Equity"], label="Test Buy&Hold", linewidth=1.8)
    axes[3].set_title("Test Equity - ML Strategy (Logistic Regression)")
    axes[3].set_ylabel("Equity")
    axes[3].set_xlabel("Date")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    if len(wf_oos) > 0:
        axes[4].plot(wf_oos["Date"], wf_oos["Strategy_Equity"], label="Walk-Forward OOS ML", linewidth=1.8)
        axes[4].plot(wf_oos["Date"], wf_oos["Benchmark_Equity"], label="Walk-Forward OOS Buy&Hold", linewidth=1.8)
        axes[4].set_title("Walk-Forward Out-of-Sample Equity")
        axes[4].set_ylabel("Equity")
        axes[4].set_xlabel("Date")
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
    else:
        axes[4].set_title("Walk-Forward Out-of-Sample Equity (insufficient data)")
        axes[4].axis("off")

    if len(holdout_bt) > 0:
        axes[5].plot(holdout_bt["Date"], holdout_bt["Strategy_Equity"], label="Final Holdout ML", linewidth=1.8)
        axes[5].plot(holdout_bt["Date"], holdout_bt["Benchmark_Equity"], label="Final Holdout Buy&Hold", linewidth=1.8)
        axes[5].set_title("Final Holdout Equity (One-Shot Evaluation)")
        axes[5].set_ylabel("Equity")
        axes[5].set_xlabel("Date")
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
    else:
        axes[5].set_title("Final Holdout Equity (insufficient data)")
        axes[5].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
