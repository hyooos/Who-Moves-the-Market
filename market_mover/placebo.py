import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contamination import classify_contamination
from .event_windows import add_daily_event_windows
from .stats import run_stats


def _p_value(stats_results: dict, key: str):
    value = stats_results.get("tests", {}).get(key, {})
    return value.get("p_value")


def build_placebo_events(events: pd.DataFrame, prices: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trading_days_by_ticker = {
        ticker: pd.DatetimeIndex(group["date"].sort_values().unique())
        for ticker, group in prices.groupby("ticker")
    }
    actual_dates = {
        ticker: set(pd.to_datetime(group["event_date"]).dt.normalize())
        for ticker, group in events.groupby("ticker")
    }
    rows = []
    price_like_columns = [
        "stock_return",
        "market_return",
        "ar_market",
        "abnormal_return",
        "volatility",
        "log_volume",
        "z_ar",
        "z_volume",
        "z_volatility",
        "impact_score",
        "direction",
        "volatility_before",
        "volatility_after",
        "market_shock_z",
        "market_shock_flag",
        "macro_overlap_flag",
        "multi_post_flag",
        "contamination_count",
        "contamination_level",
    ]
    for _, event in events.iterrows():
        ticker = event["ticker"]
        candidates = trading_days_by_ticker[ticker]
        excluded = actual_dates.get(ticker, set())
        candidates = pd.DatetimeIndex([day for day in candidates if day not in excluded])
        if len(candidates) == 0:
            continue
        fake_date = candidates[int(rng.integers(0, len(candidates)))]
        row = event.drop(labels=price_like_columns, errors="ignore").copy()
        row["event_id"] = f"placebo_{seed}_{event['event_id']}"
        row["posted_at"] = fake_date + pd.Timedelta(hours=12)
        row["event_date"] = fake_date
        row["calendar_date"] = fake_date
        row["track"] = "placebo"
        rows.append(row)
    return pd.DataFrame(rows)


def run_placebo_tests(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    scored_prices: pd.DataFrame,
    fomc_calendar: pd.DataFrame,
    iterations: int,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    clean_events = events[events["contamination_level"].eq("CLEAN")].copy()
    if clean_events.empty:
        result = {"iterations": 0, "status": "생략: CLEAN 이벤트가 없습니다."}
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "placebo_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    rows = []
    for i in range(iterations):
        placebo = build_placebo_events(clean_events, prices, seed + i)
        placebo = classify_contamination(placebo, prices, fomc_calendar)
        placebo = add_daily_event_windows(placebo, scored_prices)
        stats = run_stats(placebo)
        rows.append(
            {
                "iteration": i + 1,
                "n_events": len(placebo),
                "n_clean": stats["n_clean"],
                "mean_impact_score": placebo["impact_score"].mean(),
                "h2_topic_p_value": _p_value(stats, "h2_topic_difference"),
                "h3_person_p_value": _p_value(stats, "h3_musk_vs_trump"),
                "sentiment_p_value": _p_value(stats, "exploratory_sentiment_positive_vs_negative"),
            }
        )

    placebo_results = pd.DataFrame(rows)
    real_stats = run_stats(events)
    real_h2 = _p_value(real_stats, "h2_topic_difference")
    real_h3 = _p_value(real_stats, "h3_musk_vs_trump")
    valid_h2 = placebo_results["h2_topic_p_value"].dropna()
    valid_h3 = placebo_results["h3_person_p_value"].dropna()
    empirical_h2 = None
    empirical_h3 = None
    if real_h2 is not None and len(valid_h2) > 0:
        empirical_h2 = float((valid_h2 <= real_h2).mean())
    if real_h3 is not None and len(valid_h3) > 0:
        empirical_h3 = float((valid_h3 <= real_h3).mean())

    summary = {
        "iterations": int(iterations),
        "real_h2_topic_p_value": real_h2,
        "real_h3_person_p_value": real_h3,
        "placebo_h2_valid_iterations": int(len(valid_h2)),
        "placebo_h3_valid_iterations": int(len(valid_h3)),
        "placebo_h2_share_as_or_more_significant": empirical_h2,
        "placebo_h3_share_as_or_more_significant": empirical_h3,
        "placebo_mean_impact_score_mean": float(placebo_results["mean_impact_score"].mean()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    placebo_results.to_csv(output_dir / "placebo_results.csv", index=False)
    (output_dir / "placebo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
