from pathlib import Path

import pandas as pd

from . import config
from .event_windows import add_daily_event_windows
from .impact import compute_price_features
from .stats import run_stats


def run_rivn_sensitivity(events: pd.DataFrame, prices: pd.DataFrame, output_dir: Path) -> dict:
    tsla_events = events[events["ticker"].eq("TSLA")].copy()
    if tsla_events.empty:
        summary = {"status": "생략: TSLA 이벤트가 없습니다."}
        return summary

    original_peers = config.PEER_TICKERS.get("TSLA", []).copy()
    variants = {
        "with_rivn": ["GM", "F", "RIVN"],
        "without_rivn": ["GM", "F"],
    }
    rows = []
    try:
        for variant, peers in variants.items():
            config.PEER_TICKERS["TSLA"] = peers
            scored_prices = compute_price_features(prices)
            rescored = add_daily_event_windows(events, scored_prices)
            stats = run_stats(rescored)
            rows.append(
                {
                    "variant": variant,
                    "peers": ",".join(peers),
                    "tsla_mean_abs_ar": rescored[rescored["ticker"].eq("TSLA")]["abnormal_return"].abs().mean(),
                    "h2_topic_p_value": stats["tests"].get("h2_topic_difference", {}).get("p_value"),
                    "h3_person_p_value": stats["tests"].get("h3_musk_vs_trump", {}).get("p_value"),
                }
            )
    finally:
        config.PEER_TICKERS["TSLA"] = original_peers

    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rivn_sensitivity.csv"
    result.to_csv(out_path, index=False)
    return {"status": "완료", "path": str(out_path), "variants": rows}
