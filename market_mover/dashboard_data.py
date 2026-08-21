import json
from pathlib import Path

import pandas as pd

from . import config


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dashboard_data() -> dict:
    return {
        "events": read_csv_if_exists(config.PROCESSED_DIR / "events_scored.csv"),
        "daily_prices": read_csv_if_exists(config.INTERIM_DIR / "daily_prices_scored.csv"),
        "stats": read_json_if_exists(config.TABLE_DIR / "stats_results.json"),
        "placebo_summary": read_json_if_exists(config.TABLE_DIR / "placebo_summary.json"),
        "placebo_results": read_csv_if_exists(config.TABLE_DIR / "placebo_results.csv"),
        "topic_audit_summary": read_csv_if_exists(config.TABLE_DIR / "topic_audit_summary.csv"),
        "topic_audit_confusion": read_csv_if_exists(config.TABLE_DIR / "topic_audit_confusion_matrix.csv"),
        "topic_audit_pr": read_csv_if_exists(config.TABLE_DIR / "topic_audit_precision_recall.csv"),
        "rivn_sensitivity": read_csv_if_exists(config.TABLE_DIR / "rivn_sensitivity.csv"),
        "figures": sorted(config.FIGURE_DIR.glob("*.html")),
        "report": config.REPORT_DIR / "market_mover_report.html",
    }


def stats_tests_frame(stats: dict) -> pd.DataFrame:
    rows = []
    for key, value in stats.get("tests", {}).items():
        rows.append(
            {
                "검정": key,
                "방법": value.get("test"),
                "통계량": value.get("statistic"),
                "p-value": value.get("p_value"),
                "FDR 보정 p-value": value.get("p_value_fdr_bh"),
                "FDR 0.05 유의": value.get("reject_fdr_0.05"),
                "상태": value.get("status"),
            }
        )
    return pd.DataFrame(rows)


def h2_pairwise_frame(stats: dict) -> pd.DataFrame:
    comparisons = stats.get("posthoc", {}).get("h2_topic_pairwise", {}).get("comparisons", [])
    return pd.DataFrame(comparisons)


def significant_test_count(stats: dict) -> int:
    return int(
        sum(
            bool(value.get("reject_fdr_0.05"))
            for value in stats.get("tests", {}).values()
            if value.get("reject_fdr_0.05") is not None
        )
    )

