import pandas as pd

from . import config
from .topic_rules import assign_topic, is_market_relevant


def preprocess_posts(posts: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(config.TRACK1_START)
    end = pd.Timestamp(config.TRACK1_END) + pd.Timedelta(days=1)
    out = posts[(posts["posted_at"] >= start) & (posts["posted_at"] < end)].copy()
    cutoff = pd.Timestamp(config.MUSK_TWITTER_ACQUISITION) + pd.Timedelta(days=60)
    out = out[~((out["person"] == "Musk") & (out["posted_at"] < cutoff))].copy()
    out["topic"] = out.apply(lambda row: assign_topic(row["text_clean"], row["person"]), axis=1)
    out["is_market_relevant"] = out.apply(
        lambda row: is_market_relevant(row["text_clean"], row["person"]),
        axis=1,
    )
    before = len(out)
    out = out[out["is_market_relevant"]].copy()
    ratio = len(out) / before if before else 0
    print(f"[전처리] market-relevant 필터링: {before} -> {len(out)} ({ratio:.1%})")
    if ratio < 0.03:
        print("[전처리] 경고: 필터링 비율이 3% 미만입니다. 키워드가 너무 좁을 수 있습니다.")
    elif ratio > 0.30:
        print("[전처리] 경고: 필터링 비율이 30% 초과입니다. 키워드가 너무 넓을 수 있습니다.")
    out["post_id"] = [f"post_{idx:06d}" for idx in range(len(out))]
    return out.reset_index(drop=True)


def summarize_filtering(raw_posts: pd.DataFrame, filtered_posts: pd.DataFrame) -> pd.DataFrame:
    raw_counts = raw_posts.groupby("person").size().rename("raw_posts")
    filtered_counts = filtered_posts.groupby("person").size().rename("filtered_posts")
    summary = pd.concat([raw_counts, filtered_counts], axis=1).fillna(0).astype(int)
    summary["filter_rate"] = summary["filtered_posts"] / summary["raw_posts"].replace(0, pd.NA)
    return summary.reset_index()
