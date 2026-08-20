import numpy as np
import pandas as pd


def load_fomc_calendar(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "event_type"])
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()
    return df


def add_market_shock_flags(events: pd.DataFrame, prices: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    events = events.drop(columns=["market_shock_z", "market_shock_flag"], errors="ignore")
    spy = prices[prices["ticker"] == "SPY"].sort_values("date").copy()
    if spy.empty:
        events["market_shock_flag"] = False
        events["market_shock_z"] = np.nan
        return events
    spy["spy_return"] = spy["close"].pct_change()
    median = spy["spy_return"].rolling(60, min_periods=20).median()
    mad = (spy["spy_return"] - median).abs().rolling(60, min_periods=20).median() * 1.4826
    spy["market_shock_z"] = (spy["spy_return"] - median) / mad.replace(0, np.nan)
    shock = spy[["date", "market_shock_z"]]
    out = events.merge(shock, left_on="event_date", right_on="date", how="left").drop(columns=["date"])
    out["market_shock_flag"] = out["market_shock_z"].abs() >= threshold
    return out


def add_macro_flags(events: pd.DataFrame, fomc_calendar: pd.DataFrame, days: int = 1) -> pd.DataFrame:
    if fomc_calendar.empty:
        events["macro_overlap_flag"] = False
        return events
    fomc_dates = set(fomc_calendar["date"].dt.normalize())

    def has_overlap(date):
        date = pd.Timestamp(date).normalize()
        return any((date + pd.Timedelta(days=offset)) in fomc_dates for offset in range(-days, days + 1))

    out = events.copy()
    out["macro_overlap_flag"] = out["event_date"].map(has_overlap)
    return out


def add_multi_post_flags(events: pd.DataFrame, hours: int = 24, group_cols=None) -> pd.DataFrame:
    # 인물 단위(person)만 보면 같은 사람이 그날 전혀 다른 주제로 올린 글까지 오염으로
    # 잡혀 CLEAN 표본이 과도하게 줄어든다. 실제로 겹칠 위험이 있는 (인물, 종목) 단위로 좁힌다.
    group_cols = group_cols or ["person", "ticker"]
    out = events.sort_values(group_cols + ["posted_at"]).copy()
    out["multi_post_flag"] = False
    for _, idx in out.groupby(group_cols).groups.items():
        group = out.loc[idx].sort_values("posted_at")
        prev_delta = group["posted_at"].diff().dt.total_seconds().div(3600)
        next_delta = group["posted_at"].diff(-1).abs().dt.total_seconds().div(3600)
        out.loc[group.index, "multi_post_flag"] = (
            prev_delta.le(hours).fillna(False) | next_delta.le(hours).fillna(False)
        )
    return out.sort_index()


def classify_contamination(events: pd.DataFrame, prices: pd.DataFrame, fomc_calendar: pd.DataFrame, multi_post_group_cols=None) -> pd.DataFrame:
    out = add_multi_post_flags(events, group_cols=multi_post_group_cols)
    out = add_macro_flags(out, fomc_calendar)
    out = add_market_shock_flags(out, prices)
    flags = ["multi_post_flag", "macro_overlap_flag", "market_shock_flag"]
    out["contamination_count"] = out[flags].sum(axis=1)
    out["contamination_level"] = out["contamination_count"].map(
        {0: "CLEAN", 1: "MINOR", 2: "MAJOR", 3: "MAJOR"}
    )
    return out
