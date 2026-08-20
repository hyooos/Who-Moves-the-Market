import pandas as pd

from .topic_rules import map_ticker


def get_next_trading_day(date_like, trading_days: pd.DatetimeIndex):
    date = pd.Timestamp(date_like).normalize()
    pos = trading_days.searchsorted(date)
    if pos >= len(trading_days):
        return pd.NaT
    return trading_days[pos]


def build_daily_events(posts: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    trading_days_by_ticker = {
        ticker: pd.DatetimeIndex(group["date"].sort_values().unique())
        for ticker, group in prices.groupby("ticker")
    }
    records = []
    skipped = 0
    for _, post in posts.iterrows():
        ticker = map_ticker(post["person"], post["topic"])
        trading_days = trading_days_by_ticker.get(ticker)
        if trading_days is None or len(trading_days) == 0:
            skipped += 1
            continue
        event_date = get_next_trading_day(post["posted_at"], trading_days)
        if pd.isna(event_date):
            skipped += 1
            continue
        records.append(
            {
                "event_id": f"tk1_{len(records) + 1:06d}",
                "post_id": post["post_id"],
                "person": post["person"],
                "posted_at": post["posted_at"],
                "event_date": event_date,
                "calendar_date": pd.Timestamp(post["posted_at"]).normalize(),
                "platform": post["platform"],
                "trump_role": post.get("trump_role"),
                "topic": post["topic"],
                "ticker": ticker,
                "track": "track1_auto",
                "text_clean": post["text_clean"],
                "engagement": post.get("engagement", 0),
                "sentiment_label": post.get("sentiment_label"),
                "sentiment_score": post.get("sentiment_score"),
                "sentiment_confidence": post.get("sentiment_confidence"),
                "sentiment_model": post.get("sentiment_model"),
                "novelty_score": post.get("novelty_score"),
                "max_prior_similarity": post.get("max_prior_similarity"),
                "is_deleted": post.get("is_deleted", False),
            }
        )
    print(f"[이벤트] Track 1 이벤트 {len(records)}개 생성, {skipped}개 건너뜀")
    return pd.DataFrame(records)


def prepare_track2_events(track2: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if track2.empty:
        return track2
    required = ["event_id", "person", "posted_at", "platform", "topic", "ticker", "source_url", "description"]
    missing = [col for col in required if col not in track2.columns]
    if missing:
        raise ValueError(f"Track 2 CSV에 필요한 컬럼이 없습니다: {missing}")
    trading_days_by_ticker = {
        ticker: pd.DatetimeIndex(group["date"].sort_values().unique())
        for ticker, group in prices.groupby("ticker")
    }
    out = track2.copy()
    out["posted_at"] = pd.to_datetime(out["posted_at"], errors="coerce")
    out = out.dropna(subset=["posted_at", "ticker"])
    event_dates = []
    for _, row in out.iterrows():
        trading_days = trading_days_by_ticker.get(row["ticker"])
        if trading_days is None or len(trading_days) == 0:
            event_dates.append(pd.NaT)
        else:
            event_dates.append(get_next_trading_day(row["posted_at"], trading_days))
    out["event_date"] = event_dates
    out["calendar_date"] = out["posted_at"].dt.normalize()
    out["track"] = "track2_manual"
    out["contamination_level"] = None
    if "text_clean" not in out.columns:
        out["text_clean"] = out.get("description", "")
    return out.dropna(subset=["event_date"]).reset_index(drop=True)


def add_daily_event_windows(events: pd.DataFrame, scored_prices: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    price_cols = [
        "ticker",
        "date",
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
    ]
    events = events.drop(
        columns=[
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
        ],
        errors="ignore",
    )
    merged = events.merge(
        scored_prices[price_cols],
        left_on=["ticker", "event_date"],
        right_on=["ticker", "date"],
        how="left",
    ).drop(columns=["date"])

    before_after = []
    for _, event in merged.iterrows():
        ticker_prices = scored_prices[scored_prices["ticker"] == event["ticker"]].sort_values("date")
        pos = ticker_prices["date"].searchsorted(event["event_date"])
        before = ticker_prices.iloc[max(pos - 5, 0):pos]["volatility"].mean()
        after = ticker_prices.iloc[pos:min(pos + 5, len(ticker_prices))]["volatility"].mean()
        before_after.append((before, after))
    merged[["volatility_before", "volatility_after"]] = pd.DataFrame(
        before_after,
        index=merged.index,
    )
    return merged


def build_intraday_window(
    event: pd.Series,
    intraday_prices: pd.DataFrame,
    minutes_before: int = 60,
    minutes_after: int = 240,
) -> pd.DataFrame:
    event_time = pd.Timestamp(event["posted_at"])
    ticker = event["ticker"]
    start = event_time - pd.Timedelta(minutes=minutes_before)
    end = event_time + pd.Timedelta(minutes=minutes_after)
    window = intraday_prices[
        (intraday_prices["ticker"] == ticker)
        & (intraday_prices["datetime"] >= start)
        & (intraday_prices["datetime"] <= end)
    ].copy()
    if window.empty:
        return window
    base = window.loc[window["datetime"] >= event_time, "close"].iloc[0]
    window["minutes_from_post"] = (window["datetime"] - event_time).dt.total_seconds() / 60
    window["return_from_first_post_price"] = window["close"] / base - 1
    return window
