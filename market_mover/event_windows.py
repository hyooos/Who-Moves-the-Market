import pandas as pd

from . import config
from .topic_rules import map_ticker


def get_next_trading_day(date_like, trading_days: pd.DatetimeIndex):
    date = pd.Timestamp(date_like).normalize()
    pos = trading_days.searchsorted(date)
    if pos >= len(trading_days):
        return pd.NaT
    return trading_days[pos]


def align_post_to_trading_day(
    posted_at,
    trading_days: pd.DatetimeIndex,
    assume_timezone: str = config.SOURCE_TIMEZONE,
) -> dict:
    """게시 시각을 미국 동부시간으로 변환하고 일봉 반응 거래일을 정합니다.

    - 정규장 마감 전(장 전/장중): 같은 거래일
    - 정규장 마감 후: 다음 거래일
    - 주말·휴장일: 다음 거래일

    원본 Track 1 시각은 UTC이고, Track 2처럼 offset 없는 수동 시각은 호출부에서
    assume_timezone을 America/New_York으로 넘깁니다. 정규장 중 게시물은 당일
    close-to-close 수익률에 게시 전 움직임도 섞이므로 별도 품질 라벨을 붙입니다.
    """
    ts = pd.Timestamp(posted_at)
    if pd.isna(ts):
        return {
            "posted_at_utc": pd.NaT,
            "posted_at_et": pd.NaT,
            "calendar_date_et": pd.NaT,
            "market_session": "unknown",
            "event_date": pd.NaT,
            "event_date_rule": "invalid_timestamp",
            "daily_alignment_quality": "UNAVAILABLE",
        }
    if ts.tzinfo is None:
        ts = ts.tz_localize(assume_timezone, ambiguous=False, nonexistent="shift_forward")
    posted_at_utc = ts.tz_convert("UTC")
    posted_at_et = posted_at_utc.tz_convert(config.MARKET_TIMEZONE)
    local_date = posted_at_et.tz_localize(None).normalize()

    days = pd.DatetimeIndex(pd.to_datetime(trading_days)).tz_localize(None).normalize().sort_values().unique()
    is_trading_day = local_date in days
    minute = posted_at_et.hour * 60 + posted_at_et.minute
    market_open = config.MARKET_OPEN_HOUR * 60 + config.MARKET_OPEN_MINUTE
    market_close = config.MARKET_CLOSE_HOUR * 60 + config.MARKET_CLOSE_MINUTE

    if not is_trading_day:
        market_session = "market_closed"
        pos = days.searchsorted(local_date, side="left")
        rule = "next_trading_day_after_market_closed"
        quality = "DAILY_ALIGNED"
    elif minute < market_open:
        market_session = "premarket"
        pos = days.searchsorted(local_date, side="left")
        rule = "same_trading_day_after_premarket_post"
        quality = "DAILY_ALIGNED"
    elif minute < market_close:
        market_session = "regular_session"
        pos = days.searchsorted(local_date, side="left")
        rule = "same_trading_day_partial_session"
        quality = "PARTIAL_DAY_INTRADAY_PREFERRED"
    else:
        market_session = "afterhours"
        pos = days.searchsorted(local_date, side="right")
        rule = "next_trading_day_after_close"
        quality = "DAILY_ALIGNED"

    event_date = days[pos] if pos < len(days) else pd.NaT
    return {
        "posted_at_utc": posted_at_utc,
        "posted_at_et": posted_at_et,
        "calendar_date_et": local_date,
        "market_session": market_session,
        "event_date": event_date,
        "event_date_rule": rule,
        "daily_alignment_quality": quality,
    }


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
        aligned = align_post_to_trading_day(post.get("posted_at_utc", post["posted_at"]), trading_days)
        event_date = aligned["event_date"]
        if pd.isna(event_date):
            skipped += 1
            continue
        records.append(
            {
                "event_id": f"tk1_{len(records) + 1:06d}",
                "post_id": post["post_id"],
                "source_post_id": post.get("source_post_id"),
                "source_url": post.get("source_url"),
                "source_file": post.get("source_file"),
                "person": post["person"],
                "posted_at": post["posted_at"],
                "posted_at_utc": aligned["posted_at_utc"],
                "posted_at_et": aligned["posted_at_et"],
                "event_date": event_date,
                "calendar_date": aligned["calendar_date_et"],
                "market_session": aligned["market_session"],
                "event_date_rule": aligned["event_date_rule"],
                "daily_alignment_quality": aligned["daily_alignment_quality"],
                "platform": post["platform"],
                "trump_role": post.get("trump_role"),
                "topic": post["topic"],
                "ticker": ticker,
                "track": "track1_auto",
                "text_raw": post.get("text_raw", post["text_clean"]),
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
    out["posted_at_original"] = out["posted_at"].astype(str)
    out["posted_at"] = pd.to_datetime(out["posted_at"], errors="coerce")
    out = out.dropna(subset=["posted_at", "ticker"])
    if "timestamp_precision" not in out.columns:
        out["timestamp_precision"] = "unverified"
    else:
        out["timestamp_precision"] = out["timestamp_precision"].fillna("unverified").astype(str).str.lower()
    # 00:00:00은 수동 사건 CSV에서 날짜만 알려진 사건의 일반적인 자리표시입니다.
    # 실제 자정 게시물이라고 단정하지 않고 date_only로 낮춰 표시합니다. 명시적으로
    # timestamp_precision=exact를 넣은 행은 자동 변환하지 않습니다.
    midnight_placeholder = (
        out["posted_at"].map(lambda value: pd.Timestamp(value).hour == 0)
        & out["posted_at"].map(lambda value: pd.Timestamp(value).minute == 0)
        & out["posted_at"].map(lambda value: pd.Timestamp(value).second == 0)
        & out["timestamp_precision"].eq("unverified")
    )
    out.loc[midnight_placeholder, "timestamp_precision"] = "date_only"
    aligned_rows = []
    for _, row in out.iterrows():
        trading_days = trading_days_by_ticker.get(row["ticker"])
        if trading_days is None or len(trading_days) == 0:
            aligned_rows.append(align_post_to_trading_day(pd.NaT, pd.DatetimeIndex([])))
        else:
            timezone_value = row.get("posted_at_timezone")
            assume_timezone = (
                str(timezone_value).strip()
                if pd.notna(timezone_value) and str(timezone_value).strip()
                else config.TRACK2_DEFAULT_TIMEZONE
            )
            aligned_rows.append(
                align_post_to_trading_day(row["posted_at"], trading_days, assume_timezone=assume_timezone)
            )
    aligned = pd.DataFrame(aligned_rows, index=out.index)
    for column in aligned.columns:
        out[column] = aligned[column]
    date_only = out["timestamp_precision"].eq("date_only")
    out.loc[date_only, "market_session"] = "unknown_date_only"
    out.loc[date_only, "event_date_rule"] = "date_only_to_next_valid_trading_day"
    out.loc[date_only, "daily_alignment_quality"] = "DATE_ONLY_MANUAL_REVIEW"
    unverified = out["timestamp_precision"].eq("unverified")
    out.loc[unverified, "daily_alignment_quality"] = "MANUAL_TIME_UNVERIFIED"
    # posted_at은 기존 모듈과 정렬 호환성을 위해 timezone-naive UTC로 통일합니다.
    out["posted_at"] = pd.to_datetime(out["posted_at_utc"], utc=True).dt.tz_localize(None)
    out["calendar_date"] = out["calendar_date_et"]
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
