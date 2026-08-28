"""뉴스 사건을 2025-10-23까지 주가와 연결할 수 있도록 가격 범위를 보정합니다.

SNS 감성분석/novelty/placebo는 다시 실행하지 않습니다.
- daily_prices.csv가 2025-10-23보다 짧으면 yfinance로 가격만 다시 받습니다.
- 기존 Track1 CLEAN 사건일을 제외한 robust baseline으로 daily_prices_scored.csv만 재계산합니다.
- 그 뒤 refresh_track2_news.py를 다시 실행하면 뉴스 사건이 정상적으로 event_date를 갖습니다.
"""
from __future__ import annotations

import pandas as pd

from market_mover import config
from market_mover.impact import compute_price_features
from market_mover.load_prices import download_daily_prices

TARGET_END = pd.Timestamp("2025-10-23")
DOWNLOAD_END_EXCLUSIVE = "2025-10-24"
DOWNLOAD_START = "2022-11-01"


def _read_dates(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def main():
    price_path = config.INTERIM_DIR / "daily_prices.csv"
    scored_path = config.INTERIM_DIR / "daily_prices_scored.csv"
    events_path = config.PROCESSED_DIR / "events_scored.csv"

    prices = _read_dates(price_path)
    current_max = prices["date"].max() if not prices.empty and "date" in prices.columns else pd.NaT

    if pd.isna(current_max) or current_max.normalize() < TARGET_END:
        print(f"[가격 범위 보정] 현재 daily_prices 최대일: {current_max if pd.notna(current_max) else '없음'}")
        print(f"[가격 범위 보정] {DOWNLOAD_START} ~ {TARGET_END.date()} 가격만 다시 받습니다. SNS 감성분석은 하지 않습니다.")
        prices = download_daily_prices(start=DOWNLOAD_START, end=DOWNLOAD_END_EXCLUSIVE)
        price_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(price_path, index=False)
    else:
        print(f"[가격 범위 확인] daily_prices 최대일 {current_max.date()} → 이미 충분합니다.")

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices = prices[prices["date"].le(TARGET_END)].copy()

    exclude_dates = None
    if events_path.exists():
        events = pd.read_csv(events_path)
        track = events.get("track", pd.Series(index=events.index, dtype=str)).astype(str)
        clean = events[track.eq("track1_auto")].copy()
        if "contamination_level" in clean.columns:
            clean = clean[clean["contamination_level"].astype(str).eq("CLEAN")]
        if not clean.empty and "event_date" in clean.columns and "ticker" in clean.columns:
            clean["event_date"] = pd.to_datetime(clean["event_date"], errors="coerce")
            clean = clean.dropna(subset=["event_date", "ticker"])
            exclude_dates = {
                ticker: group["event_date"].tolist()
                for ticker, group in clean.groupby("ticker")
            }

    scored = compute_price_features(prices, exclude_dates_by_ticker=exclude_dates)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_path, index=False)

    print(f"[완료] daily_prices_scored 최대일: {pd.to_datetime(scored['date']).max().date()}")
    print(f"[완료] 저장: {scored_path}")
    print("다음으로 refresh_track2_news.py를 실행하면 됩니다.")


if __name__ == "__main__":
    main()
