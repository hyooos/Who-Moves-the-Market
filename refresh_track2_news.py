"""SNS 감성분석을 다시 돌리지 않고 뉴스 사건만 대시보드 데이터에 반영합니다.

필요한 기존 결과:
- data/interim/daily_prices_scored.csv
- data/processed/events_scored.csv (Track 1 사건 포함)

뉴스 입력:
- data/interim/track2_news_auto_events.csv
- data/manual/track2_curated_events.csv (기존 6개 검증 사례, 있으면 함께 유지)

실행 후 dashboard_app.py / run_streamlit.bat에서 바로 읽을 수 있는
processed/events_scored.csv를 갱신합니다.
"""

from __future__ import annotations

import shutil

import pandas as pd

from market_mover import config
from market_mover.event_windows import add_daily_event_windows, prepare_track2_events


def load_csv(path, required=False):
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return pd.DataFrame()
    return pd.read_csv(path)


def prepare_news(df: pd.DataFrame, prices: pd.DataFrame, scored_prices: pd.DataFrame, track_label: str) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()

    # 자동 수집 뉴스는 미국 동부시간의 DST 때문에 같은 파일 안에
    # -05:00(EST)와 -04:00(EDT) 오프셋이 함께 존재할 수 있습니다.
    # pandas 2.x는 이러한 mixed-timezone 문자열을 utc=True 없이 한 번에
    # 파싱하면 ValueError를 낼 수 있으므로, 자동 뉴스만 먼저 UTC로 통일합니다.
    # 수동 뉴스의 naive 시각은 기존 TRACK2_DEFAULT_TIMEZONE 규칙을 유지합니다.
    if track_label == "track2_news_auto" and "posted_at" in work.columns:
        try:
            parsed = pd.to_datetime(work["posted_at"], errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(work["posted_at"], errors="coerce", utc=True)
        # prepare_track2_events 내부의 기존 pd.to_datetime 호출에서도
        # 혼합 오프셋 문제가 재발하지 않도록 동일한 UTC(Z) 문자열로 넘깁니다.
        work["posted_at"] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    prepared = prepare_track2_events(work, prices)
    prepared = add_daily_event_windows(prepared, scored_prices)
    prepared["track"] = track_label
    # 뉴스는 의도적으로 감성분석 대상에서 제외합니다.
    for col in ["sentiment_label", "sentiment_score", "sentiment_confidence", "sentiment_model"]:
        prepared[col] = pd.NA
    return prepared


def remove_auto_duplicates(auto: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    if auto.empty or manual.empty:
        return auto
    auto = auto.copy()
    manual = manual.copy()
    auto_date = pd.to_datetime(auto.get("calendar_date"), errors="coerce").dt.normalize()
    manual_date = pd.to_datetime(manual.get("calendar_date"), errors="coerce").dt.normalize()
    keep = []
    for idx, row in auto.iterrows():
        same = (
            manual.get("person", pd.Series(index=manual.index, dtype=str)).astype(str).eq(str(row.get("person")))
            & manual.get("ticker", pd.Series(index=manual.index, dtype=str)).astype(str).eq(str(row.get("ticker")))
        )
        if "topic" in manual.columns:
            same &= manual["topic"].astype(str).eq(str(row.get("topic")))
        date = auto_date.loc[idx]
        if pd.notna(date):
            same &= (manual_date - date).abs().dt.days.le(1)
        keep.append(not bool(same.any()))
    return auto.loc[keep].reset_index(drop=True)


def main() -> None:
    scored_price_path = config.INTERIM_DIR / "daily_prices_scored.csv"
    existing_events_path = config.PROCESSED_DIR / "events_scored.csv"
    if not scored_price_path.exists() or not existing_events_path.exists():
        raise FileNotFoundError(
            "기존 분석 결과가 필요합니다. 먼저 한 번만 run_daily_pipeline.py를 실행해 "
            "daily_prices_scored.csv와 events_scored.csv를 만들어주세요."
        )

    scored_prices = pd.read_csv(scored_price_path, parse_dates=["date"])
    # prepare_track2_events에는 ticker별 거래일만 있으면 되므로 scored_prices를 그대로 사용해도 됩니다.
    prices = scored_prices.copy()
    existing = pd.read_csv(existing_events_path)
    track = existing.get("track", pd.Series(index=existing.index, dtype=str)).astype(str)
    track1 = existing[track.eq("track1_auto")].copy()

    manual_path = config.MANUAL_DIR / "track2_curated_events.csv"
    auto_path = config.INTERIM_DIR / "track2_news_auto_events.csv"
    manual_raw = load_csv(manual_path)
    auto_raw = load_csv(auto_path, required=True)

    manual = prepare_news(manual_raw, prices, scored_prices, "track2_manual") if not manual_raw.empty else pd.DataFrame()
    auto = prepare_news(auto_raw, prices, scored_prices, "track2_news_auto") if not auto_raw.empty else pd.DataFrame()
    auto = remove_auto_duplicates(auto, manual)

    parts = [frame for frame in [track1, manual, auto] if not frame.empty]
    final = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()

    # Track1은 YYYY-MM-DD, Track2는 YYYY-MM-DD 00:00:00 형태로 저장될 수 있습니다.
    # pandas 2.x가 혼합 형식을 엄격하게 파싱해 Track2만 NaT가 되는 문제를 막기 위해
    # 대시보드 입력 파일에서는 사건 거래일 형식을 하나로 통일합니다.
    if not final.empty and "event_date" in final.columns:
        try:
            parsed_event_date = pd.to_datetime(final["event_date"], errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed_event_date = pd.to_datetime(final["event_date"], errors="coerce")
        final["event_date"] = parsed_event_date.dt.strftime("%Y-%m-%d")

    backup = config.PROCESSED_DIR / "events_scored_before_news_refresh.csv"
    if existing_events_path.exists() and not backup.exists():
        shutil.copy2(existing_events_path, backup)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(existing_events_path, index=False)
    final.to_csv(config.TABLE_DIR / "events_scored.csv", index=False)

    print("[뉴스 반영 완료]")
    print(f"Track1 SNS 사건: {len(track1):,}건 (재분석 안 함)")
    print(f"기존 검증 뉴스 사례: {len(manual):,}건")
    print(f"자동 수집 뉴스 사건: {len(auto):,}건")
    print(f"최종 대시보드 사건: {len(final):,}건")
    print(f"저장: {existing_events_path}")
    print("이제 run_streamlit.bat만 실행하면 됩니다.")


if __name__ == "__main__":
    main()
