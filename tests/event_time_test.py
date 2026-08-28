import pandas as pd

from market_mover.event_windows import align_post_to_trading_day, prepare_track2_events


def _assert_case(timestamp, expected_date, expected_session, expected_quality=None):
    trading_days = pd.DatetimeIndex(
        pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-07-02", "2025-07-03", "2025-07-07"])
    )
    result = align_post_to_trading_day(timestamp, trading_days)
    assert result["event_date"] == pd.Timestamp(expected_date), result
    assert result["market_session"] == expected_session, result
    if expected_quality is not None:
        assert result["daily_alignment_quality"] == expected_quality, result


def main():
    # 겨울: EST(UTC-5)
    _assert_case("2025-01-02 13:00:00+00:00", "2025-01-02", "premarket")
    _assert_case(
        "2025-01-02 15:00:00+00:00",
        "2025-01-02",
        "regular_session",
        "PARTIAL_DAY_INTRADAY_PREFERRED",
    )
    _assert_case("2025-01-02 22:00:00+00:00", "2025-01-03", "afterhours")
    _assert_case("2025-01-03 22:00:00+00:00", "2025-01-06", "afterhours")
    _assert_case("2025-01-04 17:00:00+00:00", "2025-01-06", "market_closed")

    # UTC로는 다음 날이지만 ET로는 전날 장 마감 후인 경우도 다음 거래일로 정렬됩니다.
    _assert_case("2025-01-03 01:00:00+00:00", "2025-01-03", "afterhours")

    # 여름: EDT(UTC-4). 7월 4일 휴장 뒤 다음 거래일로 이동합니다.
    _assert_case("2025-07-03 21:00:00+00:00", "2025-07-07", "afterhours")
    _assert_case("2025-07-05 14:00:00+00:00", "2025-07-07", "market_closed")

    track2 = pd.DataFrame(
        [
            {
                "event_id": "date_only",
                "person": "Trump",
                "posted_at": "2025-01-02T00:00:00",
                "platform": "Truth Social",
                "topic": "tariff",
                "ticker": "QQQ",
                "source_url": "https://example.com/date",
                "description": "날짜만 알려진 사건",
            },
            {
                "event_id": "exact_utc",
                "person": "Musk",
                "posted_at": "2025-01-03T01:00:00",
                "posted_at_timezone": "UTC",
                "timestamp_precision": "exact",
                "platform": "X",
                "topic": "ai",
                "ticker": "QQQ",
                "source_url": "https://example.com/exact",
                "description": "UTC 시각이 검증된 사건",
            },
        ]
    )
    prices = pd.DataFrame(
        {
            "ticker": ["QQQ", "QQQ", "QQQ"],
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
        }
    )
    prepared = prepare_track2_events(track2, prices).set_index("event_id")
    assert prepared.loc["date_only", "market_session"] == "unknown_date_only"
    assert prepared.loc["date_only", "daily_alignment_quality"] == "DATE_ONLY_MANUAL_REVIEW"
    assert prepared.loc["exact_utc", "market_session"] == "afterhours"
    assert prepared.loc["exact_utc", "event_date"] == pd.Timestamp("2025-01-03")
    print("event time alignment test passed")


if __name__ == "__main__":
    main()
