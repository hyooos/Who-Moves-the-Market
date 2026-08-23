import json

import pandas as pd

from market_mover.event_clustering import cluster_daily_events


def _event(event_id, hour, topic="ai", event_date="2025-01-02", engagement=10):
    timestamp = pd.Timestamp(f"2025-01-02 {hour}:00:00", tz="UTC")
    return {
        "event_id": event_id,
        "post_id": f"post_{event_id}",
        "source_post_id": f"source_{event_id}",
        "source_url": f"https://example.com/{event_id}",
        "person": "Musk",
        "ticker": "QQQ",
        "topic": topic,
        "event_date": pd.Timestamp(event_date),
        "posted_at": timestamp.tz_localize(None),
        "posted_at_utc": timestamp,
        "posted_at_et": timestamp.tz_convert("America/New_York"),
        "market_session": "premarket",
        "text_raw": f"raw {event_id}",
        "text_clean": f"clean {event_id}",
        "engagement": engagement,
    }


def main():
    events = pd.DataFrame(
        [
            _event("a", "00", engagement=10),
            _event("b", "05", engagement=20),
            # 직전 글과는 5시간 차이지만 첫 글과 10시간 차이: chaining 없이 새 cluster.
            _event("c", "10", engagement=30),
            # 같은 시각이어도 topic이 다르면 별도 사건.
            _event("d", "10", topic="semiconductor", engagement=40),
        ]
    )
    clustered = cluster_daily_events(events, window_hours=6)
    assert len(clustered) == 3, clustered
    first = clustered[clustered["cluster_size"].eq(2)].iloc[0]
    assert first["engagement"] == 30
    assert json.loads(first["member_event_ids_json"]) == ["a", "b"]
    assert json.loads(first["member_source_urls_json"]) == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert first["cluster_duration_minutes"] == 300
    assert first["cluster_text_raw"] == "raw a\n\nraw b"

    singleton = cluster_daily_events(events, window_hours=0)
    assert len(singleton) == len(events)
    assert singleton["cluster_size"].eq(1).all()
    print("event clustering test passed")


if __name__ == "__main__":
    main()
