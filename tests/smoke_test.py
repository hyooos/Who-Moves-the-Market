import numpy as np
import pandas as pd

from market_mover.contamination import classify_contamination
from market_mover.event_windows import add_daily_event_windows, build_daily_events
from market_mover.impact import compute_price_features
from market_mover.preprocess import preprocess_posts
from market_mover.stats import run_stats


def make_prices():
    dates = pd.bdate_range("2022-11-01", "2025-04-15")
    tickers = ["TSLA", "SPY", "QQQ", "GM", "F", "RIVN"]
    rows = []
    for ticker_i, ticker in enumerate(tickers):
        base = 100 + ticker_i * 10
        for i, date in enumerate(dates):
            close = base * (1 + 0.0008 * i + 0.01 * np.sin(i / 9))
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000000 + i * 1000 + ticker_i * 10000,
                }
            )
    return pd.DataFrame(rows)


def make_posts():
    return pd.DataFrame(
        [
            {
                "person": "Musk",
                "posted_at": pd.Timestamp("2023-03-04 12:00"),
                "text_clean": "Tesla production and FSD update",
                "platform": "X",
                "trump_role": None,
                "engagement": 1000,
                "sentiment_label": "positive",
                "sentiment_score": 1,
                "sentiment_confidence": 0.9,
                "sentiment_model": "test-model",
                "novelty_score": 0.7,
                "max_prior_similarity": 0.3,
            },
            {
                "person": "Trump",
                "posted_at": pd.Timestamp("2023-03-05 09:00"),
                "text_clean": "China tariff trade policy",
                "platform": "Truth Social",
                "trump_role": "candidate_or_citizen",
                "engagement": 2000,
                "sentiment_label": "negative",
                "sentiment_score": -1,
                "sentiment_confidence": 0.8,
                "sentiment_model": "test-model",
                "novelty_score": 0.4,
                "max_prior_similarity": 0.6,
            },
        ]
    )


def main():
    prices = make_prices()
    posts = preprocess_posts(make_posts())
    scored_prices = compute_price_features(prices)
    events = build_daily_events(posts, prices)
    events = classify_contamination(events, prices, pd.DataFrame(columns=["date", "event_type"]))
    events = add_daily_event_windows(events, scored_prices)
    stats = run_stats(events)
    assert len(posts) == 2
    assert len(events) == 2
    assert "impact_score" in events.columns
    assert "sentiment_label" in events.columns
    assert "exploratory_sentiment_positive_vs_negative" in stats["tests"]
    assert "exploratory_novelty_correlation" in stats["tests"]
    assert "multiple_testing" in stats
    assert "posthoc" in stats
    print("smoke test passed")


if __name__ == "__main__":
    main()
