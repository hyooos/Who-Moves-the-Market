import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from market_mover import config
from market_mover.load_posts import clean_text
from market_mover.topic_rules import assign_topic, is_market_relevant, map_ticker


def parse_args():
    parser = argparse.ArgumentParser(
        description="twscrape로 Musk Track 2 후보 이벤트를 백필합니다. 비공식 도구이므로 1회성 보조 수집에만 사용하세요."
    )
    parser.add_argument("--since", default="2025-04-14")
    parser.add_argument("--until", default=config.FREEZE_DATE)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--out", default=str(config.MANUAL_DIR / "track2_musk_backfill_candidates.csv"))
    return parser.parse_args()


async def fetch(args):
    try:
        from twscrape import API, gather
    except ImportError as exc:
        raise ImportError("twscrape가 설치되어 있지 않습니다. .venv/bin/pip install -r requirements-optional.txt 를 실행하세요.") from exc

    username = os.getenv("TWSCRAPE_USERNAME")
    password = os.getenv("TWSCRAPE_PASSWORD")
    email = os.getenv("TWSCRAPE_EMAIL")
    email_password = os.getenv("TWSCRAPE_EMAIL_PASSWORD", "")
    if not username or not password or not email:
        raise RuntimeError("TWSCRAPE_USERNAME, TWSCRAPE_PASSWORD, TWSCRAPE_EMAIL 환경변수가 필요합니다.")

    api = API()
    await api.pool.add_account(username, password, email, email_password)
    await api.pool.login_all()
    query = f"from:elonmusk since:{args.since} until:{args.until}"
    return await gather(api.search(query, limit=args.limit))


def normalize_tweets(tweets) -> pd.DataFrame:
    rows = []
    for i, tweet in enumerate(tweets, start=1):
        text = clean_text(getattr(tweet, "rawContent", "") or getattr(tweet, "content", ""))
        if not is_market_relevant(text, "Musk"):
            continue
        topic = assign_topic(text, "Musk")
        rows.append(
            {
                "event_id": f"tk2_auto_musk_{i:06d}",
                "person": "Musk",
                "posted_at": getattr(tweet, "date", None),
                "platform": "X",
                "topic": topic,
                "ticker": map_ticker("Musk", topic),
                "source_url": getattr(tweet, "url", ""),
                "description": text[:240],
                "text_clean": text,
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    tweets = asyncio.run(fetch(args))
    out = normalize_tweets(tweets)
    out_path = Path(args.out) if os.path.isabs(args.out) else config.ROOT / args.out
    out.to_csv(out_path, index=False)
    print({"후보_이벤트_수": len(out), "저장_경로": str(out_path)})


if __name__ == "__main__":
    main()
