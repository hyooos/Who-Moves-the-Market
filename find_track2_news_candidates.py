"""Track 2 큐레이션을 돕는 헬퍼 스크립트.

data/processed/events_scored.csv에서 impact_score 상위 이벤트를 골라, 각 이벤트
날짜 근처에 관련 뉴스가 실제로 있었는지 조회해 보여준다. 이 스크립트는 "찾아주는"
역할만 하고, Track 2에 넣을지 말지는 사람이 직접
outputs/tables/track2_news_candidates.csv를 열어 판단해야 한다.

뉴스 소스 두 가지를 지원한다(--source):
- google (기본값): Google News RSS. 키 불필요. 이 프로젝트 개발 환경에서 실제로
  동작 확인됨(268건 시도 중 147건에서 관련 기사 발견).
- gdelt: GDELT DOC 2.0 API. 더 구조화된 JSON을 주고 뉴스량 추이도 낼 수 있지만,
  이 프로젝트 개발 환경(샌드박스)에서는 요청마다 HTTP 429(rate limit)로 막혔다.
  데이터센터 IP 대역을 막는 경우가 흔해서 그런 것으로 보임 — 조원 개인 네트워크에서는
  될 수도 있으니 시도해볼 가치는 있지만, 기본값으로 쓰지 말 것.

사용 예:
    PYTHONPATH=. .venv/bin/python find_track2_news_candidates.py --top-n 15
    PYTHONPATH=. .venv/bin/python find_track2_news_candidates.py --person Musk --window-days 2
    PYTHONPATH=. .venv/bin/python find_track2_news_candidates.py --source gdelt --top-n 5
"""

import argparse
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import pandas as pd
import requests

from market_mover import config

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def parse_args():
    parser = argparse.ArgumentParser(description="impact_score 상위 이벤트 주변의 관련 뉴스를 자동으로 찾습니다.")
    parser.add_argument("--top-n", type=int, default=20, help="상위 몇 개 이벤트를 검색할지")
    parser.add_argument("--person", choices=["Musk", "Trump"], default=None, help="특정 인물만 검색")
    parser.add_argument("--window-days", type=int, default=3, help="이벤트 날짜 앞뒤로 며칠까지 뉴스를 찾을지")
    parser.add_argument("--articles-per-event", type=int, default=5, help="이벤트당 최대 기사 수")
    parser.add_argument(
        "--contamination",
        default="CLEAN",
        help="이 오염 수준 이하만 검색 (CLEAN, MINOR, MAJOR, all). 기본 CLEAN.",
    )
    parser.add_argument(
        "--source",
        choices=["google", "gdelt"],
        default="google",
        help="뉴스 소스. google(기본, 검증됨) 또는 gdelt(이 환경에서는 429로 막힘, 다른 네트워크에서 시도 가능).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="요청 사이 대기 시간(초). 기본값은 소스별로 다름(google=1.0, gdelt=5.5 — GDELT는 5초당 1건 제한 명시).",
    )
    return parser.parse_args()


def build_query(row: pd.Series) -> str:
    ticker_names = {
        "TSLA": "Tesla",
        "SPY": "S&P 500",
        "QQQ": "Nasdaq",
        "GM": "General Motors",
        "F": "Ford",
        "RIVN": "Rivian",
    }
    topic_terms = str(row["topic"]).replace("_", " ")
    return f"{row['person']} {topic_terms} {ticker_names.get(row['ticker'], row['ticker'])}"


def fetch_google_news_rss(query: str, start_date: pd.Timestamp, end_date: pd.Timestamp, limit: int) -> list:
    date_range = f"after:{start_date.date()} before:{end_date.date()}"
    url = f"{GOOGLE_NEWS_RSS}?q={quote(query + ' ' + date_range)}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    root = ET.fromstring(response.content)
    items = []
    for item in root.findall(".//item")[:limit]:
        items.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "pub_date": (item.findtext("pubDate") or "").strip(),
                "source": (item.findtext("source") or "").strip(),
            }
        )
    return items


def fetch_gdelt_doc(query: str, start_date: pd.Timestamp, end_date: pd.Timestamp, limit: int) -> list:
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": limit,
        "format": "json",
        "startdatetime": start_date.strftime("%Y%m%d000000"),
        "enddatetime": end_date.strftime("%Y%m%d235959"),
    }
    response = requests.get(GDELT_DOC_API, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 429:
        raise RuntimeError("GDELT rate limit(429) — 데이터센터 IP 차단일 가능성이 높습니다. --source google을 쓰세요.")
    response.raise_for_status()
    data = response.json()
    items = []
    for article in data.get("articles", [])[:limit]:
        items.append(
            {
                "title": article.get("title", ""),
                "link": article.get("url", ""),
                "pub_date": article.get("seendate", ""),
                "source": article.get("domain", ""),
            }
        )
    return items


def main():
    args = parse_args()
    if args.sleep is None:
        args.sleep = 5.5 if args.source == "gdelt" else 1.0
    fetch = fetch_gdelt_doc if args.source == "gdelt" else fetch_google_news_rss
    events_path = config.PROCESSED_DIR / "events_scored.csv"
    if not events_path.exists():
        raise FileNotFoundError("먼저 run_daily_pipeline.py를 실행해서 events_scored.csv를 만드세요.")

    events = pd.read_csv(events_path, parse_dates=["posted_at", "event_date"])
    events = events[events["track"].eq("track1_auto")].copy()
    if args.person:
        events = events[events["person"].eq(args.person)]
    if args.contamination != "all":
        events = events[events["contamination_level"].eq(args.contamination)]

    candidates = events.dropna(subset=["impact_score"]).nlargest(args.top_n, "impact_score")
    if candidates.empty:
        print("검색할 이벤트가 없습니다. --contamination all 로 범위를 넓혀보세요.")
        return

    rows = []
    for _, event in candidates.iterrows():
        query = build_query(event)
        start = event["event_date"] - pd.Timedelta(days=args.window_days)
        end = event["event_date"] + pd.Timedelta(days=args.window_days)
        try:
            articles = fetch(query, start, end, args.articles_per_event)
        except Exception as exc:
            print(f"[뉴스 조회 실패] {event['event_id']} ({query}): {exc}")
            articles = []
        print(f"\n=== {event['event_id']} | {event['person']} | {event['event_date'].date()} | {event['topic']} | impact={event['impact_score']:.2f} ===")
        print(f"게시물: {str(event['text_clean'])[:120]}")
        print(f"검색어: {query}")
        if not articles:
            print("  (관련 뉴스 없음 또는 조회 실패)")
        for article in articles:
            print(f"  - {article['title']} ({article['source']}) {article['link']}")
        rows.append(
            {
                "event_id": event["event_id"],
                "person": event["person"],
                "event_date": event["event_date"],
                "topic": event["topic"],
                "ticker": event["ticker"],
                "impact_score": event["impact_score"],
                "abnormal_return": event.get("abnormal_return"),
                "text_clean": event["text_clean"],
                "search_query": query,
                "n_articles_found": len(articles),
                "articles": " | ".join(f"{a['title']} <{a['link']}>" for a in articles),
            }
        )
        time.sleep(args.sleep)

    out_path = config.TABLE_DIR / "track2_news_candidates.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")
    print("이 CSV에서 관련 뉴스가 실제로 있는 이벤트만 골라 data/manual/track2_curated_events.csv에 직접 옮겨 담으세요.")


if __name__ == "__main__":
    main()
