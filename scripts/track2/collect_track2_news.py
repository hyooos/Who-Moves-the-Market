"""Track 2용 뉴스 원문 메타데이터를 대량 수집합니다.

기본 기간은 프로젝트 전체 분석 기간(2023-01-03 ~ 2025-10-23)입니다.
기사 자체를 사건으로 바로 쓰지 않고, 먼저 article-level CSV로 최대한 많이 모읍니다.
감성분석은 하지 않습니다.

예시
----
# 한 달 샘플 테스트
python scripts/track2/collect_track2_news.py --start 2025-06-01 --end 2025-06-30 --source google

# 전체 기간 수집 (권장: 먼저 샘플 테스트 후 실행)
python scripts/track2/collect_track2_news.py --start 2023-01-03 --end 2025-10-23 --source google

# GDELT까지 같이 시도
python scripts/track2/collect_track2_news.py --start 2023-01-03 --end 2025-10-23 --source both

수집 결과
---------
data/interim/news_articles_raw.csv

같은 명령을 다시 실행하면 이미 완료된 query/window 조합은 기본적으로 건너뜁니다.
--no-resume을 주면 처음부터 다시 조회합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from market_mover import config
from market_mover.topic_rules import assign_topic, map_ticker

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_START = "2023-01-03"
DEFAULT_END = "2025-10-23"

# '많이 모으되 시장과 무관한 일반 연예/정치 기사만 무한정 늘리지 않기' 위한 검색어입니다.
# query_key는 재시작(resume) 상태 식별에도 쓰이므로 가급적 이름을 유지합니다.
SEARCH_QUERIES = {
    "Musk": [
        ("musk_tesla", '"Elon Musk" Tesla'),
        ("musk_tsla_stock", '"Elon Musk" TSLA stock'),
        ("musk_tesla_earnings", '"Elon Musk" Tesla earnings'),
        ("musk_tesla_deliveries", '"Elon Musk" Tesla deliveries'),
        ("musk_tesla_price", '"Elon Musk" Tesla price cuts'),
        ("musk_robotaxi", '"Elon Musk" robotaxi Tesla'),
        ("musk_autopilot", '"Elon Musk" FSD Autopilot Tesla'),
        ("musk_ai", '"Elon Musk" AI xAI Nvidia'),
        ("musk_chips", '"Elon Musk" chips semiconductor Tesla'),
        ("musk_macro", '"Elon Musk" economy rates market'),
        ("musk_trump", '"Elon Musk" Trump Tesla'),
        ("musk_doge", '"Elon Musk" DOGE budget Tesla'),
    ],
    "Trump": [
        ("trump_market", '"Donald Trump" stock market'),
        ("trump_sp500", '"Donald Trump" S&P 500'),
        ("trump_nasdaq", '"Donald Trump" Nasdaq'),
        ("trump_tariff", '"Donald Trump" tariff tariffs'),
        ("trump_china", '"Donald Trump" China tariff trade'),
        ("trump_trade", '"Donald Trump" trade policy exports imports'),
        ("trump_fed", '"Donald Trump" Federal Reserve Powell rates'),
        ("trump_economy", '"Donald Trump" economy inflation jobs market'),
        ("trump_tech", '"Donald Trump" tech regulation chips AI'),
        ("trump_tesla", '"Donald Trump" Tesla'),
        ("trump_musk", '"Donald Trump" "Elon Musk"'),
        ("trump_truth_market", '"Donald Trump" "Truth Social" market'),
    ],
}


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Musk/Trump 관련 시장 뉴스 메타데이터를 대량 수집합니다.")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--person", choices=["Musk", "Trump", "all"], default="all")
    p.add_argument("--source", choices=["google", "gdelt", "both"], default="google")
    p.add_argument("--window-days", type=int, default=31, help="한 번의 검색 기간 길이. 기본 31일")
    p.add_argument("--max-articles", type=int, default=100, help="query/window당 최대 기사 수")
    p.add_argument("--sleep", type=float, default=0.8, help="요청 사이 대기 시간(초)")
    p.add_argument("--no-resume", action="store_true", help="기존 수집 상태를 무시하고 다시 조회")
    p.add_argument("--include-other", action="store_true", help="현재 topic rule에 안 걸린 기사도 보존")
    return p.parse_args()


def make_windows(start: str, end: str, days: int) -> list[Window]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise ValueError("--end는 --start 이후여야 합니다.")
    out: list[Window] = []
    cur = start_ts
    step = max(1, int(days))
    while cur <= end_ts:
        win_end = min(cur + pd.Timedelta(days=step - 1), end_ts)
        out.append(Window(cur, win_end))
        cur = win_end + pd.Timedelta(days=1)
    return out


def _request_with_retry(url: str, *, params=None, timeout=20, retries=3) -> requests.Response:
    last_exc = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (MarketMover academic project)"},
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                wait = 2.5 * (attempt + 1)
                print(f"  HTTP {response.status_code} → {wait:.1f}초 후 재시도")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(str(last_exc) if last_exc else "뉴스 요청 실패")


def fetch_google(query: str, window: Window, limit: int) -> list[dict]:
    # Google News after/before 경계가 엄격할 수 있어 하루씩 넓혀 받은 뒤 실제 날짜로 다시 필터합니다.
    after = (window.start - pd.Timedelta(days=1)).date()
    before = (window.end + pd.Timedelta(days=1)).date()
    q = f"{query} after:{after} before:{before}"
    url = f"{GOOGLE_NEWS_RSS}?q={quote(q)}&hl=en-US&gl=US&ceid=US:en"
    response = _request_with_retry(url)
    root = ET.fromstring(response.content)
    rows = []
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        source = (item.findtext("source") or "").strip()
        rows.append({"title": title, "url": link, "published_at": pub_date, "publisher": source})
    return rows


def fetch_gdelt(query: str, window: Window, limit: int) -> list[dict]:
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": min(250, int(limit)),
        "format": "json",
        "startdatetime": window.start.strftime("%Y%m%d000000"),
        "enddatetime": window.end.strftime("%Y%m%d235959"),
    }
    response = _request_with_retry(GDELT_DOC_API, params=params)
    data = response.json()
    rows = []
    for article in data.get("articles", [])[:limit]:
        rows.append(
            {
                "title": str(article.get("title", "")).strip(),
                "url": str(article.get("url", "")).strip(),
                "published_at": article.get("seendate", ""),
                "publisher": str(article.get("domain", "")).strip(),
            }
        )
    return rows


def normalize_title(value: str) -> str:
    import re

    text = str(value).lower()
    text = re.sub(r"\s+-\s+[^-]{1,80}$", "", text)  # Google News의 ' - 매체명' 꼬리 제거
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def article_id(person: str, published_date: str, title: str, publisher: str) -> str:
    raw = f"{person}|{published_date}|{normalize_title(title)}|{publisher.lower()}"
    return "newsart_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def save_deduped(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    work = df.copy()
    work["published_at"] = pd.to_datetime(work["published_at"], errors="coerce", utc=True)
    work = work.dropna(subset=["published_at", "title"])
    work["article_date"] = work["published_at"].dt.strftime("%Y-%m-%d")
    work["title_norm"] = work["title"].map(normalize_title)
    work = work.drop_duplicates(subset=["article_id"], keep="first")
    # 같은 기사 제목이 여러 검색어에 걸린 경우 하나만 보존하되 query_key는 최초 수집값을 유지합니다.
    work = work.sort_values(["published_at", "person", "title_norm"]).drop_duplicates(
        subset=["person", "article_date", "title_norm", "publisher"], keep="first"
    )
    work = work.sort_values("published_at").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(path, index=False)
    return work


def main() -> None:
    args = parse_args()
    config.ensure_output_folders()
    out_path = config.INTERIM_DIR / "news_articles_raw.csv"
    state_path = config.INTERIM_DIR / "news_collection_state.csv"

    existing = pd.DataFrame() if args.no_resume else load_existing(out_path)
    state = pd.DataFrame() if args.no_resume else load_existing(state_path)
    completed = set()
    if not state.empty:
        completed = set(
            zip(
                state.get("source", pd.Series(dtype=str)).astype(str),
                state.get("query_key", pd.Series(dtype=str)).astype(str),
                state.get("window_start", pd.Series(dtype=str)).astype(str),
                state.get("window_end", pd.Series(dtype=str)).astype(str),
            )
        )

    people = ["Musk", "Trump"] if args.person == "all" else [args.person]
    sources = ["google", "gdelt"] if args.source == "both" else [args.source]
    windows = make_windows(args.start, args.end, args.window_days)

    total_jobs = sum(len(SEARCH_QUERIES[p]) for p in people) * len(windows) * len(sources)
    print(f"[뉴스 수집] 기간 {args.start} ~ {args.end} | jobs={total_jobs:,} | resume={not args.no_resume}")
    collected_batches: list[pd.DataFrame] = [existing] if not existing.empty else []
    state_rows = state.to_dict("records") if not state.empty else []
    done = 0

    for source in sources:
        fetch = fetch_google if source == "google" else fetch_gdelt
        for person in people:
            for query_key, query in SEARCH_QUERIES[person]:
                for window in windows:
                    key = (source, query_key, str(window.start.date()), str(window.end.date()))
                    done += 1
                    if key in completed:
                        print(f"[{done}/{total_jobs}] skip {source} {query_key} {window.start.date()}~{window.end.date()}")
                        continue
                    print(f"[{done}/{total_jobs}] {source} | {person} | {query_key} | {window.start.date()}~{window.end.date()}")
                    status = "ok"
                    error = ""
                    try:
                        items = fetch(query, window, args.max_articles)
                    except Exception as exc:  # noqa: BLE001
                        items = []
                        status = "error"
                        error = str(exc)
                        print(f"  실패: {error}")

                    rows = []
                    for item in items:
                        published = pd.to_datetime(item.get("published_at"), errors="coerce", utc=True)
                        if pd.isna(published):
                            continue
                        published_day = published.tz_convert("UTC").tz_localize(None).normalize()
                        if published_day < window.start or published_day > window.end:
                            continue
                        title = str(item.get("title", "")).strip()
                        if not title:
                            continue
                        topic = assign_topic(f"{title} {query}", person)
                        market_relevant = topic != "other"
                        if not market_relevant and not args.include_other:
                            continue
                        ticker = map_ticker(person, topic) if market_relevant else ""
                        pub_date = published.strftime("%Y-%m-%d")
                        rows.append(
                            {
                                "article_id": article_id(person, pub_date, title, str(item.get("publisher", ""))),
                                "person": person,
                                "published_at": published.isoformat(),
                                "title": title,
                                "publisher": str(item.get("publisher", "")).strip(),
                                "url": str(item.get("url", "")).strip(),
                                "source_api": source,
                                "query_key": query_key,
                                "search_query": query,
                                "topic": topic,
                                "ticker": ticker,
                                "market_relevant": market_relevant,
                                "window_start": str(window.start.date()),
                                "window_end": str(window.end.date()),
                            }
                        )
                    if rows:
                        batch = pd.DataFrame(rows)
                        collected_batches.append(batch)
                        print(f"  저장 후보 {len(batch):,}건")
                    else:
                        print("  저장 후보 0건")

                    state_rows.append(
                        {
                            "source": source,
                            "query_key": query_key,
                            "window_start": str(window.start.date()),
                            "window_end": str(window.end.date()),
                            "status": status,
                            "n_items": len(items),
                            "n_saved": len(rows),
                            "error": error,
                            "finished_at": pd.Timestamp.utcnow().isoformat(),
                        }
                    )
                    pd.DataFrame(state_rows).to_csv(state_path, index=False)
                    merged = pd.concat(collected_batches, ignore_index=True, sort=False) if collected_batches else pd.DataFrame()
                    merged = save_deduped(merged, out_path)
                    collected_batches = [merged]
                    if status == "ok":
                        completed.add(key)
                    time.sleep(max(0.0, args.sleep))

    final = save_deduped(pd.concat(collected_batches, ignore_index=True, sort=False) if collected_batches else pd.DataFrame(), out_path)
    print("\n[완료]")
    print(f"원본 기사 메타데이터: {len(final):,}건")
    if not final.empty:
        print("인물별:", final["person"].value_counts().to_dict())
        print("ticker별:", final["ticker"].replace("", pd.NA).dropna().value_counts().to_dict())
        print("topic별 상위:", final["topic"].value_counts().head(12).to_dict())
    print(f"저장: {out_path}")
    print("다음: python build_track2_news_events.py")


if __name__ == "__main__":
    main()
