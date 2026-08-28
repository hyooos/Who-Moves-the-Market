"""수집된 기사들을 '뉴스 사건' 단위로 묶습니다.

입력:  data/interim/news_articles_raw.csv
출력:  data/interim/track2_news_auto_events.csv
        outputs/tables/news_event_articles.csv

감성분석은 하지 않습니다. 같은 보도를 여러 매체가 반복한 경우 사건 하나로 묶고,
원본 기사 메타데이터는 별도 매핑 테이블로 모두 보존합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from difflib import SequenceMatcher

import pandas as pd

from market_mover import config

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "as", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "has", "have", "had", "will", "would", "could", "should",
    "musk", "elon", "trump", "donald", "says", "say", "said", "report", "reports", "news", "update",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="뉴스 기사들을 사건 단위로 클러스터링합니다.")
    p.add_argument("--cluster-window-days", type=int, default=1, help="같은 사건으로 묶을 수 있는 최대 날짜 차이")
    p.add_argument("--similarity", type=float, default=0.38, help="제목 유사도 임계값(낮을수록 더 많이 합쳐짐)")
    p.add_argument("--min-sources", type=int, default=1, help="사건으로 남길 최소 매체 수. 많이 보존하려면 1")
    p.add_argument("--min-articles", type=int, default=1, help="사건으로 남길 최소 기사 수")
    return p.parse_args()


def norm(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+-\s+[^-]{1,80}$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def tokens(text: str) -> set[str]:
    return {tok for tok in norm(text).split() if len(tok) > 2 and tok not in STOPWORDS}


def similarity(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    ta, tb = tokens(na), tokens(nb)
    jaccard = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    # 제목 표현이 조금 달라도 핵심 단어가 겹치면 같은 사건일 가능성이 높습니다.
    return max(jaccard, seq * 0.78)


def stable_event_id(person: str, ticker: str, topic: str, date: str, title: str) -> str:
    raw = f"{person}|{ticker}|{topic}|{date}|{norm(title)}"
    return "news_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    args = parse_args()
    source_path = config.INTERIM_DIR / "news_articles_raw.csv"
    if not source_path.exists():
        raise FileNotFoundError("먼저 collect_track2_news.py를 실행하세요.")

    articles = pd.read_csv(source_path)
    if articles.empty:
        raise RuntimeError("수집된 뉴스가 없습니다.")
    articles["published_at"] = pd.to_datetime(articles["published_at"], errors="coerce", utc=True)
    articles = articles.dropna(subset=["published_at", "title", "person", "topic", "ticker"]).copy()
    articles = articles[articles["ticker"].astype(str).str.len().gt(0)]
    articles["article_date"] = articles["published_at"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    articles["title_norm"] = articles["title"].map(norm)
    articles = articles.drop_duplicates(subset=["person", "article_date", "title_norm", "publisher"], keep="first")
    articles = articles.sort_values(["person", "ticker", "topic", "published_at"]).reset_index(drop=True)

    cluster_rows: list[dict] = []
    membership_rows: list[dict] = []

    for (person, ticker, topic), group in articles.groupby(["person", "ticker", "topic"], sort=False):
        clusters: list[dict] = []
        for _, article in group.iterrows():
            article_date = article["article_date"]
            best_idx = None
            best_score = -1.0
            for idx, cluster in enumerate(clusters):
                date_gap = abs((article_date - cluster["last_date"]).days)
                if date_gap > args.cluster_window_days:
                    continue
                score = max(similarity(article["title"], title) for title in cluster["titles"][:8])
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None and best_score >= args.similarity:
                cluster = clusters[best_idx]
                cluster["articles"].append(article.to_dict())
                cluster["titles"].append(article["title"])
                cluster["last_date"] = max(cluster["last_date"], article_date)
            else:
                clusters.append(
                    {
                        "articles": [article.to_dict()],
                        "titles": [article["title"]],
                        "first_date": article_date,
                        "last_date": article_date,
                    }
                )

        for cluster in clusters:
            members = pd.DataFrame(cluster["articles"]).sort_values("published_at")
            publishers = [p for p in members["publisher"].fillna("").astype(str).tolist() if p]
            unique_publishers = list(dict.fromkeys(publishers))
            if len(members) < args.min_articles or len(unique_publishers) < args.min_sources:
                continue

            # 대표 제목은 가장 많은 핵심 토큰을 가진 제목으로 두어 지나치게 짧은 헤드라인을 피합니다.
            representative_idx = max(
                members.index,
                key=lambda idx: (len(tokens(members.loc[idx, "title"])), len(str(members.loc[idx, "title"]))),
            )
            rep = members.loc[representative_idx]
            event_date = members["published_at"].min().tz_convert("America/New_York")
            event_id = stable_event_id(person, ticker, topic, event_date.strftime("%Y-%m-%d"), rep["title"])

            cluster_rows.append(
                {
                    "event_id": event_id,
                    "person": person,
                    "posted_at": event_date.isoformat(),
                    "posted_at_timezone": "America/New_York",
                    "timestamp_precision": "exact",
                    "platform": "News",
                    "topic": topic,
                    "ticker": ticker,
                    "source_url": str(rep.get("url", "")),
                    "description": str(rep["title"]),
                    "track": "track2_news_auto",
                    "related_article_count": int(len(members)),
                    "related_source_count": int(len(unique_publishers)),
                    "news_sources": " | ".join(unique_publishers[:20]),
                    "first_article_at": members["published_at"].min().isoformat(),
                    "last_article_at": members["published_at"].max().isoformat(),
                    "collection_source": " | ".join(dict.fromkeys(members["source_api"].fillna("").astype(str))),
                }
            )
            for _, member in members.iterrows():
                membership_rows.append(
                    {
                        "event_id": event_id,
                        "article_id": member.get("article_id"),
                        "published_at": member.get("published_at"),
                        "title": member.get("title"),
                        "publisher": member.get("publisher"),
                        "url": member.get("url"),
                        "source_api": member.get("source_api"),
                    }
                )

    events = pd.DataFrame(cluster_rows)
    mapping = pd.DataFrame(membership_rows)
    if not events.empty:
        events = events.sort_values("posted_at").drop_duplicates(subset=["event_id"], keep="first").reset_index(drop=True)

    auto_path = config.INTERIM_DIR / "track2_news_auto_events.csv"
    map_path = config.TABLE_DIR / "news_event_articles.csv"
    auto_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(auto_path, index=False)
    mapping.to_csv(map_path, index=False)

    print(f"[뉴스 사건화] 기사 {len(articles):,}건 → 사건 {len(events):,}건")
    if not events.empty:
        print("인물별:", events["person"].value_counts().to_dict())
        print("ticker별:", events["ticker"].value_counts().to_dict())
        print("관련기사 중앙값:", float(events["related_article_count"].median()))
    print(f"사건 저장: {auto_path}")
    print(f"기사-사건 매핑: {map_path}")
    print("다음: python refresh_track2_news.py")


if __name__ == "__main__":
    main()
