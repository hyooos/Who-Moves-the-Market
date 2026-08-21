"""실시간 게시물 관심 필터(라이브 모니터).

새 게시물이 올라오면 이 프로젝트의 topic_rules.py로 분류해서 "우리 파이프라인
기준으로 market-relevant하고 어느 종목에 매핑되는지"를 즉시 알려준다.

**이건 가격을 예측하는 게 아니다.** §5-3에서 topic/person 내용이 반응 크기를
설명하지 못한다는 걸 우리 스스로 확인했기 때문에, "이 topic은 역사적으로
반응이 컸다"는 식의 알림은 만들지 않는다. 대신 다음 세 가지만 결정론적으로
보여준다.
  1. 이 게시물이 market-relevant로 분류되는가(topic_rules.py 그대로 적용)
  2. 어느 종목(TSLA/QQQ/SPY)에 매핑되는가
  3. 같은 인물이 오늘 이미 다른 관련 게시물을 올렸는가(다중게시 — 실제 반응이
     나와도 어느 게시물 때문인지 특정하기 어려워짐을 미리 경고)

데이터 소스:
  - Trump: https://trumpstruth.org/feed (Truth Social 비공식 아카이브 RSS, 무료,
    키 불필요. 실제로 살아있는지 확인함 — 이 프로젝트 개발 시점 기준 정상 동작).
  - Musk: 무료로 안정적인 실시간 X RSS가 없음(X가 무료 API를 막음). 이 스크립트는
    Musk용 소스를 비워두고, `--paste` 옵션으로 텍스트를 직접 붙여넣어 같은 분류
    로직을 테스트할 수 있게 해둔다.

사용 예:
    PYTHONPATH=. .venv/bin/python live_monitor.py --person Trump
    PYTHONPATH=. .venv/bin/python live_monitor.py --person Musk --paste "Tesla FSD is amazing"

cron으로 주기 실행하도록 설계했다(상시 실행 프로세스가 아니라, 실행할 때마다
마지막으로 본 게시물 이후의 새 글만 확인).
"""

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

from market_mover import config
from market_mover.load_posts import clean_text
from market_mover.topic_rules import assign_topic, is_market_relevant, map_ticker

TRUMP_RSS = "https://trumpstruth.org/feed"
STATE_PATH = config.MANUAL_DIR / "live_monitor_state.json"
TODAY_LOG_PATH = config.MANUAL_DIR / "live_monitor_today_log.json"


def parse_args():
    parser = argparse.ArgumentParser(description="새 게시물을 실시간으로 분류해 관심 대상인지 알려줍니다(가격 예측 아님).")
    parser.add_argument("--person", choices=["Trump", "Musk"], required=True)
    parser.add_argument("--paste", default=None, help="RSS 대신 텍스트를 직접 붙여넣어 분류 로직만 테스트")
    parser.add_argument("--max-check", type=int, default=20, help="RSS에서 최근 몇 건까지 확인할지")
    return parser.parse_args()


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_today_log() -> dict:
    if TODAY_LOG_PATH.exists():
        return json.loads(TODAY_LOG_PATH.read_text(encoding="utf-8"))
    return {}


def _save_today_log(log: dict) -> None:
    TODAY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TODAY_LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_trump_feed(limit: int) -> list:
    from email.utils import parsedate_to_datetime

    response = requests.get(TRUMP_RSS, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    root = ET.fromstring(response.content)
    items = []
    for item in root.findall(".//item")[:limit]:
        guid = item.findtext("guid") or item.findtext("link") or ""
        link = item.findtext("link") or ""
        pub_date_raw = item.findtext("pubDate")
        try:
            pub_date = parsedate_to_datetime(pub_date_raw) if pub_date_raw else None
        except (TypeError, ValueError):
            pub_date = None
        items.append({"id": guid, "link": link, "pub_date": pub_date})
    return items


def fetch_post_text(link: str) -> str:
    """개별 게시물 페이지를 Jina Reader로 가져와 본문만 추출한다."""
    import re

    response = requests.get(f"https://r.jina.ai/{link}", timeout=20)
    response.raise_for_status()
    match = re.search(r'Title: Donald J\. Trump: "(.*)"', response.text)
    if match:
        return match.group(1)
    return ""


def classify_and_report(person: str, raw_text: str, today_log: dict) -> dict:
    text = clean_text(raw_text)
    relevant = is_market_relevant(text, person)
    topic = assign_topic(text, person)
    ticker = map_ticker(person, topic) if relevant else None

    today_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    person_ticker_key = f"{person}_{ticker}"
    posted_today = today_log.get(today_key, {}).get(person_ticker_key, 0) if relevant else 0

    result = {
        "person": person,
        "text_preview": text[:120],
        "market_relevant": relevant,
        "topic": topic,
        "ticker": ticker,
        "same_ticker_posts_today": posted_today,
    }

    if relevant:
        today_log.setdefault(today_key, {}).setdefault(person_ticker_key, 0)
        today_log[today_key][person_ticker_key] += 1

    return result


def print_result(result: dict) -> None:
    if not result["market_relevant"]:
        print(f"  [무관] {result['text_preview']}")
        return
    warn = ""
    if result["same_ticker_posts_today"] >= 1:
        warn = f" ⚠️ 오늘 {result['person']}이(가) {result['ticker']} 관련 글을 이미 {result['same_ticker_posts_today']}건 더 올림 — 다중게시로 반응 원인 특정이 어려울 수 있음(우리 contamination.py 기준)"
    print(
        f"  [관심 대상] topic={result['topic']} → ticker={result['ticker']}{warn}\n"
        f"    원문 일부: {result['text_preview']}\n"
        f"    (주의: 이건 topic→ticker 규칙 매칭 결과일 뿐, 가격 반응을 예측하는 것이 아님 — §5-3 참고)"
    )


def main():
    args = parse_args()
    config.ensure_output_folders()
    today_log = _load_today_log()

    if args.paste:
        result = classify_and_report(args.person, args.paste, today_log)
        print_result(result)
        _save_today_log(today_log)
        return

    if args.person == "Trump":
        state = _load_state()
        last_seen = state.get("trump_last_id")
        items = fetch_trump_feed(args.max_check)
        new_items = []
        for item in items:
            if item["id"] == last_seen:
                break
            new_items.append(item)
        if not new_items:
            print("새 게시물 없음.")
            return
        print(f"새 게시물 {len(new_items)}건 확인:\n")
        for item in reversed(new_items):
            text = fetch_post_text(item["link"])
            if not text:
                print(f"  (본문 없음 — 리트윗/이미지 전용 게시물로 보임: {item['link']})")
                continue
            result = classify_and_report("Trump", text, today_log)
            print_result(result)
            print()
        state["trump_last_id"] = items[0]["id"]
        _save_state(state)
        _save_today_log(today_log)
    else:
        print(
            "Musk용 무료 실시간 RSS 소스가 아직 없습니다(X가 무료 API를 막아둠). "
            "twscrape 등 비공식 도구가 필요하거나(backfill_track2_musk_twscrape.py 참고), "
            "--paste로 텍스트를 직접 넣어 분류 로직만 테스트할 수 있습니다."
        )


if __name__ == "__main__":
    main()
