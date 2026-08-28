import argparse
import json

import pandas as pd

from market_mover import config
from market_mover.case_narratives import build_case_narratives
from market_mover.contamination import classify_contamination, load_fomc_calendar
from market_mover.event_clustering import DEFAULT_CLUSTER_HOURS, cluster_daily_events
from market_mover.event_windows import add_daily_event_windows, build_daily_events, prepare_track2_events
from market_mover.impact import compute_price_features
from market_mover.load_posts import load_all_posts
from market_mover.load_prices import load_or_download_daily_prices
from market_mover.novelty import add_novelty_scores
from market_mover.placebo import run_placebo_tests
from market_mover.plots import plot_daily_distribution, plot_decay_curve, plot_impact_ranking, plot_topic_heatmap
from market_mover.preprocess import preprocess_posts, summarize_filtering
from market_mover.report import write_report
from market_mover.sentiment import DEFAULT_MODEL, add_sentiment_columns
from market_mover.sensitivity import run_rivn_sensitivity
from market_mover.stats import run_stats, save_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Market Mover 일봉 이벤트 스터디 파이프라인을 실행합니다.")
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="가격 데이터 다운로드 없이 게시물 CSV 정제까지만 실행합니다.",
    )
    parser.add_argument(
        "--build-narratives",
        action="store_true",
        help="Track 2 이벤트에 대해 Jina Reader와 선택한 LLM으로 근거 기반 내러티브를 생성합니다.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["gemini", "groq", "ollama", "none"],
        default="gemini",
        help="Track 2 내러티브 생성에 사용할 LLM provider입니다.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM 모델명을 직접 지정합니다. 비우면 provider별 기본값을 사용합니다.",
    )
    parser.add_argument(
        "--add-sentiment",
        action="store_true",
        help="FinBERT 대신 Twitter-RoBERTa로 SNS 게시물 감성분석 컬럼을 추가합니다.",
    )
    parser.add_argument(
        "--sentiment-model",
        default=DEFAULT_MODEL,
        help="감성분석에 사용할 Hugging Face 모델명입니다.",
    )
    parser.add_argument(
        "--add-novelty",
        action="store_true",
        help="최근 30일 내 같은 인물의 유사 게시물 대비 novelty score를 추가하고 검정에 사용합니다.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["plain", "exclude-events"],
        default="exclude-events",
        help="robust z-score baseline 계산 방식입니다. exclude-events는 실제 이벤트 날짜를 baseline에서 제외합니다.",
    )
    parser.add_argument(
        "--run-placebo",
        action="store_true",
        help="실제 이벤트 대신 무작위 거래일을 사용한 placebo 검정을 실행합니다.",
    )
    parser.add_argument(
        "--placebo-iterations",
        type=int,
        default=200,
        help="placebo 반복 횟수입니다.",
    )
    parser.add_argument(
        "--run-rivn-sensitivity",
        action="store_true",
        help="TSLA peer에서 RIVN 포함/제외 결과를 비교하는 민감도 분석을 실행합니다.",
    )
    parser.add_argument(
        "--freeze-date",
        default=config.FREEZE_DATE,
        help="리포트에 표시할 분석 기준일입니다.",
    )
    parser.add_argument(
        "--cluster-hours",
        type=float,
        default=DEFAULT_CLUSTER_HOURS,
        help="같은 화자/ticker/topic/반응 거래일의 연속 게시물을 묶는 고정 시간 창입니다(기본 6시간).",
    )
    parser.add_argument(
        "--no-cluster-posts",
        action="store_true",
        help="게시물 clustering을 끄고 기존처럼 게시물 1개를 이벤트 1개로 취급합니다(민감도 비교용).",
    )
    return parser.parse_args()


def load_track2_events():
    path = config.MANUAL_DIR / "track2_curated_events.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    required = ["event_id", "person", "posted_at", "platform", "topic", "ticker", "source_url", "description"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Track 2 CSV에 필요한 컬럼이 없습니다: {missing}")
    df["posted_at"] = pd.to_datetime(df["posted_at"], errors="coerce")
    return df


def main():
    args = parse_args()
    config.ensure_output_folders()

    raw_posts = load_all_posts()
    posts = preprocess_posts(raw_posts)
    if args.add_novelty:
        posts = add_novelty_scores(posts)
    filtering_summary = summarize_filtering(raw_posts, posts)
    posts.to_csv(config.INTERIM_DIR / "posts_cleaned.csv", index=False)
    filtering_summary.to_csv(config.OUTPUT_DIR / "tables" / "filtering_summary.csv", index=False)

    if args.skip_prices:
        print(json.dumps({"posts_cleaned": len(posts)}, ensure_ascii=False, indent=2))
        return

    prices = load_or_download_daily_prices()

    fomc_calendar = load_fomc_calendar(config.MANUAL_DIR / "fomc_calendar.csv")
    post_events = build_daily_events(posts, prices)
    post_events.to_csv(config.INTERIM_DIR / "events_posts_aligned.csv", index=False)
    cluster_hours = 0.0 if args.no_cluster_posts else args.cluster_hours
    events = cluster_daily_events(post_events, window_hours=cluster_hours)
    if args.add_sentiment:
        # 통계 단위는 게시물 한 건이 아니라 6시간 안의 연속 게시물을 묶은 사건이다.
        # 대표 게시물의 감성만 남지 않도록 clustering이 끝난 뒤 사건 전체 문장을 분석한다.
        sentiment_text_column = "cluster_text_clean" if "cluster_text_clean" in events.columns else "text_clean"
        events = add_sentiment_columns(
            events,
            model_name=args.sentiment_model,
            text_column=sentiment_text_column,
        )
    first_pass_prices = compute_price_features(prices)
    events = add_daily_event_windows(events, first_pass_prices)
    events = classify_contamination(events, prices, fomc_calendar)
    exclude_dates = None
    if args.baseline_mode == "exclude-events":
        exclude_dates = {
            ticker: group["event_date"].tolist()
            for ticker, group in events[events["contamination_level"].eq("CLEAN")].groupby("ticker")
        }
    scored_prices = compute_price_features(prices, exclude_dates_by_ticker=exclude_dates)
    scored_prices.to_csv(config.INTERIM_DIR / "daily_prices_scored.csv", index=False)
    events = add_daily_event_windows(events, scored_prices)
    events = classify_contamination(events, prices, fomc_calendar)
    events.to_csv(config.INTERIM_DIR / "events_daily.csv", index=False)

    track2 = load_track2_events()
    if not track2.empty:
        track2 = prepare_track2_events(track2, prices)
        track2 = add_daily_event_windows(track2, scored_prices)
        if args.build_narratives:
            narratives = build_case_narratives(
                track2,
                config.TABLE_DIR / "case_narratives.csv",
                provider=args.llm_provider,
                model_name=args.llm_model,
            )
            track2 = track2.merge(narratives, on="event_id", how="left")
        final_events = pd.concat([events, track2], ignore_index=True, sort=False)
    else:
        final_events = events

    final_events.to_csv(config.PROCESSED_DIR / "events_scored.csv", index=False)
    final_events.to_csv(config.TABLE_DIR / "events_scored.csv", index=False)

    stats_results = run_stats(events)
    save_stats(stats_results, config.TABLE_DIR / "stats_results.json")
    if args.run_placebo:
        placebo_summary = run_placebo_tests(
            events,
            prices,
            scored_prices,
            fomc_calendar,
            iterations=args.placebo_iterations,
            output_dir=config.TABLE_DIR,
        )
        stats_results["placebo"] = placebo_summary
        save_stats(stats_results, config.TABLE_DIR / "stats_results.json")
    if args.run_rivn_sensitivity:
        stats_results["rivn_sensitivity"] = run_rivn_sensitivity(events, prices, config.TABLE_DIR)
        save_stats(stats_results, config.TABLE_DIR / "stats_results.json")

    plot_impact_ranking(final_events, config.FIGURE_DIR / "impact_ranking.html")
    plot_topic_heatmap(events, config.FIGURE_DIR / "topic_heatmap.html")
    plot_daily_distribution(events, config.FIGURE_DIR / "daily_return_distribution.html")
    plot_decay_curve(events, scored_prices, config.FIGURE_DIR / "decay_curve.html")
    write_report(
        final_events,
        stats_results,
        config.FIGURE_DIR,
        config.REPORT_DIR / "market_mover_report.html",
        freeze_date=args.freeze_date,
    )

    print(
        json.dumps(
            {
                "원본_게시물_수": len(raw_posts),
                "시장관련_게시물_수": len(posts),
                "Track1_정렬게시물_수": len(post_events),
                "Track1_클러스터_이벤트_수": len(events),
                "cluster_window_hours": cluster_hours,
                "Track2_이벤트_수": len(track2),
                "내러티브_생성": bool(args.build_narratives),
                "LLM_provider": args.llm_provider if args.build_narratives else None,
                "감성분석_추가": bool(args.add_sentiment),
                "novelty_추가": bool(args.add_novelty),
                "baseline_mode": args.baseline_mode,
                "placebo_검정": bool(args.run_placebo),
                "RIVN_민감도": bool(args.run_rivn_sensitivity),
                "리포트_경로": str(config.REPORT_DIR / "market_mover_report.html"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
