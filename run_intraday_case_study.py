import argparse

import pandas as pd

from market_mover import config
from market_mover.event_windows import build_intraday_window
from market_mover.load_prices import load_intraday_prices
from market_mover.plots import plot_intraday_case


def parse_args():
    parser = argparse.ArgumentParser(description="선택한 이벤트 하나에 대해 장중 가격 움직임 차트를 생성합니다.")
    parser.add_argument("--event-id", required=True)
    parser.add_argument(
        "--intraday-csv",
        required=True,
        help="datetime,ticker,open,high,low,close,volume 컬럼을 가진 장중 가격 CSV입니다.",
    )
    parser.add_argument("--minutes-before", type=int, default=60)
    parser.add_argument("--minutes-after", type=int, default=240)
    return parser.parse_args()


def main():
    args = parse_args()
    events_path = config.PROCESSED_DIR / "events_scored.csv"
    if not events_path.exists():
        raise FileNotFoundError("장중 케이스 차트를 만들기 전에 run_daily_pipeline.py를 먼저 실행하세요.")

    events = pd.read_csv(events_path, parse_dates=["posted_at", "event_date"])
    matches = events[events["event_id"] == args.event_id]
    if matches.empty:
        raise ValueError(f"이벤트를 찾을 수 없습니다: {args.event_id}")
    event = matches.iloc[0]

    intraday = load_intraday_prices(config.ROOT / args.intraday_csv)
    window = build_intraday_window(
        event,
        intraday,
        minutes_before=args.minutes_before,
        minutes_after=args.minutes_after,
    )
    out_csv = config.TABLE_DIR / f"intraday_{args.event_id}.csv"
    out_fig = config.FIGURE_DIR / f"intraday_{args.event_id}.html"
    window.to_csv(out_csv, index=False)
    plot_intraday_case(window, event, out_fig)
    print({"행_수": len(window), "CSV_경로": str(out_csv), "차트_경로": str(out_fig)})


if __name__ == "__main__":
    main()
