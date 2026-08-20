from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

HOVER_LABELS = {
    "ticker": "종목",
    "abnormal_return": "초과수익률",
    "engagement": "참여도",
    "contamination_level": "오염 수준",
    "text_clean": "게시물 내용",
    "event_date": "이벤트 거래일",
    "topic": "토픽",
    "impact_score": "영향 점수",
    "person": "인물",
}


def plot_impact_ranking(events: pd.DataFrame, path: Path, top_n: int = 30) -> None:
    top = events.dropna(subset=["impact_score"]).nlargest(top_n, "impact_score").copy()
    if top.empty:
        return
    top["label"] = top["person"] + " | " + top["topic"] + " | " + top["event_date"].astype(str)
    fig = px.bar(
        top.sort_values("impact_score"),
        x="impact_score",
        y="label",
        color="person",
        orientation="h",
        hover_data=["ticker", "abnormal_return", "engagement", "contamination_level", "text_clean"],
        title="영향 점수 상위 이벤트",
        labels={"impact_score": "영향 점수", "label": "", "person": "인물"},
    )
    fig.update_traces(hovertemplate=None)
    fig.update_layout(height=max(500, top_n * 24), yaxis_title="", xaxis_title="영향 점수")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)


def plot_topic_heatmap(events: pd.DataFrame, path: Path) -> None:
    clean = events[events["contamination_level"] == "CLEAN"].copy()
    if clean.empty:
        return
    table = clean.pivot_table(
        index="topic",
        columns="person",
        values="impact_score",
        aggfunc="mean",
    ).fillna(0)
    fig = px.imshow(
        table,
        text_auto=".2f",
        aspect="auto",
        title="토픽/인물별 평균 영향 점수",
        color_continuous_scale="RdBu_r",
        labels={"x": "인물", "y": "토픽", "color": "평균 영향 점수"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)


def plot_daily_distribution(events: pd.DataFrame, path: Path) -> None:
    clean = events[events["contamination_level"] == "CLEAN"].copy()
    if clean.empty:
        return
    fig = px.box(
        clean,
        x="person",
        y="abnormal_return",
        color="person",
        points="all",
        hover_data=["event_date", "topic", "ticker", "impact_score", "text_clean"],
        title="오염이 적은 이벤트의 초과수익률 분포",
        labels={"person": "인물", "abnormal_return": "초과수익률"},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)


def plot_decay_curve(events: pd.DataFrame, scored_prices: pd.DataFrame, path: Path, window: int = 3) -> None:
    clean = events[events["contamination_level"] == "CLEAN"].copy()
    if clean.empty:
        return
    rows = []
    for _, event in clean.iterrows():
        ticker_prices = scored_prices[scored_prices["ticker"] == event["ticker"]].sort_values("date").reset_index(drop=True)
        positions = ticker_prices.index[ticker_prices["date"].eq(event["event_date"])]
        if len(positions) == 0:
            continue
        pos = int(positions[0])
        for offset in range(-window, window + 1):
            idx = pos + offset
            if 0 <= idx < len(ticker_prices):
                row = ticker_prices.iloc[idx]
                rows.append(
                    {
                        "offset": offset,
                        "abnormal_return_abs": abs(row["abnormal_return"]),
                        "person": event["person"],
                        "topic": event["topic"],
                        "ticker": event["ticker"],
                    }
                )
    curve = pd.DataFrame(rows)
    if curve.empty:
        return
    summary = curve.groupby(["offset", "person"], as_index=False)["abnormal_return_abs"].mean()
    fig = px.line(
        summary,
        x="offset",
        y="abnormal_return_abs",
        color="person",
        markers=True,
        title="이벤트 전후 일봉 반응 감쇠곡선",
        labels={"offset": "이벤트 거래일 기준 일수", "abnormal_return_abs": "평균 절대 초과수익률", "person": "인물"},
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)


def plot_intraday_case(window: pd.DataFrame, event: pd.Series, path: Path) -> None:
    if window.empty:
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=window["minutes_from_post"],
            y=window["return_from_first_post_price"],
            mode="lines",
            name=event["ticker"],
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
    fig.update_layout(
        title=f"장중 이벤트 윈도우: {event['person']} {event['topic']} {event['posted_at']}",
        xaxis_title="게시 시점 기준 경과 시간(분)",
        yaxis_title="게시 직후 기준 수익률",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path)
