from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from market_mover.dashboard_data import (
    h2_pairwise_frame,
    load_dashboard_data,
    significant_test_count,
    stats_tests_frame,
)


st.set_page_config(page_title="Market Mover Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def cached_data() -> dict:
    return load_dashboard_data()


def _as_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _filter_events(events: pd.DataFrame) -> pd.DataFrame:
    filtered = events.copy()
    with st.sidebar:
        st.header("필터")
        for column, label in [
            ("person", "인물"),
            ("topic", "토픽"),
            ("ticker", "종목"),
            ("contamination_level", "오염 수준"),
            ("sentiment_label", "감성"),
            ("track", "트랙"),
        ]:
            if column not in filtered.columns:
                continue
            options = sorted(str(value) for value in filtered[column].dropna().unique())
            selected = st.multiselect(label, options, default=options)
            if selected:
                filtered = filtered[filtered[column].astype(str).isin(selected)]
    return filtered


def _metric_value(value, fallback="0"):
    if value is None:
        return fallback
    return value


def render_overview(events: pd.DataFrame, stats: dict, placebo_summary: dict) -> None:
    clean = events[events.get("contamination_level", pd.Series(dtype=str)).eq("CLEAN")] if not events.empty else events
    cols = st.columns(4)
    cols[0].metric("전체 이벤트", f"{len(events):,}")
    cols[1].metric("CLEAN 이벤트", f"{len(clean):,}")
    cols[2].metric("토픽 수", f"{events['topic'].nunique():,}" if "topic" in events else "0")
    cols[3].metric("FDR 유의 검정", f"{significant_test_count(stats):,}")

    if events.empty:
        st.info("아직 `events_scored.csv`가 없습니다. 먼저 일봉 파이프라인을 실행하면 화면이 채워집니다.")
        return

    timeline = events.dropna(subset=["event_date", "impact_score"]).copy()
    if not timeline.empty:
        timeline["event_date"] = _as_datetime(timeline["event_date"])
        timeline["abs_abnormal_return"] = timeline.get("abnormal_return", 0).abs()
        fig = px.scatter(
            timeline,
            x="event_date",
            y="impact_score",
            color="person" if "person" in timeline else None,
            size="abs_abnormal_return",
            hover_data=[
                col
                for col in ["event_id", "topic", "ticker", "contamination_level", "abnormal_return", "text_clean"]
                if col in timeline.columns
            ],
            labels={
                "event_date": "이벤트 거래일",
                "impact_score": "영향 점수",
                "person": "인물",
                "abs_abnormal_return": "절대 초과수익률",
            },
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    if placebo_summary:
        st.subheader("Placebo 요약")
        st.json(placebo_summary, expanded=False)

    top_cols = [
        col
        for col in [
            "event_id",
            "person",
            "event_date",
            "topic",
            "ticker",
            "contamination_level",
            "impact_score",
            "abnormal_return",
            "sentiment_label",
            "novelty_score",
            "text_clean",
        ]
        if col in events.columns
    ]
    st.subheader("영향 점수 상위 이벤트")
    st.dataframe(events.sort_values("impact_score", ascending=False)[top_cols].head(25), use_container_width=True)


def render_event_explorer(events: pd.DataFrame) -> None:
    if events.empty:
        st.info("이벤트 테이블이 생성되면 개별 게시물 탐색기가 활성화됩니다.")
        return

    filtered = _filter_events(events)
    left, right = st.columns([1.1, 1])
    table_cols = [
        col
        for col in [
            "event_id",
            "person",
            "event_date",
            "topic",
            "ticker",
            "contamination_level",
            "impact_score",
            "abnormal_return",
            "sentiment_label",
            "novelty_score",
        ]
        if col in filtered.columns
    ]
    with left:
        st.dataframe(
            filtered.sort_values("impact_score", ascending=False)[table_cols],
            use_container_width=True,
            height=540,
        )
    with right:
        event_ids = filtered["event_id"].astype(str).tolist() if "event_id" in filtered else []
        selected_id = st.selectbox("선택 이벤트", event_ids) if event_ids else None
        if selected_id:
            event = filtered[filtered["event_id"].astype(str).eq(selected_id)].iloc[0]
            st.metric("영향 점수", _metric_value(event.get("impact_score")))
            st.metric("초과수익률", _metric_value(event.get("abnormal_return")))
            st.write(event.get("text_clean", event.get("description", "")))
            detail_cols = [
                col
                for col in [
                    "posted_at",
                    "event_date",
                    "person",
                    "topic",
                    "ticker",
                    "platform",
                    "engagement",
                    "contamination_level",
                    "sentiment_label",
                    "sentiment_score",
                    "sentiment_confidence",
                    "novelty_score",
                    "max_prior_similarity",
                    "source_url",
                ]
                if col in event.index
            ]
            st.dataframe(event[detail_cols].to_frame("값"), use_container_width=True)
            narrative = event.get("narrative")
            if isinstance(narrative, str) and narrative.strip():
                st.subheader("Track 2 내러티브")
                st.write(narrative)


def render_hypotheses(stats: dict) -> None:
    tests = stats_tests_frame(stats)
    if tests.empty:
        st.info("통계검정 결과가 아직 없습니다.")
        return
    st.dataframe(tests, use_container_width=True)
    pairwise = h2_pairwise_frame(stats)
    if not pairwise.empty:
        st.subheader("H2 토픽 사후검정")
        st.dataframe(pairwise, use_container_width=True)
    effect_sizes = stats.get("effect_sizes", {})
    if effect_sizes:
        st.subheader("효과크기")
        st.json(effect_sizes, expanded=False)


def render_method_checks(data: dict, events: pd.DataFrame) -> None:
    if not events.empty and "contamination_level" in events:
        counts = events["contamination_level"].fillna("UNKNOWN").value_counts().reset_index()
        counts.columns = ["오염 수준", "이벤트 수"]
        fig = px.bar(counts, x="오염 수준", y="이벤트 수", text="이벤트 수")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Topic audit")
        summary = data["topic_audit_summary"]
        confusion = data["topic_audit_confusion"]
        st.dataframe(summary if not summary.empty else pd.DataFrame({"상태": ["아직 topic audit 결과가 없습니다."]}))
        if not confusion.empty:
            st.dataframe(confusion, use_container_width=True)
    with cols[1]:
        st.subheader("RIVN 민감도")
        rivn = data["rivn_sensitivity"]
        st.dataframe(rivn if not rivn.empty else pd.DataFrame({"상태": ["아직 RIVN 민감도 결과가 없습니다."]}))

    placebo = data["placebo_results"]
    if not placebo.empty:
        st.subheader("Placebo 반복 결과")
        st.dataframe(placebo, use_container_width=True)


def render_case_studies(events: pd.DataFrame, figures: list[Path]) -> None:
    track2 = events[events.get("track", pd.Series(dtype=str)).astype(str).str.contains("track2", na=False)] if not events.empty else events
    if not track2.empty:
        cols = [
            col
            for col in ["event_id", "person", "posted_at", "topic", "ticker", "description", "narrative", "source_url"]
            if col in track2.columns
        ]
        st.dataframe(track2[cols], use_container_width=True, height=420)
    else:
        st.info("Track 2 수동 이벤트 또는 내러티브가 생성되면 이 탭에 표시됩니다.")

    intraday = [path for path in figures if "intraday" in path.name.lower()]
    if intraday:
        st.subheader("장중 케이스 HTML")
        for path in intraday:
            st.markdown(f"- [{path.name}]({path.as_posix()})")


def render_ask_data_design(stats: dict, events: pd.DataFrame) -> None:
    question = st.text_input("질문")
    if not question:
        st.info("LLM 연결 전에는 산출물 기반 요약만 표시합니다.")
        return
    clean_n = int(stats.get("n_clean", 0))
    total_n = int(stats.get("n_total", len(events)))
    st.write(
        {
            "질문": question,
            "현재_답변_범위": "events_scored.csv와 stats_results.json에 있는 값만 근거로 답변해야 합니다.",
            "전체_이벤트": total_n,
            "CLEAN_이벤트": clean_n,
            "FDR_유의_검정_수": significant_test_count(stats),
        }
    )


def main() -> None:
    st.title("Market Mover Dashboard")
    data = cached_data()
    events = data["events"]
    stats = data["stats"]

    tabs = st.tabs(["개요", "이벤트 탐색기", "가설 검증", "방법론 점검", "케이스 스터디", "Ask the Data"])
    with tabs[0]:
        render_overview(events, stats, data["placebo_summary"])
    with tabs[1]:
        render_event_explorer(events)
    with tabs[2]:
        render_hypotheses(stats)
    with tabs[3]:
        render_method_checks(data, events)
    with tabs[4]:
        render_case_studies(events, data["figures"])
    with tabs[5]:
        render_ask_data_design(stats, events)


if __name__ == "__main__":
    main()

