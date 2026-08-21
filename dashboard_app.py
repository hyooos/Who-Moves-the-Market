import html
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from market_mover.dashboard_data import (
    h2_pairwise_frame,
    load_dashboard_data,
    significant_test_count,
    stats_tests_frame,
)
from market_mover.dashboard_widgets import compute_ticker_gauges, render_single_gauge_html, render_mac_window_html
from market_mover.case_narratives import generate_event_commentary
from market_mover.load_posts import clean_text
from market_mover.topic_rules import assign_topic, is_market_relevant, map_ticker
from live_monitor import fetch_trump_feed, fetch_post_text


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


def _has_real_content(df: pd.DataFrame) -> pd.Series:
    """text_clean/description/source_url 중 하나라도 실제 내용이 있는 행만 True.
    차트에 원문·기사가 없는 이벤트까지 마커로 뿌려서 헛클릭을 유발하지 않기 위함."""
    if df.empty:
        return pd.Series([], dtype=bool)
    has_any = pd.Series(False, index=df.index)
    for col in ("text_clean", "description", "source_url"):
        if col in df.columns:
            has_any = has_any | df[col].fillna("").astype(str).str.strip().ne("")
    return has_any


def render_live_trump_feed() -> None:
    with st.expander("🔴 진짜 실시간 — Trump 새 게시물 감지(반응 크기 아님)", expanded=False):
        st.caption(
            "Truth Social 미러 RSS(trumpstruth.org)를 지금 이 순간 실제로 조회합니다. "
            "반응 크기·백분위는 표시하지 않습니다 — 아직 시장이 반응할 시간이 없었기 때문입니다(위 설명 참고). "
            "topic→종목 분류 결과만 즉시 보여주는 `live_monitor.py`와 동일한 로직입니다. "
            "Musk는 무료 실시간 소스가 없어 지원하지 않습니다. "
            "기본은 최근 24시간 내 게시물만 보여주고, 없으면 3일→5일→7일 순으로 범위를 넓혀서 찾습니다."
        )
        if st.button("🔄 지금 확인하기"):
            with st.spinner("Truth Social RSS를 확인하는 중..."):
                try:
                    items = fetch_trump_feed(30)
                except Exception as exc:
                    st.error(f"RSS 조회 실패: {exc}")
                    return

                now = datetime.now(timezone.utc)
                window_used = None
                selected_items = []
                for days in (1, 3, 5, 7):
                    cutoff = now - timedelta(days=days)
                    candidates = [
                        it for it in items
                        if it.get("pub_date") is not None and it["pub_date"] >= cutoff
                    ]
                    if candidates:
                        window_used = days
                        selected_items = candidates[:8]
                        break

                if not selected_items:
                    st.warning("최근 7일 이내 게시물을 RSS에서 찾지 못했습니다(피드 자체가 갱신되지 않았을 수 있음).")
                    return

                window_label = "24시간" if window_used == 1 else f"{window_used}일"
                st.caption(f"📅 최근 {window_label} 이내 게시물 {len(selected_items)}건 (범위 자동 확장 결과)")

                rows = []
                for item in selected_items:
                    try:
                        text = fetch_post_text(item["link"])
                    except Exception:
                        text = ""
                    posted = item["pub_date"].strftime("%Y-%m-%d %H:%M") if item.get("pub_date") else "-"
                    if not text:
                        rows.append(
                            {"작성자": "Trump", "게시": posted, "관련여부": "-", "topic": "-", "종목": "-",
                             "미리보기": "(본문 없음 — 리트윗/이미지 전용 추정)", "원문 링크": item.get("link")}
                        )
                        continue
                    text_c = clean_text(text)
                    relevant = is_market_relevant(text_c, "Trump")
                    topic = assign_topic(text_c, "Trump")
                    ticker = map_ticker("Trump", topic) if relevant else None
                    rows.append(
                        {
                            "작성자": "Trump",
                            "게시": posted,
                            "관련여부": "✅ 시장관련" if relevant else "-",
                            "topic": topic,
                            "종목": ticker or "-",
                            "미리보기": text_c[:100],
                            "원문 링크": item.get("link"),
                        }
                    )
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        column_config={
                            "원문 링크": st.column_config.LinkColumn("원문", display_text="열기 ↗"),
                        },
                    )
                    st.caption(f"조회 시각: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
                else:
                    st.info("피드에서 항목을 가져오지 못했습니다.")


def render_live_main(events: pd.DataFrame, daily_prices: pd.DataFrame) -> None:
    st.subheader("📊 최근 게시물 반응강도 게이지 — 종목별 비교")
    st.caption(
        "이 숫자는 실시간 계산이 아니라 **회고적(retrospective)** 지표입니다. "
        "각 게이지는 데이터셋에서 그 종목에 매핑된 게시물 중 가장 최근 것을 보여주므로 "
        "종목마다 게시 시점이 다른 게 정상입니다(예: QQQ 관련 최근 게시물과 TSLA 관련 최근 게시물은 날짜가 다름). "
        "백분위(impact_score)는 그 게시물 **다음 거래일의 실제 가격 반응**으로 계산되기 때문에, "
        "방금 올라온 글에 대해서는 이 숫자 자체를 계산할 수 없습니다 — 시장이 아직 반응할 시간이 없었으니까요. "
        "그래서 이 프로젝트는 반응 크기를 예측하지 않습니다(§8-1). 새 글이 올라왔을 때 topic→종목 분류만 "
        "즉시 보여주는 진짜 실시간 도구는 `live_monitor.py`가 따로 담당합니다."
    )
    render_live_trump_feed()
    gauge_states = compute_ticker_gauges(events, tickers=("QQQ", "SPY", "TSLA"))
    gauge_cols = st.columns(3)
    for col, gstate in zip(gauge_cols, gauge_states):
        with col:
            st.markdown(render_single_gauge_html(gstate), unsafe_allow_html=True)

    st.divider()
    st.subheader("종목 가격 추이 — 차트를 클릭하면 그 시점 게시물 원문을 볼 수 있습니다")

    if events.empty or "ticker" not in events.columns or daily_prices.empty:
        st.info("아직 표시할 가격/이벤트 데이터가 없습니다.")
        return

    tickers = sorted(events["ticker"].dropna().unique().tolist())
    if not tickers:
        st.info("종목 정보가 없습니다.")
        return
    latest_ticker = next((g["ticker"] for g in gauge_states if g.get("has_data")), tickers[0])
    default_idx = tickers.index(latest_ticker) if latest_ticker in tickers else 0
    ticker_options = tickers + ["전체 비교(정규화)"]
    ticker = st.selectbox("종목 선택", ticker_options, index=default_idx)

    persons = sorted(events["person"].dropna().unique().tolist()) if "person" in events.columns else []
    person_choice = st.radio("인물", ["전체"] + persons, horizontal=True)

    events_f = events if person_choice == "전체" else events[events["person"] == person_choice]

    fig = go.Figure()

    if ticker == "전체 비교(정규화)":
        # 종목마다 가격 단위(TSLA 수백 달러 vs QQQ/SPY)가 달라 그대로 겹치면 비교가
        # 안 되므로, 각 종목의 표시 구간 첫날을 100으로 맞춘 정규화 지수로 겹쳐 그린다.
        line_colors = {"QQQ": "#2563eb", "SPY": "#16a34a", "TSLA": "#ef4444"}
        for tkr in tickers:
            p = daily_prices[daily_prices["ticker"] == tkr].copy()
            p["date"] = _as_datetime(p["date"])
            p = p.dropna(subset=["date", "close"]).sort_values("date")
            if p.empty:
                continue
            base_price = p["close"].iloc[0]
            p["indexed"] = p["close"] / base_price * 100
            fig.add_trace(
                go.Scatter(
                    x=p["date"], y=p["indexed"], mode="lines",
                    line=dict(color=line_colors.get(tkr, "#64748b"), width=2),
                    name=tkr, hovertemplate=f"{tkr} %{{y:.1f}}<extra></extra>",
                )
            )
            tkr_events = events_f[events_f["ticker"] == tkr].dropna(subset=["event_date"]).copy()
            tkr_events = tkr_events[_has_real_content(tkr_events)]
            tkr_events["event_date"] = _as_datetime(tkr_events["event_date"])
            tkr_merged = tkr_events.merge(p[["date", "indexed"]], left_on="event_date", right_on="date", how="left")
            tkr_merged = tkr_merged.dropna(subset=["indexed"])
            if not tkr_merged.empty:
                fig.add_trace(
                    go.Scatter(
                        x=tkr_merged["event_date"], y=tkr_merged["indexed"], mode="markers",
                        marker=dict(size=6, color="rgba(30,41,59,0.02)", line=dict(width=0)),
                        customdata=tkr_merged[["event_id"]].values if "event_id" in tkr_merged.columns else None,
                        text=tkr_merged.get("person"), hovertemplate="%{text}<br>%{x|%Y-%m-%d}<extra></extra>",
                        showlegend=False,
                    )
                )
        fig.update_layout(yaxis_title="첫날=100 지수")
        fig.update_xaxes(
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="#94a3b8", spikethickness=1, spikedash="solid",
        )
        click = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun", selection_mode="points", key="live_main_chart"
        )
        _render_selected_event(click, events)
        return

    prices = daily_prices[daily_prices["ticker"] == ticker].copy()
    prices["date"] = _as_datetime(prices["date"])
    prices = prices.dropna(subset=["date", "close"]).sort_values("date")
    if prices.empty:
        st.info(f"{ticker} 가격 데이터가 없습니다.")
        return

    y_min = prices["close"].min()
    baseline = y_min - (prices["close"].max() - y_min) * 0.05

    fig.add_trace(
        go.Scatter(
            x=prices["date"],
            y=[baseline] * len(prices),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prices["date"],
            y=prices["close"],
            mode="lines",
            line=dict(color="#2563eb", width=2.5),
            fill="tonexty",
            fillgradient=dict(
                type="vertical",
                colorscale=[[0, "rgba(37,99,235,0.0)"], [1, "rgba(37,99,235,0.35)"]],
            ),
            name=f"{ticker} 종가",
            hovertemplate="%{x|%Y-%m-%d}<br>종가 %{y:.2f}<extra></extra>",
        )
    )

    ticker_events = events_f[events_f["ticker"] == ticker].dropna(subset=["event_date"]).copy()
    ticker_events = ticker_events[_has_real_content(ticker_events)]
    ticker_events["event_date"] = _as_datetime(ticker_events["event_date"])
    merged = ticker_events.merge(
        prices[["date", "close"]], left_on="event_date", right_on="date", how="left"
    )
    merged = merged.dropna(subset=["close"])
    if not merged.empty:
        fig.add_trace(
            go.Scatter(
                x=merged["event_date"],
                y=merged["close"],
                mode="markers",
                marker=dict(size=6, color="rgba(30,41,59,0.02)", line=dict(width=0)),
                customdata=merged[["event_id"]].values if "event_id" in merged.columns else None,
                text=merged.get("person"),
                hovertemplate="%{text}<br>%{x|%Y-%m-%d}<extra></extra>",
                name="게시물 이벤트",
            )
        )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="종가($)",
        plot_bgcolor="white",
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_yaxes(
        range=[baseline, prices["close"].max() * 1.03],
        showspikes=False,
    )
    fig.update_xaxes(
        range=[prices["date"].min(), prices["date"].max()],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#94a3b8",
        spikethickness=1,
        spikedash="solid",
    )

    click = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="live_main_chart",
    )
    _render_selected_event(click, events)


def _render_selected_event(click, events: pd.DataFrame) -> None:
    selected_event_id = None
    points = (click or {}).get("selection", {}).get("points", [])
    for point in points:
        customdata = point.get("customdata")
        if customdata:
            selected_event_id = customdata[0]
            break

    if selected_event_id is None or "event_id" not in events.columns:
        st.caption("차트를 클릭하면 그 시점 게시물 원문이 여기에 표시됩니다(점이 안 보여도 클릭은 됩니다).")
        return

    row = events[events["event_id"].astype(str) == str(selected_event_id)]
    if row.empty:
        st.caption("차트를 클릭하면 그 시점 게시물 원문이 여기에 표시됩니다(점이 안 보여도 클릭은 됩니다).")
        return

    row = row.iloc[0]
    is_track2 = str(row.get("track")) == "track2_manual"
    person = html.escape(str(row.get("person") or "-"))
    topic = html.escape(str(row.get("topic") or "-"))
    ticker = html.escape(str(row.get("ticker") or "-"))
    event_date = html.escape(str(row.get("event_date") or "-"))
    title = f"{row.get('person', '-')} · {row.get('topic', '-')} → {row.get('ticker', '-')} · {row.get('event_date')}"
    source_url = row.get("source_url")
    has_source = isinstance(source_url, str) and source_url.strip()

    # 누가/어떤 분야/어느 종목인지 표에서 바로 안 보여서 헷갈린다는 피드백 반영 —
    # 카드 맨 위에 라벨 붙여서 명확하게 정리해둔다.
    meta = (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px;">'
        f'<span style="background:#eef2ff;color:#3730a3;font-size:12px;font-weight:600;'
        f'padding:3px 9px;border-radius:999px;">👤 {person}</span>'
        f'<span style="background:#ecfdf5;color:#065f46;font-size:12px;font-weight:600;'
        f'padding:3px 9px;border-radius:999px;">🏷 {topic}</span>'
        f'<span style="background:#fef2f2;color:#991b1b;font-size:12px;font-weight:600;'
        f'padding:3px 9px;border-radius:999px;">📈 {ticker}</span>'
        f'<span style="background:#f1f5f9;color:#334155;font-size:12px;font-weight:600;'
        f'padding:3px 9px;border-radius:999px;">🗓 {event_date}</span>'
        "</div>"
    )

    if is_track2:
        # Musk 원본 게시물 데이터는 2025-04-13까지만 있어(캐글 수집 범위), 그 이후
        # 사건은 게시물 원문이 아니라 뉴스 보도를 근거로 수동 등록했다. 원문인 것처럼
        # 오인되지 않도록 라벨을 분리하고 실제 기사 링크를 맨 위로 올린다.
        body = meta + (
            '<div style="display:inline-block;background:#fef3c7;color:#92400e;'
            'font-size:11.5px;font-weight:600;padding:2px 8px;border-radius:6px;margin-bottom:8px;">'
            "⚠ 게시물 원문 아님 — 뉴스 보도 기반 요약(캐글 수집 범위 밖 사건)</div>"
        )
        if has_source:
            body += (
                f'<div style="margin-bottom:10px;"><a href="{html.escape(source_url)}" target="_blank" '
                f'style="color:#2563eb;font-weight:600;">📰 실제 기사 원문 보기 ↗</a></div>'
            )
        body += f"<div>{html.escape(str(row.get('description') or '(요약 없음)'))}</div>"
    else:
        body = meta + f"<div>{html.escape(str(row.get('text_clean') or row.get('description') or '(원문 없음)'))}</div>"
        if has_source:
            body += (
                f'<div style="margin-top:10px;"><a href="{html.escape(source_url)}" target="_blank" '
                f'style="color:#2563eb;">원문/기사 링크 열기 ↗</a></div>'
            )
        else:
            # Kaggle 원본 CSV에는 게시물의 실제 URL이 없다(post_id는 합성 인덱스일 뿐) —
            # 없는 링크를 만들어내지 않고 정직하게 이유를 밝힌다.
            body += (
                '<div style="margin-top:10px;font-size:12px;color:#94a3b8;">'
                "🔗 원문 링크 없음 — Kaggle 원본 데이터셋에 게시물 URL이 포함돼 있지 않습니다.</div>"
            )
    manual_narrative = row.get("narrative_reviewed") or row.get("narrative")
    if isinstance(manual_narrative, str) and manual_narrative.strip():
        body += (
            f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid #e2e8f0;'
            f'font-size:13px;color:#475569;"><b>Track2 내러티브(LLM):</b> '
            f"{html.escape(manual_narrative)}</div>"
        )
    st.markdown(render_mac_window_html(title, body), unsafe_allow_html=True)

    row_ticker = row.get("ticker")
    clean = events[events.get("contamination_level", pd.Series(dtype=str)).eq("CLEAN")]
    same_ticker = clean[clean["ticker"] == row_ticker] if "ticker" in clean.columns else clean.iloc[0:0]
    n = len(same_ticker)
    mean_abs_ar = same_ticker.get("abnormal_return", pd.Series(dtype=float)).abs().mean()
    impact_score = row.get("impact_score")
    same_ticker_scores = same_ticker.get("impact_score", pd.Series(dtype=float)).dropna()
    pct = (
        float((same_ticker_scores < impact_score).mean())
        if len(same_ticker_scores) and pd.notna(impact_score)
        else None
    )
    row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}

    with st.spinner("이 게시물에 대한 통계 기반 해설을 생성하는 중..."):
        commentary = _cached_event_commentary(
            str(selected_event_id), row_dict, n, None if pd.isna(mean_abs_ar) else float(mean_abs_ar), pct
        )
    summary_part, analysis_part = _split_commentary(commentary)
    if summary_part:
        st.markdown(f"**📝 요약**\n\n{summary_part}")
    st.info(f"📊 **분석**\n\n{analysis_part}")


def _split_commentary(commentary: str) -> tuple:
    """LLM 응답을 '요약: .../분석: ...' 두 문단으로 나눈다. 형식을 못 지켰으면
    통째로 분석 문단에 넣어 정보 손실 없이 그대로 보여준다."""
    text = commentary.strip()
    if "분석:" in text:
        before, _, after = text.partition("분석:")
        summary = before.replace("요약:", "").strip()
        return summary, after.strip()
    return "", text


@st.cache_data(show_spinner=False)
def _cached_event_commentary(event_id: str, row_dict: dict, n: int, mean_abs_ar, pct) -> str:
    ticker_stats = {
        "n": n,
        "mean_abs_ar": f"{mean_abs_ar:.2%}" if mean_abs_ar is not None else "-",
        "percentile": pct,
    }
    return generate_event_commentary(row_dict, ticker_stats)


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
            for col in [
                "event_id",
                "person",
                "posted_at",
                "topic",
                "ticker",
                "description",
                "narrative",
                "narrative_reviewed",
                "source_url",
            ]
            if col in track2.columns
        ]
        rename = {"narrative": "narrative (LLM 원문)", "narrative_reviewed": "narrative (사람 검수)"}
        st.dataframe(track2[cols].rename(columns=rename), use_container_width=True, height=420)
        if "narrative_reviewed" in track2.columns and track2["narrative_reviewed"].fillna("").str.strip().ne("").any():
            st.caption("'narrative (사람 검수)'가 채워진 행은 LLM 원문에 오류가 있어 사람이 고친 버전입니다. 최종적으로는 검수본을 우선 참고하세요.")
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

    tabs = st.tabs(
        ["🔴 실시간 메인", "개요", "이벤트 탐색기", "가설 검증", "방법론 점검", "케이스 스터디", "Ask the Data"]
    )
    with tabs[0]:
        render_live_main(events, data["daily_prices"])
    with tabs[1]:
        render_overview(events, stats, data["placebo_summary"])
    with tabs[2]:
        render_event_explorer(events)
    with tabs[3]:
        render_hypotheses(stats)
    with tabs[4]:
        render_method_checks(data, events)
    with tabs[5]:
        render_case_studies(events, data["figures"])
    with tabs[6]:
        render_ask_data_design(stats, events)


if __name__ == "__main__":
    main()

