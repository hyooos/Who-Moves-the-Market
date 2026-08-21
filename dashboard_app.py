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
from market_mover.case_narratives import (
    generate_event_commentary,
    fetch_article_markdown,
    translate_to_korean,
    answer_data_question,
)
from market_mover.load_posts import clean_text
from market_mover.topic_rules import assign_topic, is_market_relevant, map_ticker
from live_monitor import fetch_trump_feed, fetch_post_text
from find_track2_news_candidates import build_query, fetch_google_news_rss


st.set_page_config(page_title="Who Moves the Market?", layout="wide")


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
            ("contamination_level", "원인 특정 여부"),
            ("sentiment_label", "감성"),
            ("track", "트랙"),
        ]:
            if column not in filtered.columns:
                continue
            options = sorted(str(value) for value in filtered[column].dropna().unique())
            format_func = _contam_label if column == "contamination_level" else (lambda v: v)
            selected = st.multiselect(label, options, default=options, format_func=format_func)
            if selected:
                filtered = filtered[filtered[column].astype(str).isin(selected)]
    return filtered


def _metric_value(value, fallback="0"):
    if value is None:
        return fallback
    return value


_TICKER_NAMES = {"TSLA": "Tesla", "SPY": "S&P 500", "QQQ": "Nasdaq", "GM": "General Motors", "F": "Ford", "RIVN": "Rivian"}

# "CLEAN"은 텍스트가 정제됐다는 뜻이 아니라(그건 preprocess.py가 이미 다른 단계에서 처리함),
# "이 이벤트가 다중게시/매크로/시장충격과 안 섞여서 원인을 특정할 수 있는가"를 뜻하는
# 이벤트 스터디 방법론 용어다(contamination.py, §4-2). 화면에는 오해 없는 표현으로 보여준다.
CONTAM_LABELS = {"CLEAN": "단일요인", "MINOR": "경미중첩", "MAJOR": "다중중첩"}


def _contam_label(value) -> str:
    return CONTAM_LABELS.get(str(value), str(value))


def _looks_korean(text: str) -> bool:
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    return hangul >= max(3, len(text) * 0.15)


@st.cache_data(show_spinner=False)
def _cached_translate_preview(text: str) -> str:
    """게이지 카드의 원문 미리보기(영어 원문)를 한국어로 번역한다. Track2 설명처럼
    이미 한국어인 텍스트는 그대로 두고 불필요한 LLM 호출을 건너뛴다."""
    if not text or _looks_korean(text):
        return text
    translated = translate_to_korean(text)
    return translated.strip() or text


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
    with st.expander("🔴 Trump 실시간 게시물 탐지(반응 크기 아님)", expanded=False):
        st.caption(
            "Truth Social 미러 RSS(trumpstruth.org)를 지금 이 순간 실제로 조회합니다. "
            "반응 크기·백분위는 표시하지 않습니다 — 아직 시장이 반응할 시간이 없었기 때문입니다(위 설명 참고). "
            "topic→종목 분류 결과만 즉시 보여주는 `live_monitor.py`와 동일한 로직입니다. "
            "Musk는 무료 실시간 소스가 없어 지원하지 않습니다. "
            "기본은 최근 24시간 내 게시물만 보여주고, 없으면 3일→5일→7일 순으로 범위를 넓혀서 찾습니다. "
            "아래 '게시' 시각은 RSS가 제공하는 값을 **UTC로 통일해서 표시**합니다 — 실제 게시물 페이지가 "
            "다른 시간대(예: 미국 동부시간)로 보여준다면 그만큼 차이나 보일 수 있습니다."
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
                    pub_date = item.get("pub_date")
                    posted = f"{pub_date.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC" if pub_date else "-"
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
                    st.caption(f"조회 시각(UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}")
                else:
                    st.info("피드에서 항목을 가져오지 못했습니다.")


def _render_ask_data_box(gauge_states: list) -> None:
    """이 프로젝트가 실제로 계산한 값(가설검정 결과, 게이지 통계)만 가지고 답하는
    LLM 채팅형 Q&A. 예전엔 'Ask the Data'가 별도 탭에서 LLM 연결도 안 된 채였는데,
    차트 보면서 바로 여러 번 물어볼 수 있게 옆으로 옮기고 실제로 Ollama에 연결했다."""
    st.markdown("**🤖 이 데이터에 대해 물어보기**")
    st.caption("이 프로젝트가 계산한 값만 가지고 답합니다(예측·조언 아님)")

    history = st.session_state.setdefault("ask_data_history", [])
    chat_box = st.container(height=420)
    with chat_box:
        for role, content in history:
            with st.chat_message(role):
                st.write(content)

    question = st.chat_input("예: QQQ는 왜 이런 결과가 나왔어?")
    if question:
        history.append(("user", question))
        stats = cached_data().get("stats", {})
        tests_summary = {
            name: {"p_value": t.get("p_value"), "유의함": t.get("reject_fdr_0.05")}
            for name, t in stats.get("tests", {}).items()
        }
        context = {
            "이전_대화": [{"역할": r, "내용": c} for r, c in history[-6:-1]],
            "가설검정_결과": tests_summary,
            "효과크기": stats.get("effect_sizes", {}),
            "전체_이벤트_수": stats.get("n_total"),
            "단일요인_이벤트_수": stats.get("n_clean"),
            "종목별_게이지_현황": [
                {
                    "종목": g.get("ticker"),
                    "반응크기_백분위": g.get("percentile"),
                    "최근_게시물_인물": g.get("person"),
                    "최근_게시물_topic": g.get("topic"),
                }
                for g in gauge_states
            ],
        }
        with st.spinner("답변 생성 중..."):
            answer = answer_data_question(question, context)
        history.append(("assistant", answer))
        st.rerun()


@st.cache_data(show_spinner=False)
def _cached_ask_data_answer(question: str, context: dict) -> str:
    return answer_data_question(question, context)


def render_live_main(events: pd.DataFrame, daily_prices: pd.DataFrame) -> None:
    st.subheader("📊 최근 게시물 반응강도 게이지 — 종목별 비교")
    st.caption(
        "이 숫자는 실시간 계산이 아니라 **회고적(retrospective)** 지표입니다. "
        "각 게이지는 아래에서 고른 기간 안에서 그 종목에 매핑된 게시물 중 가장 최근 것을 보여주므로 "
        "종목마다 게시 시점이 다른 게 정상입니다(예: QQQ 관련 최근 게시물과 TSLA 관련 최근 게시물은 날짜가 다름). "
        "이 숫자는 그 게시물 **다음 거래일의 실제 가격 반응**으로 계산되기 때문에, "
        "방금 올라온 글에 대해서는 계산할 수 없습니다 — 시장이 아직 반응할 시간이 없었으니까요. "
        "그리고 이 프로젝트가 가진 데이터 자체가 2025-10-23 이후로는 없어서, 기간을 아무리 넓게 잡아도 "
        "그 이후는 볼 수 없습니다 — 진짜 지금 이 순간의 새 글은 바로 아래 실시간 감지 패널에서만 확인 가능합니다."
    )
    render_live_trump_feed()

    posted_dates = pd.to_datetime(events.get("posted_at"), errors="coerce").dropna() if not events.empty else pd.Series(dtype="datetime64[ns]")
    if not posted_dates.empty:
        data_min, data_max = posted_dates.min().date(), posted_dates.max().date()
        gcol1, gcol2 = st.columns([2, 1])
        with gcol1:
            date_range = st.date_input(
                "게이지 계산 기간(이 기간 안에서 종목별 '가장 최근' 게시물을 찾습니다)",
                value=(data_min, data_max),
                min_value=data_min,
                max_value=data_max,
            )
        with gcol2:
            st.caption(f"데이터 보유 범위: {data_min} ~ {data_max}. 오늘({datetime.now():%Y-%m-%d}) 이후 데이터는 없습니다.")
        if isinstance(date_range, tuple) and len(date_range) == 2:
            gauge_start, gauge_end = date_range
        else:
            gauge_start, gauge_end = data_min, data_max
    else:
        gauge_start = gauge_end = None

    with st.expander("ℹ️ 이 게이지는 어떻게 계산되나요?"):
        st.markdown(
            "- **바늘 위치·%**: 선택한 기간 안에서 그 종목에 매핑된 가장 최근 게시물의 반응이, "
            "그 종목의 과거 사례들과 비교했을 때 **얼마나 큰 편이었는지**(순위, 방향 아님)를 나타냅니다.\n"
            "- **📈/📉 배지**: 그 게시물 다음 거래일 주가가 실제로 **올랐는지/내렸는지**와 그 폭(%)입니다. "
            "게이지 자체는 크기만 보여주므로 방향은 이 배지로 따로 확인하세요.\n"
            "- 회고적(retrospective) 지표입니다 — 미래 반응을 예측하지 않습니다(§8-1)."
        )

    gauge_states = compute_ticker_gauges(events, tickers=("QQQ", "SPY", "TSLA"), start_date=gauge_start, end_date=gauge_end)
    gauge_cols = st.columns(3)
    with st.spinner("원문 미리보기 번역 중..."):
        for col, gstate in zip(gauge_cols, gauge_states):
            if gstate.get("text_preview"):
                gstate = dict(gstate, text_preview=_cached_translate_preview(gstate["text_preview"]))
            with col:
                st.markdown(render_single_gauge_html(gstate), unsafe_allow_html=True)

    st.divider()
    left_col, right_col = st.columns([7, 3])
    with right_col:
        _render_ask_data_box(gauge_states)
    with left_col:
        _render_price_chart_section(events, daily_prices, gauge_states)


def _render_price_chart_section(events: pd.DataFrame, daily_prices: pd.DataFrame, gauge_states: list) -> None:
    st.subheader("종목 가격 추이")
    st.caption("차트를 클릭하면 그 시점 게시물 원문을 볼 수 있습니다(점이 안 보여도 클릭은 됩니다).")

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

    # 2025-04-14 이후 사건(Track2)은 차트에 딱 6개만 뚝뚝 떨어져 있어서 점을 정확히
    # 클릭하기 어렵다는 피드백이 있었다 — 클릭 대신 목록에서 바로 골라 보는 방법을 추가한다.
    track2 = events[events.get("track", pd.Series(dtype=str)).eq("track2_manual")] if "track" in events.columns else events.iloc[0:0]
    if not track2.empty:
        track2_options = ["(선택 안 함)"] + [
            f"{r.get('event_date')} · {r.get('person')} · {r.get('ticker')} · {r.get('topic')}"
            for _, r in track2.sort_values("event_date").iterrows()
        ]
        track2_ids = [None] + track2.sort_values("event_date")["event_id"].tolist()
        track2_pick = st.selectbox(
            "📌 2025-04 이후 주요 사건 바로가기(클릭이 잘 안 될 때 여기서 바로 선택)",
            track2_options,
        )
        if track2_pick != "(선택 안 함)":
            picked_id = track2_ids[track2_options.index(track2_pick)]
            _render_event_detail(events, picked_id)
            return

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
                        marker=dict(size=16, color="rgba(30,41,59,0.02)", line=dict(width=0)),
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
                marker=dict(size=16, color="rgba(30,41,59,0.02)", line=dict(width=0)),
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


def _render_news_search(row: pd.Series, event_id) -> None:
    """Kaggle 원본에는 게시물 URL이 없어서, 대신 그 시점 전후로 실제 관련 뉴스가
    있었는지 Google News RSS로 그 자리에서 자동검색해 보여준다."""
    if st.button("🔍 이 시점 관련 기사 자동검색(Google News)", key=f"news_search_{event_id}"):
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
        if pd.isna(event_date):
            st.info("이벤트 날짜가 없어 검색할 수 없습니다.")
            return
        query = build_query(row)
        ticker_name = _TICKER_NAMES.get(row.get("ticker"), row.get("ticker"))
        window_start = event_date - pd.Timedelta(days=2)
        window_end = event_date + pd.Timedelta(days=2)
        with st.spinner(f'Google News에서 "{query}" 검색 중...'):
            try:
                articles = fetch_google_news_rss(query, window_start, window_end, limit=5)
            except Exception as exc:
                st.error(f"뉴스 검색 실패: {exc}")
                return
        st.caption(
            f"검색 기준: 인물(**{row.get('person')}**) + topic(**{row.get('topic')}**) + 종목명(**{ticker_name}**)을 "
            f"조합해 검색어 \"{query}\"를 만들고, 게시 날짜 {event_date.date()} 기준 전후 ±2일"
            f"({window_start.date()} ~ {window_end.date()}) 사이 뉴스를 찾았습니다. 게시물 원문 문구가 아니라 "
            f"이 메타데이터 조합으로만 검색했습니다."
        )
        if not articles:
            st.info(f"'{query}' 관련 기사를 {event_date.date()} 전후 4일 내에서 찾지 못했습니다.")
            return
        st.dataframe(
            pd.DataFrame(articles)[["title", "source", "pub_date", "link"]],
            use_container_width=True,
            column_config={"link": st.column_config.LinkColumn("링크", display_text="열기 ↗")},
        )


def _render_selected_event(click, events: pd.DataFrame) -> None:
    selected_event_id = None
    points = (click or {}).get("selection", {}).get("points", [])
    for point in points:
        customdata = point.get("customdata")
        if customdata:
            selected_event_id = customdata[0]
            break

    if selected_event_id is None:
        st.caption("차트를 클릭하면 그 시점 게시물 원문이 여기에 표시됩니다(점이 안 보여도 클릭은 됩니다).")
        return

    _render_event_detail(events, selected_event_id)


def _render_event_detail(events: pd.DataFrame, selected_event_id) -> None:
    if "event_id" not in events.columns:
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

    article_text = None
    if is_track2 and has_source:
        # 2025-04-14 이후 사건(Track2)은 원본 게시물이 아니라 뉴스 기반이라, 링크만
        # 던져두지 않고 기사 본문을 바로 그 자리에서 가져와 보여준다.
        article_text = _render_full_article(source_url, selected_event_id)
    elif not is_track2 and not has_source:
        _render_news_search(row, selected_event_id)

    row_ticker = row.get("ticker")
    row_topic = row.get("topic")
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
    same_topic = same_ticker[same_ticker["topic"] == row_topic] if "topic" in same_ticker.columns else same_ticker.iloc[0:0]
    topic_mean_abs_ar = (
        float(same_topic["abnormal_return"].abs().mean()) if len(same_topic) >= 5 else None
    )
    row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}

    with st.spinner("이 게시물에 대한 통계 기반 해설을 생성하는 중..."):
        commentary = _cached_event_commentary(
            str(selected_event_id),
            row_dict,
            n,
            None if pd.isna(mean_abs_ar) else float(mean_abs_ar),
            pct,
            topic_mean_abs_ar,
            article_text,
        )
    summary_part, analysis_part = _split_commentary(commentary)
    if summary_part:
        st.markdown(f"**📝 요약**\n\n{summary_part}")
    st.info(f"📊 **분석**\n\n{analysis_part}")

    _render_price_context_chart(row_ticker, row.get("event_date"))


@st.cache_data(show_spinner=False)
def _cached_article_markdown(url: str) -> str:
    return fetch_article_markdown(url)


@st.cache_data(show_spinner=False)
def _cached_translate_article(text: str) -> str:
    return translate_to_korean(text)


def _render_full_article(source_url: str, event_id):
    with st.spinner("기사 원문을 불러오는 중..."):
        try:
            article_md = _cached_article_markdown(source_url)
        except Exception as exc:
            st.warning(f"기사 원문을 불러오지 못했습니다: {exc}")
            return None
    if not article_md.strip():
        st.warning("기사 원문을 가져오지 못했습니다(사이트가 자동 수집을 막았을 수 있습니다). 위 링크로 직접 열어보세요.")
        return None
    with st.expander("📰 기사 원문 전체 보기", expanded=True):
        st.caption(f"Jina Reader로 지금 이 순간 실시간으로 가져온 기사 본문입니다 · 출처: {source_url}")
        preview = article_md[:4000]
        show_korean = st.toggle("🇰🇷 한국어로 번역해서 보기", key=f"translate_{event_id}")
        if show_korean:
            with st.spinner("기사를 한국어로 번역하는 중... (원문이 길면 시간이 걸립니다)"):
                st.markdown(_cached_translate_article(preview))
        else:
            st.markdown(preview + ("..." if len(article_md) > 4000 else ""))
    return article_md


def _render_price_context_chart(ticker: str, event_date) -> None:
    """"영향 점수가 몇 점이다"는 숫자로는 감이 안 온다는 피드백을 반영해, 대신
    그 시점 전후 실제 주가가 평소보다 튀었는지를 캔들스틱으로 눈으로 바로 보여준다.
    한국 주식 차트 관행대로 상승은 빨강, 하락은 파랑으로 칠한다(미국 관행과 반대)."""
    event_date = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(event_date):
        return
    daily_prices = cached_data()["daily_prices"]
    if daily_prices.empty:
        return
    prices = daily_prices[daily_prices["ticker"] == ticker].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.dropna(subset=["date", "close"]).sort_values("date")
    if prices.empty:
        return
    prices["ma5"] = prices["close"].rolling(5).mean()
    prices["daily_return"] = prices["close"].pct_change() * 100

    baseline = prices[prices["date"] < event_date].tail(60)
    typical_move = float(baseline["daily_return"].abs().median()) if not baseline.empty else None

    window = prices[
        (prices["date"] >= event_date - pd.Timedelta(days=15))
        & (prices["date"] <= event_date + pd.Timedelta(days=15))
    ].dropna(subset=["open", "high", "low", "close"])
    if window.empty:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=window["date"],
            open=window["open"], high=window["high"], low=window["low"], close=window["close"],
            increasing_line_color="#ef4444", increasing_fillcolor="#ef4444",
            decreasing_line_color="#3b82f6", decreasing_fillcolor="#3b82f6",
            name=ticker,
        )
    )
    if window["ma5"].notna().any():
        fig.add_trace(
            go.Scatter(x=window["date"], y=window["ma5"], mode="lines", line=dict(color="#f8fafc", width=1.3), name="5일 이동평균")
        )
    fig.add_vline(x=event_date, line=dict(color="#facc15", width=1.5, dash="dot"))
    fig.add_annotation(x=event_date, y=1, yref="paper", text="이 게시물", showarrow=False, yshift=10, font=dict(color="#facc15", size=11))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="가격($)",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")

    with st.expander(f"📉 {ticker} 종목, 그 시점 전후 실제 주가 변동(참고용)", expanded=True):
        typical_text = f"약 ±{typical_move:.1f}%" if typical_move is not None else "계산 불가"
        st.caption(
            f"빨간 캔들=상승, 파란 캔들=하락(한국 주식 차트 관행). 노란 점선이 이 게시물이 있었던 날입니다. "
            f"흰 선은 5일 이동평균이고, 이 종목은 평소(직전 60거래일) 하루에 보통 {typical_text} 정도 움직입니다."
        )
        st.plotly_chart(fig, use_container_width=True)


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
def _cached_event_commentary(
    event_id: str, row_dict: dict, n: int, mean_abs_ar, pct, topic_mean_abs_ar=None, article_text=None
) -> str:
    ticker_stats = {
        "n": n,
        "mean_abs_ar": f"{mean_abs_ar:.2%}" if mean_abs_ar is not None else "-",
        "percentile": pct,
        "topic_mean_abs_ar": f"{topic_mean_abs_ar:.2%}" if topic_mean_abs_ar is not None else None,
    }
    return generate_event_commentary(row_dict, ticker_stats, content_text=article_text)


def render_overview(events: pd.DataFrame, stats: dict, placebo_summary: dict) -> None:
    clean = events[events.get("contamination_level", pd.Series(dtype=str)).eq("CLEAN")] if not events.empty else events
    cols = st.columns(4)
    cols[0].metric("전체 이벤트", f"{len(events):,}")
    cols[1].metric("단일요인 이벤트", f"{len(clean):,}")
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
    top_table = events.sort_values("impact_score", ascending=False)[top_cols].head(25).copy()
    if "contamination_level" in top_table.columns:
        top_table["contamination_level"] = top_table["contamination_level"].map(_contam_label)
    st.dataframe(top_table, use_container_width=True)


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
        left_table = filtered.sort_values("impact_score", ascending=False)[table_cols].copy()
        if "contamination_level" in left_table.columns:
            left_table["contamination_level"] = left_table["contamination_level"].map(_contam_label)
        st.dataframe(left_table, use_container_width=True, height=540)
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
            detail_view = event[detail_cols].copy()
            if "contamination_level" in detail_view.index:
                detail_view["contamination_level"] = _contam_label(detail_view["contamination_level"])
            st.dataframe(detail_view.to_frame("값"), use_container_width=True)
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
        counts = events["contamination_level"].fillna("UNKNOWN").map(lambda v: _contam_label(v) if v != "UNKNOWN" else v).value_counts().reset_index()
        counts.columns = ["원인 특정 여부", "이벤트 수"]
        fig = px.bar(counts, x="원인 특정 여부", y="이벤트 수", text="이벤트 수")
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
    st.caption("이 프로젝트가 계산한 값(가설검정 결과 등)만 가지고 로컬 LLM이 답합니다. 예측·투자조언은 하지 않습니다.")
    question = st.text_input("질문", key="ask_data_tab")
    if not question:
        st.info("질문을 입력하면 답변이 여기에 표시됩니다.")
        return
    clean_n = int(stats.get("n_clean", 0))
    total_n = int(stats.get("n_total", len(events)))
    tests_summary = {
        name: {"p_value": t.get("p_value"), "유의함": t.get("reject_fdr_0.05")}
        for name, t in stats.get("tests", {}).items()
    }
    context = {
        "가설검정_결과": tests_summary,
        "효과크기": stats.get("effect_sizes", {}),
        "전체_이벤트": total_n,
        "단일요인_이벤트": clean_n,
        "FDR_유의_검정_수": significant_test_count(stats),
    }
    with st.spinner("답변 생성 중..."):
        answer = _cached_ask_data_answer(question, context)
    st.info(answer)
    with st.expander("근거로 사용된 데이터 보기"):
        st.json(context, expanded=False)


def main() -> None:
    st.title("Who Moves the Market?")
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

