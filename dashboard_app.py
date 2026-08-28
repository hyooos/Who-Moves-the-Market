import hashlib
import html
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from market_mover.case_narratives import (
    DEFAULT_LLM_MODELS,
    answer_data_question,
    generate_event_commentary,
    test_llm_connection,
    translate_to_korean,
)
from market_mover.dashboard_data import load_dashboard_data
from market_mover.dashboard_widgets import compute_ticker_gauges, render_single_gauge_html
from market_mover.load_posts import clean_text
from market_mover.topic_rules import assign_topic, is_market_relevant, map_ticker
from live_monitor import fetch_post_text, fetch_trump_feed


st.set_page_config(
    page_title="Who Moves the Market?",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PERSON_LABELS = {"Musk": "일론 머스크", "Trump": "도널드 트럼프"}
TICKER_LABELS = {
    "TSLA": "테슬라(TSLA)",
    "QQQ": "나스닥 100(QQQ)",
    "SPY": "S&P 500(SPY)",
    "GM": "제너럴모터스(GM)",
    "F": "포드(F)",
    "RIVN": "리비안(RIVN)",
}
TOPIC_LABELS = {
    "ai": "인공지능",
    "autonomy_robotaxi": "자율주행·로보택시",
    "china": "중국 관련",
    "doge_budget_feud": "정부효율부·예산 갈등",
    "fed_rates": "연방준비제도·금리",
    "jobs_economy": "고용·경제",
    "macro_economy": "거시경제",
    "semiconductor": "반도체",
    "tariff": "관세",
    "tesla_business": "테슬라 사업",
    "trade_policy": "무역정책",
}
SOURCE_LABELS = {
    "track1_auto": "SNS 원문 사건",
    "track2_manual": "검증된 뉴스 사례",
    "track2_news_auto": "자동 수집 뉴스 사건",
}
SESSION_LABELS = {
    "premarket": "장 시작 전",
    "regular_session": "장중",
    "afterhours": "장 마감 후",
    "market_closed": "휴장일",
    "unknown_date_only": "날짜만 확인됨",
}
PROVIDER_LABELS = {
    "gemini": "Gemini",
    "groq": "Groq",
    "ollama": "Ollama",
    "none": "연결 안 됨",
}
# 연구에서 사용하는 고정 분석 범위. 사건이 없는 날짜도 기간 선택에서 유지합니다.
ANALYSIS_START = pd.Timestamp("2023-01-03").date()
ANALYSIS_END = pd.Timestamp("2025-10-23").date()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background:#f6f8fb; color:#172033; }
        header[data-testid="stHeader"] { background:rgba(246,248,251,.92); }
        .block-container { max-width:1380px; padding-top:1.4rem; padding-bottom:4rem; }
        [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e5eaf0; }
        [data-testid="stMetric"] {
            background:#ffffff; border:1px solid #e4e9ef; border-radius:14px;
            padding:14px 16px; box-shadow:0 5px 18px rgba(15,23,42,.035);
        }
        [data-testid="stMetricLabel"] { color:#718096; font-size:.8rem; }
        [data-testid="stMetricValue"] { color:#172033; }
        .app-kicker { color:#e0475b; font-weight:800; font-size:.78rem; letter-spacing:.08em; }
        .app-title { color:#13213a; font-size:2.5rem; font-weight:850; letter-spacing:-.045em; margin:.25rem 0 .25rem; }
        .app-copy { color:#5f6e82; font-size:1rem; line-height:1.7; max-width:920px; }
        .service-path { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:1.15rem 0 1.25rem; }
        .path-item { background:#fff; border:1px solid #e4e9ef; border-radius:12px; padding:12px 14px; color:#536174; font-size:.86rem; }
        .path-item b { color:#172033; margin-right:6px; }
        .section-title { color:#172033; font-size:1.48rem; font-weight:820; letter-spacing:-.025em; margin:1.75rem 0 .3rem; }
        .section-copy { color:#6b7788; font-size:.86rem; line-height:1.6; margin-bottom:.9rem; }
        .overview-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:.9rem 0 1rem; }
        .overview-card { background:#fff; border:1px solid #e4e9ef; border-radius:15px; padding:15px 16px; box-shadow:0 6px 20px rgba(15,23,42,.04); min-width:0; }
        .overview-label { color:#718096; font-size:.78rem; font-weight:700; margin-bottom:7px; }
        .overview-value { color:#172033; font-size:clamp(1.45rem,2vw,1.9rem); font-weight:820; letter-spacing:-.035em; line-height:1.1; min-width:0; overflow-wrap:anywhere; }
        .overview-value.period { font-size:clamp(1rem,1.45vw,1.35rem); line-height:1.3; letter-spacing:-.025em; white-space:normal; }
        .overview-note { color:#91a0b3; font-size:.7rem; margin-top:5px; line-height:1.45; }
        .range-heading { color:#172033; font-size:.9rem; font-weight:800; margin:.2rem 0 .2rem; }
        .range-copy { color:#718096; font-size:.76rem; line-height:1.55; margin-bottom:.45rem; }
        .detail-card { background:#fff; border:1px solid #e3e8ef; border-radius:16px; padding:20px 22px; box-shadow:0 7px 24px rgba(15,23,42,.04); margin:.5rem 0 1rem; }
        .badge-row { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:12px; }
        .badge { background:#f0f3f7; color:#48566a; padding:4px 9px; border-radius:7px; font-size:.76rem; font-weight:720; }
        .badge.source { background:#eef4ff; color:#275db5; }
        .detail-title { color:#172033; font-size:1.12rem; font-weight:820; margin-bottom:9px; }
        .detail-text { color:#344256; font-size:.92rem; line-height:1.72; white-space:pre-wrap; }
        .note-box { background:#f8fafc; border:1px solid #e5eaf0; border-radius:11px; padding:11px 13px; color:#64748b; font-size:.82rem; line-height:1.55; margin-top:12px; }
        .summary-box { background:linear-gradient(135deg,#eef4ff,#f7f9fd); border:1px solid #d9e4f6; border-radius:13px; padding:15px 17px; color:#30415b; line-height:1.68; margin:.7rem 0; }
        .summary-box b { color:#172033; }
        .event-kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:.75rem 0 .8rem; }
        .event-kpi { background:#fff; border:1px solid #e4e9ef; border-radius:14px; padding:14px 15px; min-width:0; box-shadow:0 5px 18px rgba(15,23,42,.035); }
        .event-kpi-label { color:#718096; font-size:.76rem; font-weight:700; margin-bottom:6px; }
        .event-kpi-value { color:#172033; font-size:clamp(1.25rem,1.75vw,1.65rem); font-weight:820; letter-spacing:-.035em; line-height:1.18; overflow-wrap:anywhere; }
        .event-kpi-value.positive { color:#c93d50; }
        .event-kpi-value.negative { color:#2d63bd; }
        .event-kpi-note { color:#8a98aa; font-size:.69rem; line-height:1.45; margin-top:5px; min-height:1.95em; }
        .timeline-guide { background:#eef4ff; border:1px solid #dce7f8; border-radius:12px; padding:10px 13px; color:#607089; font-size:.75rem; line-height:1.52; margin-top:28px; }
        .timeline-guide b { color:#244f94; }
        .timeline-selection { display:flex; justify-content:space-between; align-items:center; gap:22px; background:#fff; border:1px solid #dfe6ef; border-radius:15px; padding:16px 18px; margin:.55rem 0 .65rem; box-shadow:0 6px 20px rgba(15,23,42,.035); }
        .timeline-selection.empty { justify-content:flex-start; color:#526176; }
        .timeline-selection.empty b { color:#172033; margin-right:12px; }
        .timeline-selection.empty span { color:#7a8799; font-size:.8rem; }
        .timeline-selection-label { color:#8a97a9; font-size:.71rem; font-weight:700; margin-bottom:3px; }
        .timeline-selection-date { color:#172033; font-size:1.28rem; font-weight:820; letter-spacing:-.025em; }
        .timeline-selection-stats { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:24px; }
        .timeline-selection-stats span { min-width:110px; }
        .timeline-selection-stats b { display:block; color:#26364f; font-size:.92rem; }
        .timeline-selection-stats small { display:block; color:#8a97a9; font-size:.68rem; margin-top:2px; }
        .timeline-picker-head { display:flex; align-items:center; gap:13px; margin:.8rem 0 .65rem; padding:15px 17px; background:linear-gradient(135deg,#fff5f6 0%,#fff 72%); border:1px solid #f2d7dc; border-left:4px solid #df3e52; border-radius:13px; }
        .timeline-picker-copy { min-width:0; flex:1; }
        .timeline-picker-kicker { color:#d53b50; font-size:.68rem; font-weight:820; letter-spacing:.075em; margin-bottom:3px; }
        .timeline-picker-title { color:#172033; font-size:1.08rem; font-weight:840; letter-spacing:-.025em; line-height:1.3; }
        .timeline-picker-help { color:#708096; font-size:.75rem; line-height:1.5; margin-top:3px; }
        .timeline-picker-count { flex:0 0 auto; min-width:54px; text-align:center; background:#fff; color:#c93449; border:1px solid #efcbd2; border-radius:999px; padding:7px 11px; font-size:.78rem; font-weight:820; }
        .timeline-picked-event { display:flex; flex-wrap:wrap; align-items:center; gap:7px 14px; background:#f8fafc; border:1px solid #e5eaf0; border-radius:10px; padding:10px 12px; margin:.15rem 0 .75rem; color:#536174; font-size:.75rem; line-height:1.45; }
        .timeline-picked-event b { color:#172033; }
        .timeline-picked-event .picked-label { color:#d53b50; font-weight:820; }
        .sentiment-card { background:#fff; border:1px solid #e4e9ef; border-radius:13px; padding:14px 16px; min-height:98px; }
        .sentiment-label { color:#6b7788; font-size:.78rem; margin-bottom:5px; }
        .sentiment-value { color:#172033; font-size:1.28rem; font-weight:820; }
        .sentiment-note { color:#7b8797; font-size:.74rem; margin-top:4px; }
        .status-pill { display:inline-block; background:#f1f5f9; color:#536174; border:1px solid #e2e8f0; border-radius:999px; padding:6px 10px; font-size:.76rem; font-weight:700; margin-bottom:6px; }
        div[data-testid="stExpander"] { background:#fff; border:1px solid #e4e9ef; border-radius:12px; }
        .stButton > button { border-radius:10px; font-weight:720; }
        .stTabs [data-baseweb="tab-list"] { gap:1.6rem; border-bottom:1px solid #dfe5ec; }
        .stTabs [data-baseweb="tab"] { color:#536174; font-weight:720; padding:.7rem .1rem .8rem; }
        .stTabs [aria-selected="true"] { color:#df3e52 !important; }
        .stTabs [data-baseweb="tab-highlight"] { background:#df3e52; height:2px; }
        [data-testid="stChatMessage"] { background:#fff; border:1px solid #e5eaf0; border-radius:12px; }
        @media (max-width:1050px) { .overview-grid, .event-kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
        @media (max-width:900px) { .service-path { grid-template-columns:1fr; } .app-title { font-size:2rem; } .timeline-selection { align-items:flex-start; flex-direction:column; } .timeline-selection-stats { justify-content:flex-start; } }
        @media (max-width:620px) { .overview-grid, .event-kpi-grid { grid-template-columns:1fr; } .timeline-picker-head { align-items:flex-start; } .timeline-picker-count { min-width:auto; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_data() -> dict:
    return load_dashboard_data()


def _safe_text(*values, fallback="-") -> str:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text and text.lower() not in {"nan", "nat", "none", "<na>"}:
            return text
    return fallback


def _number(value, default=None):
    parsed = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(parsed) else float(parsed)


def _format_pct(value, digits: int = 2, fallback: str = "-") -> str:
    number = _number(value)
    return fallback if number is None else f"{number * 100:+.{digits}f}%"


def _format_number(value, digits: int = 2, fallback: str = "-") -> str:
    number = _number(value)
    return fallback if number is None else f"{number:.{digits}f}"




def _parse_event_datetime(value):
    """Track1의 YYYY-MM-DD와 Track2의 YYYY-MM-DD HH:MM:SS가 섞여도 모두 파싱합니다."""
    try:
        return pd.to_datetime(value, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        # pandas 구버전 호환
        return pd.to_datetime(value, errors="coerce")

def _person_label(value) -> str:
    return PERSON_LABELS.get(str(value), _safe_text(value))


def _ticker_label(value) -> str:
    return TICKER_LABELS.get(str(value), _safe_text(value))


def _topic_label(value) -> str:
    return TOPIC_LABELS.get(str(value), _safe_text(value).replace("_", " "))


def _source_label(value) -> str:
    return SOURCE_LABELS.get(str(value), _safe_text(value, fallback="자료 구분 미확인"))


def _is_news_track(value) -> bool:
    return str(value).startswith("track2")


def _session_label(value) -> str:
    return SESSION_LABELS.get(str(value), _safe_text(value, fallback="시각 확인 필요"))


def _sentiment_label(value) -> str:
    raw = _safe_text(value, fallback="").lower()
    if "positive" in raw or raw == "label_2":
        return "긍정"
    if "negative" in raw or raw == "label_0":
        return "부정"
    if "neutral" in raw or raw == "label_1":
        return "중립"
    return "감성 미분석"


def _sentiment_display(row: pd.Series) -> str:
    """화면에 보여줄 감성 상태를 사건 유형까지 고려해 정리합니다.

    Track2는 뉴스·기자회견·수동 등록 사례가 섞여 있어 SNS 문장 감성분석의
    직접 대상이 아닙니다. 따라서 NaN을 '분석 전'처럼 보이게 하지 않고
    '감성 미적용'으로 명확히 구분합니다.
    """
    if _is_news_track(row.get("track")):
        return "감성 미적용"
    return _sentiment_label(row.get("sentiment_label"))


def _sentiment_note(row: pd.Series, display_value: str) -> str:
    if display_value == "감성 미적용":
        return "뉴스 사건은 감성분석 대상에서 제외"
    if display_value == "감성 미분석":
        return "--add-sentiment 실행 시 표시"
    confidence = _number(row.get("sentiment_confidence"))
    return "게시물 문장 어조" if confidence is None else f"모델 신뢰도 {confidence * 100:.0f}%"


def _compact_period(start_date, end_date) -> str:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start.year == end.year and start.month == end.month:
        return f"{start:%Y.%m.%d}–{end:%m.%d}"
    if start.year == end.year:
        return f"{start:%Y.%m.%d}–{end:%m.%d}"
    return f"{start:%Y.%m.%d}–{end:%Y.%m.%d}"


def _looks_korean(text: str) -> bool:
    if not text:
        return False
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    return hangul >= max(3, len(text) * 0.15)


def _short_text(row: pd.Series, limit: int = 105) -> str:
    text = _safe_text(row.get("description"), row.get("text_raw"), row.get("text_clean"), fallback="내용 없음")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _decorate_events(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    if work.empty:
        return work
    work["_date"] = _parse_event_datetime(work.get("event_date"))
    work["_id"] = work.get("event_id", pd.Series(index=work.index, dtype=str)).astype(str)
    ordered = work.sort_values(["_date", "_id"], na_position="last").index
    width = max(4, len(str(len(work))))
    number_map = {row_index: number for number, row_index in enumerate(ordered, start=1)}
    work["사건"] = [f"사건 {number_map[index]:0{width}d}" for index in work.index]
    work["날짜"] = work["_date"].dt.strftime("%Y-%m-%d").fillna("날짜 미확인")
    work["인물"] = work.get("person", pd.Series(index=work.index, dtype=str)).map(_person_label)
    work["시장"] = work.get("ticker", pd.Series(index=work.index, dtype=str)).map(_ticker_label)
    work["주제"] = work.get("topic", pd.Series(index=work.index, dtype=str)).map(_topic_label)
    work["자료 형태"] = work.get("track", pd.Series(index=work.index, dtype=str)).map(_source_label)
    work["감성"] = work.apply(_sentiment_display, axis=1)
    work["실제 등락률"] = pd.to_numeric(work.get("stock_return"), errors="coerce")
    work["시장 대비 등락률"] = pd.to_numeric(work.get("abnormal_return"), errors="coerce")
    work["반응 강도"] = pd.to_numeric(work.get("impact_score"), errors="coerce")
    work["내용"] = work.apply(_short_text, axis=1)
    return work


def _auto_provider() -> str:
    configured = os.getenv("MARKET_MOVER_LLM_PROVIDER", "").strip().lower()
    if configured in {"gemini", "groq", "ollama", "none"}:
        return configured
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "none"


def _llm_config_token(provider: str, model_name: str | None) -> str:
    """캐시에 API 키 원문을 남기지 않고 현재 연결 설정만 구분합니다."""
    provider = (provider or "none").lower()
    if provider == "gemini":
        credential = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    elif provider == "groq":
        credential = os.getenv("GROQ_API_KEY") or ""
    elif provider == "ollama":
        credential = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    else:
        credential = ""
    credential_hash = hashlib.sha256(credential.encode("utf-8")).hexdigest() if credential else "no-credential"
    raw = f"{provider}\0{model_name or ''}\0{credential_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _connection_state(provider: str, model_name: str | None) -> str:
    if provider == "none":
        return "off"
    status = st.session_state.get("llm_connection_status", {})
    if status.get("config_token") != _llm_config_token(provider, model_name):
        return "untested"
    return "connected" if status.get("ok") is True else "failed"


def _connection_label(provider: str, model_name: str | None, *, compact: bool = False) -> str:
    state = _connection_state(provider, model_name)
    provider_label = PROVIDER_LABELS.get(provider, provider)
    if state == "connected":
        return f"{provider_label} · 연결됨"
    if state == "failed":
        return f"{provider_label} · 연결 실패"
    if state == "untested":
        return f"{provider_label} · 확인 필요"
    return "연결 안 됨" if compact else "기본 데이터 답변"


def _render_ai_settings() -> tuple[str, str | None]:
    provider_options = ["자동", "Gemini", "Groq", "Ollama", "사용하지 않음"]
    provider_map = {"자동": _auto_provider(), "Gemini": "gemini", "Groq": "groq", "Ollama": "ollama", "사용하지 않음": "none"}
    with st.popover("AI 설정"):
        choice = st.selectbox("요약·질문 답변 서비스", provider_options, key="service_ai_provider_choice")
        provider = provider_map[choice]
        model_name = DEFAULT_LLM_MODELS.get(provider)
        if provider == "gemini":
            key = st.text_input("Gemini API 키", type="password", key="service_gemini_key")
            model_name = st.text_input(
                "Gemini 모델",
                value=model_name or "gemini-3.6-flash",
                key="service_gemini_model",
                help="기본값은 정식 Flash 모델인 gemini-3.6-flash입니다.",
            ).strip() or DEFAULT_LLM_MODELS["gemini"]
            if key:
                os.environ["GEMINI_API_KEY"] = key.strip()
                st.session_state["_service_gemini_key_injected"] = True
            elif st.session_state.get("_service_gemini_key_injected"):
                os.environ.pop("GEMINI_API_KEY", None)
                st.session_state["_service_gemini_key_injected"] = False
        elif provider == "groq":
            key = st.text_input("Groq API 키", type="password", key="service_groq_key")
            if key:
                os.environ["GROQ_API_KEY"] = key.strip()
                st.session_state["_service_groq_key_injected"] = True
            elif st.session_state.get("_service_groq_key_injected"):
                os.environ.pop("GROQ_API_KEY", None)
                st.session_state["_service_groq_key_injected"] = False
        elif provider == "ollama":
            model_name = st.text_input("Ollama 모델", value=model_name or "qwen2.5:7b") or None

        config_token = _llm_config_token(provider, model_name)
        if provider != "none":
            has_credentials = provider == "ollama" or bool(
                (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
                if provider == "gemini"
                else os.getenv("GROQ_API_KEY")
            )
            if st.button(
                "실제 연결 테스트",
                key="service_ai_connection_test",
                width="stretch",
                disabled=not has_credentials,
                help=None if has_credentials else "먼저 API 키를 입력해주세요.",
            ):
                with st.spinner("AI 서비스에 실제 요청을 보내는 중..."):
                    ok, message = test_llm_connection(provider, model_name=model_name)
                st.session_state["llm_connection_status"] = {
                    "config_token": config_token,
                    "ok": ok,
                    "message": message,
                }

            state = _connection_state(provider, model_name)
            status = st.session_state.get("llm_connection_status", {})
            if state == "connected":
                st.success(f"연결됨 · {model_name}")
            elif state == "failed":
                st.error(f"연결 실패 · {status.get('message', '설정을 확인해주세요.')}")
            elif has_credentials:
                st.info("키 또는 모델이 준비되었습니다. ‘실제 연결 테스트’를 눌러 확인해주세요.")
            else:
                st.caption("API 키를 입력한 뒤 실제 연결 테스트를 진행해주세요.")
        allow_online_translation = st.checkbox(
            "AI가 없을 때 기본 온라인 번역 허용",
            value=st.session_state.get("allow_online_translation", False),
            key="service_online_translation_consent",
        )
        if allow_online_translation:
            st.caption("AI 번역이 실패하면 선택한 영문 원문을 기본 온라인 번역 서비스로 전송합니다.")
        if provider == "none":
            st.caption("AI 연결이 없어도 데이터 요약과 질문 답변은 기본 기능으로 동작합니다.")
        else:
            st.caption(f"현재 선택: {PROVIDER_LABELS[provider]} · {model_name}")
    st.session_state["active_llm_provider"] = provider
    st.session_state["active_llm_model"] = model_name
    st.session_state["active_llm_config_token"] = config_token
    st.session_state["allow_online_translation"] = allow_online_translation
    return provider, model_name


def _current_llm() -> tuple[str, str | None]:
    return (st.session_state.get("active_llm_provider", _auto_provider()), st.session_state.get("active_llm_model"))


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_latest_trump_posts() -> dict:
    """Read Trump's latest posts via the free mirror RSS and Jina Reader.

    The mirror is unofficial, so this is intentionally a user-triggered lookup and
    never mixed into the historical event-study results automatically.
    """
    items = fetch_trump_feed(30)
    now = datetime.now(timezone.utc)
    selected_items = []
    window_days = None
    for days in (1, 3, 5, 7):
        cutoff = now - timedelta(days=days)
        candidates = []
        for item in items:
            published = item.get("pub_date")
            if published is None:
                continue
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if published >= cutoff:
                candidates.append(item)
        if candidates:
            selected_items = candidates[:8]
            window_days = days
            break

    if not selected_items:
        return {"posts": [], "window_days": None}

    def load_text(item: dict) -> str:
        try:
            return fetch_post_text(item.get("link", ""))
        except Exception:
            return ""

    with ThreadPoolExecutor(max_workers=min(4, len(selected_items))) as pool:
        texts = list(pool.map(load_text, selected_items))

    posts = []
    for item, raw_text in zip(selected_items, texts):
        text = clean_text(raw_text) if raw_text else ""
        relevant = is_market_relevant(text, "Trump") if text else False
        topic = assign_topic(text, "Trump") if text else None
        ticker = map_ticker("Trump", topic) if relevant and topic else None
        published = item.get("pub_date")
        if published is not None and published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        posts.append(
            {
                "text": text,
                "link": item.get("link", ""),
                "published": published,
                "market_relevant": relevant,
                "topic": topic,
                "ticker": ticker,
            }
        )
    return {"posts": posts, "window_days": window_days}


def _render_latest_trump_posts() -> None:
    st.markdown('<div class="section-title">트럼프 최신 Truth Social 게시물</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">무료 비공식 미러 RSS에서 목록을 확인한 뒤 Jina Reader로 본문을 읽습니다. '
        '최근 24시간에 글이 없으면 3일·5일·7일 순으로 범위를 넓힙니다.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("최신 게시물 조회", expanded=False):
        st.caption(
            "Trump 전용 기능입니다. Musk는 무료로 안정적인 실시간 X 소스가 없어 지원하지 않습니다. "
            "새 게시물은 아직 시장 반응이 확정되지 않았으므로 기존 사건·계기판에 자동으로 합치지 않습니다."
        )
        if st.button("트럼프 최신 게시물 확인", key="fetch_latest_trump_posts", type="primary"):
            try:
                with st.spinner("Truth Social 미러 RSS와 게시물 본문을 확인하는 중..."):
                    st.session_state["latest_trump_result"] = _fetch_latest_trump_posts()
            except Exception as exc:
                st.error(
                    "최신 게시물을 불러오지 못했습니다. 비공식 RSS 또는 Jina Reader가 일시적으로 응답하지 않을 수 있습니다."
                )
                st.caption(f"오류 정보: {exc}")

        result = st.session_state.get("latest_trump_result")
        if not result:
            return
        posts = result.get("posts") or []
        days = result.get("window_days")
        if not posts:
            st.info("최근 7일 이내 게시물을 찾지 못했습니다. 미러 피드 갱신이 늦을 수도 있습니다.")
            return
        range_label = "24시간" if days == 1 else f"{days}일"
        st.caption(f"최근 {range_label} 범위에서 {len(posts):,}건 확인 · 조회 시각 {datetime.now():%Y-%m-%d %H:%M} KST")
        for post in posts:
            published = post.get("published")
            posted_text = (
                "게시 시각 확인 불가"
                if published is None
                else published.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            )
            relevant = bool(post.get("market_relevant"))
            badge = "시장 관련" if relevant else "시장 비관련"
            topic = _topic_label(post.get("topic")) if post.get("topic") else "주제 미분류"
            ticker = _ticker_label(post.get("ticker")) if post.get("ticker") else "연결 시장 없음"
            with st.container(border=True):
                st.markdown(
                    '<div class="badge-row">'
                    f'<span class="badge source">{html.escape(badge)}</span>'
                    f'<span class="badge">{html.escape(topic)}</span>'
                    f'<span class="badge">{html.escape(ticker)}</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.caption(posted_text)
                if post.get("text"):
                    st.write(post["text"])
                else:
                    st.caption("본문을 추출하지 못했습니다. 이미지 전용·재게시이거나 미러 페이지 형식이 달라졌을 수 있습니다.")
                if post.get("link"):
                    st.markdown(f"[원문 열기 ↗]({post['link']})")


def _event_rank(events: pd.DataFrame, row: pd.Series) -> dict:
    ticker = row.get("ticker")
    tracks = events.get("track", pd.Series(index=events.index, dtype=str)).astype(str)
    levels = events.get("contamination_level", pd.Series(index=events.index, dtype=str)).astype(str)
    clean = events[tracks.eq("track1_auto") & levels.eq("CLEAN")]
    same_mask = clean.get("ticker", pd.Series(index=clean.index, dtype=str)).eq(ticker)
    baseline = pd.to_numeric(clean.loc[same_mask, "impact_score"], errors="coerce").dropna()
    score = _number(row.get("impact_score"))
    if score is None or baseline.empty:
        return {"percentile": None, "top_pct": None, "label": "비교 자료 부족", "n": int(len(baseline))}
    percentile = float((baseline < score).mean())
    top_pct = max(1, round((1 - percentile) * 100))
    if percentile >= 0.9:
        label = f"같은 시장의 과거 SNS 사건 중 상위 {top_pct}%"
    elif percentile >= 0.7:
        label = "평소보다 큰 반응"
    elif percentile >= 0.3:
        label = "평소 범위의 반응"
    else:
        label = "평소보다 작은 반응"
    return {"percentile": percentile, "top_pct": top_pct, "label": label, "n": int(len(baseline))}


def _source_scope(events: pd.DataFrame, source_choice: str) -> pd.DataFrame:
    track = events.get("track", pd.Series(index=events.index, dtype=str)).astype(str)
    if source_choice == "SNS 원문 사건":
        return events[track.eq("track1_auto")]
    if source_choice == "뉴스 사건":
        return events[track.str.startswith("track2")]
    return events


def _render_overview_cards(events: pd.DataFrame, start_date, end_date) -> None:
    track = events.get("track", pd.Series(index=events.index, dtype=str)).astype(str)
    ticker_count = events["ticker"].nunique() if "ticker" in events.columns else 0
    period_text = _compact_period(start_date, end_date)
    cards = [
        ("SNS 원문 사건", f"{int(track.eq('track1_auto').sum()):,}건", "분석 파이프라인으로 정렬된 사건"),
        ("뉴스 사건", f"{int(track.str.startswith('track2').sum()):,}건", "전체 기간에서 수집·검증한 뉴스 기반 사건"),
        ("연결 시장", f"{ticker_count:,}개", "사건과 연결된 종목·시장"),
        ("현재 분석 기간", period_text, "아래 기간 설정과 동일한 범위"),
    ]
    html_cards = []
    for index, (label, value, note) in enumerate(cards):
        value_class = "overview-value period" if index == 3 else "overview-value"
        html_cards.append(
            '<div class="overview-card">'
            f'<div class="overview-label">{html.escape(label)}</div>'
            f'<div class="{value_class}">{html.escape(value)}</div>'
            f'<div class="overview-note">{html.escape(note)}</div>'
            '</div>'
        )
    st.markdown(f'<div class="overview-grid">{"".join(html_cards)}</div>', unsafe_allow_html=True)


def _render_header(events: pd.DataFrame) -> pd.DataFrame:
    left, right = st.columns([5.5, 1.2], vertical_alignment="top")
    with left:
        st.markdown('<div class="app-kicker">MARKET EVENT EXPLORER</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-title">Who Moves the Market?</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-copy">머스크와 트럼프의 SNS·관련 사건 이후, 테슬라와 주요 시장이 평소보다 얼마나 크게 움직였는지 사건별로 확인합니다.</div>', unsafe_allow_html=True)
    with right:
        status_placeholder = st.empty()
        _render_ai_settings()
        provider, model_name = _current_llm()
        status_placeholder.markdown(
            f'<div class="status-pill">AI · {html.escape(_connection_label(provider, model_name, compact=True))}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="service-path">'
        '<div class="path-item"><b>1</b> 인물·시장·사건 선택</div>'
        '<div class="path-item"><b>2</b> 실제·시장 대비 반응 확인</div>'
        '<div class="path-item"><b>3</b> 원문·번역·감성·주가 흐름 확인</div>'
        '</div>', unsafe_allow_html=True,
    )
    if events.empty:
        return events
    dates = _parse_event_datetime(events.get("event_date"))
    valid_dates = dates.dropna()
    if valid_dates.empty:
        return events
    # 사건 발생일의 최대값이 SNS 데이터 종료일에 묶여도, 프로젝트의 분석 범위는
    # 2023-01-03 ~ 2025-10-23으로 고정합니다. 뉴스 샘플/전체 수집 시에도
    # 기간 선택기가 사라지거나 줄어들지 않게 하기 위함입니다.
    min_date, max_date = ANALYSIS_START, ANALYSIS_END

    # 기존 Streamlit session_state에 2025-04-14 같은 오래된 종료일이 남아 있을 수 있어
    # 버전이 붙은 새 키를 사용해 한 번 전체 범위로 초기화합니다.
    state_key = "global_event_date_range_v3"
    state_range = st.session_state.get(state_key)
    if not isinstance(state_range, (tuple, list)) or len(state_range) == 0:
        st.session_state[state_key] = (min_date, max_date)
        state_range = st.session_state[state_key]
    elif len(state_range) == 2:
        state_start = pd.to_datetime(state_range[0], errors="coerce")
        state_end = pd.to_datetime(state_range[1], errors="coerce")
        if pd.isna(state_start) or pd.isna(state_end):
            st.session_state[state_key] = (min_date, max_date)
        else:
            clamped_start = max(min_date, state_start.date())
            clamped_end = min(max_date, state_end.date())
            if clamped_start > clamped_end:
                clamped_start, clamped_end = min_date, max_date
            st.session_state[state_key] = (clamped_start, clamped_end)
        state_range = st.session_state[state_key]

    current_range = st.session_state[state_key]
    current_start = current_range[0]
    current_end = current_range[-1]
    _render_overview_cards(events, current_start, current_end)
    st.markdown('<div class="range-heading">분석 기간 설정</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="range-copy">전체 데이터 범위는 {min_date:%Y-%m-%d} ~ {max_date:%Y-%m-%d}입니다. '
        '아래에서 실제로 분석할 기간만 선택하면 시장 반응·사건 찾기·데이터 질문에 함께 적용됩니다.</div>',
        unsafe_allow_html=True,
    )
    selected_range = st.date_input(
        "현재 분석 기간",
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD",
        key=state_key,
        label_visibility="collapsed",
        help="선택한 기간은 시장 반응·사건 찾기·데이터 질문에 함께 적용됩니다. 평소 대비 비교 기준은 전체 과거 SNS 사건을 유지합니다.",
    )
    if not isinstance(selected_range, (tuple, list)) or len(selected_range) != 2:
        st.caption("시작일과 종료일을 모두 선택하면 조회 범위가 적용됩니다.")
        return events
    start_date, end_date = selected_range
    filtered = events[dates.dt.date.between(start_date, end_date)].copy()
    st.caption(f"현재 분석 범위 · {start_date:%Y-%m-%d} ~ {end_date:%Y-%m-%d} · 사건 {len(filtered):,}건")
    return filtered


def _event_original_text(row: pd.Series) -> str:
    if _is_news_track(row.get("track")):
        return _safe_text(row.get("description"), row.get("text_clean"), fallback="사건 설명 없음")
    return _safe_text(row.get("cluster_text_raw"), row.get("text_raw"), row.get("cluster_text_clean"), row.get("text_clean"), fallback="원문 없음")


def _automatic_summary(row: pd.Series, rank: dict) -> str:
    actual_number = _number(row.get("stock_return"), 0) or 0
    direction = "상승" if actual_number > 0 else "하락" if actual_number < 0 else "보합"
    source_note = (
        " 이 사건은 SNS 원문과 별도로 뉴스 보도를 통해 추가한 사례입니다."
        if _is_news_track(row.get("track"))
        else ""
    )
    return (
        f"{_person_label(row.get('person'))}의 {_topic_label(row.get('topic'))} 관련 사건과 연결된 거래일에 "
        f"{_ticker_label(row.get('ticker'))}는 {_format_pct(row.get('stock_return'))} {direction}했고, "
        f"시장 전체 움직임을 제외한 차이는 {_format_pct(row.get('abnormal_return'))}였습니다. "
        f"반응 크기는 {rank['label']}입니다.{source_note}"
    )


def _extractive_text_summary(text: str, limit: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "요약할 원문이 없습니다."
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    summary = " ".join(sentences[:2]) if sentences else cleaned
    return summary if len(summary) <= limit else summary[: limit - 1].rstrip() + "…"


def _translation_result_is_error(value: str | None) -> bool:
    if not value:
        return True
    normalized = re.sub(r"\s+", " ", str(value)).strip().lower()
    error_markers = (
        "error 500",
        "server error",
        "that's an error",
        "that’s an error",
        "please try again later",
        "service unavailable",
        "too many requests",
        "http error",
        "번역 실패:",
    )
    return any(marker in normalized for marker in error_markers)


def _google_translate_fallback(text: str) -> str:
    response = requests.get(
        "https://translate.googleapis.com/translate_a/single",
        params={"client": "gtx", "sl": "auto", "tl": "ko", "dt": "t", "q": text[:4000]},
        timeout=12,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    payload = response.json()
    parts = []
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        for item in payload[0]:
            if isinstance(item, list) and item and isinstance(item[0], str):
                parts.append(item[0])
    return "".join(parts).strip()


@st.cache_data(show_spinner=False, ttl=1800)
def _cached_translation(
    text: str,
    provider: str,
    model_name: str | None,
    allow_online_translation: bool,
    config_token: str = "",
) -> dict:
    if not text or _looks_korean(text):
        return {"ok": True, "text": text, "engine": "원문이 이미 한국어입니다."}
    ai_error = ""
    if provider != "none":
        translated = translate_to_korean(text[:4000], provider=provider, model_name=model_name)
        if translated and not _translation_result_is_error(translated):
            return {"ok": True, "text": translated, "engine": PROVIDER_LABELS.get(provider, provider)}
        ai_error = str(translated or "AI 번역 결과 없음")
    if not allow_online_translation:
        return {
            "ok": False,
            "needs_consent": True,
            "text": "온라인 번역 동의가 필요합니다.",
            "engine": "",
        }
    errors = []
    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="auto", target="ko").translate(text[:4000])
        if translated and not _translation_result_is_error(translated):
            return {"ok": True, "text": translated, "engine": "기본 번역"}
        errors.append("기본 번역 서비스가 오류 페이지를 반환했습니다.")
    except Exception as exc:
        errors.append(f"기본 번역: {exc}")
    try:
        translated = _google_translate_fallback(text)
        if translated and not _translation_result_is_error(translated):
            return {"ok": True, "text": translated, "engine": "기본 번역(보조 경로)"}
        errors.append("보조 번역 경로에서도 정상적인 번역문을 받지 못했습니다.")
    except Exception as exc:
        errors.append(f"보조 번역: {exc}")
    if ai_error:
        errors.insert(0, f"AI 번역: {ai_error[:180]}")
    return {"ok": False, "text": " | ".join(errors)[:600], "engine": ""}


@st.cache_data(show_spinner=False)
def _cached_event_summary(event_id: str, row_dict: dict, baseline_n: int, mean_abs_ar, percentile, original_text: str, provider: str, model_name: str | None, config_token: str = "") -> str:
    if provider == "none":
        return _extractive_text_summary(original_text)
    commentary = generate_event_commentary(
        row_dict,
        {"n": baseline_n, "mean_abs_ar": "-" if mean_abs_ar is None else f"{mean_abs_ar:.2%}", "percentile": percentile, "topic_mean_abs_ar": None},
        content_text=original_text,
        provider=provider,
        model_name=model_name,
    )
    if not commentary or commentary.startswith("판단보류:"):
        return _extractive_text_summary(original_text)
    if "분석:" in commentary:
        before, _, after = commentary.partition("분석:")
        return before.replace("요약:", "").strip() + "\n\n" + after.strip()
    return commentary


def _render_price_chart(ticker: str, event_date, daily_prices: pd.DataFrame, key: str) -> None:
    event_date = _parse_event_datetime(event_date)
    if pd.isna(event_date) or daily_prices.empty:
        return
    mask = daily_prices.get("ticker", pd.Series(index=daily_prices.index, dtype=str)).eq(ticker)
    prices = daily_prices[mask].copy()
    prices["date"] = pd.to_datetime(prices.get("date"), errors="coerce")
    prices = prices.dropna(subset=["date", "close"]).sort_values("date")
    window = prices[(prices["date"] >= event_date - pd.Timedelta(days=15)) & (prices["date"] <= event_date + pd.Timedelta(days=15))].dropna(subset=["open", "high", "low", "close"])
    if window.empty:
        return
    window["ma5"] = window["close"].rolling(5).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=window["date"], open=window["open"], high=window["high"], low=window["low"], close=window["close"], increasing_line_color="#ef4444", increasing_fillcolor="#ef4444", decreasing_line_color="#3b82f6", decreasing_fillcolor="#3b82f6", name=_ticker_label(ticker)))
    fig.add_trace(go.Scatter(x=window["date"], y=window["ma5"], mode="lines", line=dict(color="#64748b", width=1.5), name="5일 평균"))
    fig.add_vline(x=event_date, line=dict(color="#df3e52", width=1.6, dash="dot"))
    fig.add_annotation(x=event_date, y=1, yref="paper", text="사건 거래일", showarrow=False, yshift=10, font=dict(color="#c62e43", size=11))
    fig.update_layout(height=355, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False, showlegend=False, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", yaxis_title="가격($)")
    fig.update_xaxes(gridcolor="#edf0f4")
    fig.update_yaxes(gridcolor="#edf0f4")
    st.plotly_chart(fig, width="stretch", key=key)
    st.caption("빨간 봉은 상승, 파란 봉은 하락이며 점선은 사건과 연결된 거래일입니다.")


def _render_event_kpis(row: pd.Series, rank: dict) -> None:
    actual_number = _number(row.get("stock_return"))
    abnormal_number = _number(row.get("abnormal_return"))
    actual_class = " positive" if actual_number is not None and actual_number > 0 else " negative" if actual_number is not None and actual_number < 0 else ""
    abnormal_class = " positive" if abnormal_number is not None and abnormal_number > 0 else " negative" if abnormal_number is not None and abnormal_number < 0 else ""

    percentile = rank.get("percentile")
    if percentile is None:
        rank_value = "비교 자료 부족"
        rank_note = "동일 시장 과거 SNS 기준"
    else:
        top_pct = max(1, round((1 - float(percentile)) * 100))
        if percentile >= 0.9:
            rank_value = f"상위 {top_pct}%"
        elif percentile >= 0.7:
            rank_value = "큰 반응"
        elif percentile >= 0.3:
            rank_value = "평소 범위"
        else:
            rank_value = "작은 반응"
        rank_note = rank.get("label") or "동일 시장 과거 SNS 기준"

    sentiment = _sentiment_display(row)
    confidence = _number(row.get("sentiment_confidence"))
    if sentiment in {"긍정", "중립", "부정"} and confidence is not None:
        sentiment_value = f"{sentiment} {confidence * 100:.0f}%"
    else:
        sentiment_value = sentiment

    if _is_news_track(row.get("track")):
        article_count = _number(row.get("related_article_count"))
        source_count = _number(row.get("related_source_count"))
        news_value = "검증 사례" if article_count is None else f"{int(article_count):,}건"
        news_note = "뉴스 감성분석 미적용" if source_count is None else f"{int(source_count):,}개 매체에서 확인"
        fourth = ("관련 보도", news_value, news_note, "")
    else:
        fourth = ("게시물 분위기", sentiment_value, _sentiment_note(row, sentiment), "")

    cards = [
        ("해당일 실제 등락률", _format_pct(row.get("stock_return")), "사건과 연결된 거래일 종가 기준", actual_class),
        ("시장 대비 등락률", _format_pct(row.get("abnormal_return")), "같은 날 시장 움직임을 제외한 값", abnormal_class),
        ("평소 대비 반응", rank_value, rank_note, ""),
        fourth,
    ]
    blocks = []
    for label, value, note, value_class in cards:
        blocks.append(
            '<div class="event-kpi">'
            f'<div class="event-kpi-label">{html.escape(label)}</div>'
            f'<div class="event-kpi-value{value_class}">{html.escape(value)}</div>'
            f'<div class="event-kpi-note">{html.escape(note)}</div>'
            '</div>'
        )
    st.markdown(f'<div class="event-kpi-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)


def _selection_points(selection_result) -> list[dict]:
    if selection_result is None:
        return []
    selection = getattr(selection_result, "selection", None)
    if selection is None and isinstance(selection_result, dict):
        selection = selection_result.get("selection")
    if selection is None:
        return []
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    return list(points or [])


def _render_event_timeline(events: pd.DataFrame, daily_prices: pd.DataFrame, source_choice: str) -> str | None:
    """Render the original full-width price/event timeline with clearer controls.

    Events are grouped by trading date and person so dense datasets remain usable.
    The selected date and event chooser are rendered below the chart, never beside it.
    """
    if events.empty or daily_prices.empty:
        return None

    available_tickers = [
        ticker for ticker in ("TSLA", "QQQ", "SPY")
        if events.get("ticker", pd.Series(index=events.index, dtype=str)).astype(str).eq(ticker).any()
    ]
    if not available_tickers:
        available_tickers = sorted(events.get("ticker", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    if not available_tickers:
        return None

    st.markdown('<div class="section-title">주가와 사건 타임라인</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">선은 종가 흐름, 점은 사건이 연결된 거래일입니다. '
        '점을 누르면 선택한 날짜와 사건 목록이 그래프 아래에 나타나고, 이어서 상세 분석을 확인할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )
    source_token = "news" if source_choice == "뉴스 사건" else "sns"
    filter_cols = st.columns([1.25, 1.25, 2.5])
    with filter_cols[0]:
        ticker = st.selectbox(
            "관련 시장",
            available_tickers,
            format_func=_ticker_label,
            key=f"timeline_ticker_{source_token}",
        )
    people_in_scope = events.get("person", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    person_options = ["전체"] + [code for code in ("Musk", "Trump") if code in people_in_scope]
    with filter_cols[1]:
        person_choice = st.selectbox(
            "인물",
            person_options,
            format_func=lambda value: "전체" if value == "전체" else _person_label(value),
            key=f"timeline_person_{source_token}",
        )
    with filter_cols[2]:
        st.markdown(
            '<div class="timeline-guide"><b>사용 방법</b><br>'
            '점을 클릭하면 아래에서 같은 날짜의 사건을 고를 수 있습니다. '
            '보라색은 머스크, 주황색은 트럼프입니다.</div>',
            unsafe_allow_html=True,
        )

    price = daily_prices[daily_prices.get("ticker", pd.Series(index=daily_prices.index, dtype=str)).astype(str).eq(ticker)].copy()
    price["date"] = pd.to_datetime(price.get("date"), errors="coerce")
    price["close"] = pd.to_numeric(price.get("close"), errors="coerce")
    price = price.dropna(subset=["date", "close"]).sort_values("date")
    if price.empty:
        st.info("선택한 시장의 가격 데이터가 없습니다.")
        return None

    range_state = st.session_state.get("global_event_date_range_v3")
    if isinstance(range_state, (tuple, list)) and len(range_state) == 2:
        range_start = pd.Timestamp(range_state[0])
        range_end = pd.Timestamp(range_state[1])
    else:
        event_dates = _parse_event_datetime(events.get("event_date")).dropna()
        range_start = event_dates.min() if not event_dates.empty else price["date"].min()
        range_end = event_dates.max() if not event_dates.empty else price["date"].max()
    chart_price = price[price["date"].between(range_start, range_end)].copy()
    if chart_price.empty:
        st.info("현재 분석 기간에는 선택한 시장의 가격 데이터가 없습니다.")
        return None

    ticker_events = events[
        events.get("ticker", pd.Series(index=events.index, dtype=str)).astype(str).eq(ticker)
    ].copy()
    if person_choice != "전체":
        ticker_events = ticker_events[
            ticker_events.get("person", pd.Series(index=ticker_events.index, dtype=str)).astype(str).eq(person_choice)
        ]
    ticker_events["_event_date"] = _parse_event_datetime(ticker_events.get("event_date")).dt.normalize()
    ticker_events = ticker_events.dropna(subset=["_event_date"])
    if ticker_events.empty:
        st.info("현재 분석 기간에 이 시장과 연결된 사건이 없습니다.")
        return None

    chart_price["date"] = chart_price["date"].dt.normalize()
    price_lookup = chart_price.drop_duplicates("date").set_index("date")["close"]
    day_rows = []
    group_columns = ["_event_date", "person"] if "person" in ticker_events.columns else ["_event_date"]
    for group_key, group in ticker_events.groupby(group_columns, sort=True, dropna=False):
        if isinstance(group_key, tuple):
            event_date, person_code = group_key
        else:
            event_date, person_code = group_key, "Unknown"
        if event_date not in price_lookup.index:
            continue
        topics = " · ".join(
            dict.fromkeys(_topic_label(value) for value in group.get("topic", pd.Series(dtype=str)).tolist())
        )
        day_rows.append(
            {
                "date": event_date,
                "close": float(price_lookup.loc[event_date]),
                "count": int(len(group)),
                "person": str(person_code),
                "person_label": _person_label(person_code),
                "topics": topics or "주제 미확인",
            }
        )
    day_frame = pd.DataFrame(day_rows)
    if day_frame.empty:
        st.info("사건 날짜와 가격 데이터를 연결할 수 없습니다.")
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_price["date"],
            y=chart_price["close"],
            mode="lines",
            name="종가",
            line=dict(color="#26364f", width=2.25),
            fill="tozeroy",
            fillcolor="rgba(38,54,79,.055)",
            hovertemplate="%{x|%Y-%m-%d}<br>종가 $%{y:.2f}<extra></extra>",
        )
    )
    marker_styles = {
        "Musk": {"label": "일론 머스크", "color": "#7c3aed", "symbol": "circle"},
        "Trump": {"label": "도널드 트럼프", "color": "#f97316", "symbol": "diamond"},
        "Unknown": {"label": "인물 미확인", "color": "#64748b", "symbol": "circle"},
    }
    for person_code in day_frame["person"].dropna().astype(str).unique():
        subset = day_frame[day_frame["person"].astype(str).eq(person_code)].copy()
        if subset.empty:
            continue
        style = marker_styles.get(person_code, marker_styles["Unknown"])
        # If both people have an event on the same day, a small horizontal offset
        # keeps both markers clickable while customdata retains the actual date.
        hour_offset = -4 if person_code == "Musk" else 4 if person_code == "Trump" else 0
        marker_dates = subset["date"] + pd.to_timedelta(hour_offset, unit="h")
        marker_sizes = [min(15, 7.5 + 1.5 * (max(1, count) ** 0.5)) for count in subset["count"]]
        customdata = [
            [row["date"].strftime("%Y-%m-%d"), row["count"], row["person_label"], row["topics"]]
            for _, row in subset.iterrows()
        ]
        fig.add_trace(
            go.Scatter(
                x=marker_dates,
                y=subset["close"],
                mode="markers",
                name=style["label"],
                marker=dict(
                    size=marker_sizes,
                    color=style["color"],
                    symbol=style["symbol"],
                    line=dict(color="#ffffff", width=1.5),
                    opacity=.88,
                ),
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[2]} · 연결 사건 %{customdata[1]}건<br>"
                    "%{customdata[3]}<br>종가 $%{y:.2f}<br><b>클릭해 사건 보기</b><extra></extra>"
                ),
            )
        )
    fig.update_layout(
        height=440,
        margin=dict(l=8, r=8, t=38, b=8),
        showlegend=True,
        hovermode="closest",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        yaxis_title="종가($)",
        clickmode="event+select",
        selectionrevision=f"{source_choice}_{ticker}_{person_choice}",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=11), itemclick=False, itemdoubleclick=False,
        ),
    )
    fig.update_xaxes(gridcolor="#edf0f4", showgrid=True, showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(gridcolor="#edf0f4", zeroline=False, rangemode="normal")

    date_key = f"timeline_selected_date_{source_token}_{ticker}_{person_choice}"
    selection_result = st.plotly_chart(
        fig,
        width="stretch",
        key=f"timeline_chart_{source_token}_{ticker}_{person_choice}",
        on_select="rerun",
        selection_mode="points",
        config={"displayModeBar": False, "scrollZoom": False},
    )
    for point in _selection_points(selection_result):
        custom = point.get("customdata") if isinstance(point, dict) else None
        if isinstance(custom, (list, tuple)) and custom:
            clicked = pd.to_datetime(custom[0], errors="coerce")
            if pd.notna(clicked):
                st.session_state[date_key] = clicked.strftime("%Y-%m-%d")
                break

    selected_date_text = st.session_state.get(date_key)
    if not selected_date_text:
        st.markdown(
            '<div class="timeline-selection empty"><b>사건 점을 선택해주세요.</b>'
            '<span>선 위의 점을 클릭하면 해당 날짜의 사건 목록이 이 영역에 표시됩니다.</span></div>',
            unsafe_allow_html=True,
        )
        return None

    selected_date = pd.to_datetime(selected_date_text, errors="coerce")
    day_events = ticker_events[ticker_events["_event_date"].eq(selected_date)].copy()
    if day_events.empty:
        st.session_state.pop(date_key, None)
        st.info("선택한 날짜의 사건을 찾지 못했습니다. 다른 점을 선택해주세요.")
        return None

    close_match = chart_price[chart_price["date"].eq(selected_date)]
    close_text = "-" if close_match.empty else f"${close_match.iloc[0]['close']:.2f}"
    st.markdown(
        '<div class="timeline-selection">'
        '<div>'
        '<div class="timeline-selection-label">선택한 사건 날짜</div>'
        f'<div class="timeline-selection-date">{selected_date:%Y-%m-%d}</div>'
        '</div>'
        '<div class="timeline-selection-stats">'
        f'<span><b>{html.escape(_ticker_label(ticker))}</b><small>시장</small></span>'
        f'<span><b>{close_text}</b><small>해당일 종가</small></span>'
        f'<span><b>{len(day_events):,}건</b><small>연결 사건</small></span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    day_display = _decorate_events(day_events).sort_values("반응 강도", ascending=False, na_position="last")
    day_ids = day_display["event_id"].astype(str).tolist()
    labels = {
        str(row["event_id"]): f"{row['인물']} · {row['주제']} · {row['내용']}"
        for _, row in day_display.iterrows()
    }
    st.markdown(
        '<div class="timeline-picker-head">'
        '<div class="timeline-picker-copy">'
        '<div class="timeline-picker-kicker">SELECT EVENT</div>'
        '<div class="timeline-picker-title">이 날짜에서 자세히 볼 사건</div>'
        '<div class="timeline-picker-help">사건을 선택하면 바로 아래 상세 분석 카드가 해당 사건으로 바뀝니다.</div>'
        '</div>'
        f'<div class="timeline-picker-count">{len(day_events):,}건</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    selected_event_id = st.selectbox(
        "상세 분석할 사건 선택",
        day_ids,
        format_func=lambda value: labels.get(str(value), str(value)),
        key=f"timeline_event_{source_token}_{ticker}_{person_choice}_{selected_date:%Y%m%d}",
        help="같은 날짜에 사건이 여러 건이면 반응 강도가 큰 사건부터 표시됩니다.",
    )
    selected_preview = day_display[day_display["event_id"].astype(str).eq(str(selected_event_id))].iloc[0]
    st.markdown(
        '<div class="timeline-picked-event">'
        '<span class="picked-label">현재 선택</span>'
        f'<span><b>{html.escape(str(selected_preview["인물"]))}</b></span>'
        f'<span>{html.escape(str(selected_preview["주제"]))}</span>'
        f'<span>실제 등락률 <b>{html.escape(_format_pct(selected_preview.get("stock_return")))}</b></span>'
        f'<span>시장 대비 <b>{html.escape(_format_pct(selected_preview.get("abnormal_return")))}</b></span>'
        '</div>',
        unsafe_allow_html=True,
    )
    return selected_event_id


def _render_event_detail(
    events: pd.DataFrame,
    daily_prices: pd.DataFrame,
    event_id,
    scope: str,
    baseline_events: pd.DataFrame | None = None,
) -> None:
    ids = events.get("event_id", pd.Series(index=events.index, dtype=str)).astype(str)
    selected = events[ids.eq(str(event_id))]
    if selected.empty:
        st.info("선택한 사건을 찾을 수 없습니다.")
        return
    row = selected.iloc[0]
    display = _decorate_events(events)
    drow = display[display["event_id"].astype(str).eq(str(event_id))].iloc[0]
    original = _event_original_text(row)
    is_news = _is_news_track(row.get("track"))
    reference = events if baseline_events is None else baseline_events
    rank = _event_rank(reference, row)
    source_url = _safe_text(row.get("source_url"), fallback="")
    badges = [
        f'<span class="badge">{html.escape(_safe_text(drow.get("사건")))}</span>',
        f'<span class="badge">{html.escape(_person_label(row.get("person")))}</span>',
        f'<span class="badge">{html.escape(_ticker_label(row.get("ticker")))}</span>',
        f'<span class="badge">{html.escape(_topic_label(row.get("topic")))}</span>',
        f'<span class="badge source">{html.escape(_source_label(row.get("track")))}</span>',
        f'<span class="badge">{html.escape(_safe_text(drow.get("날짜")))} · {html.escape(_session_label(row.get("market_session")))}</span>',
    ]
    title = "뉴스로 확인한 사건" if is_news else "SNS 게시물 원문"
    if is_news:
        article_count = _number(row.get("related_article_count"))
        source_count = _number(row.get("related_source_count"))
        source_text = _safe_text(row.get("news_sources"), fallback="")
        details = []
        if article_count is not None:
            details.append(f"관련 기사 {int(article_count):,}건")
        if source_count is not None:
            details.append(f"매체 {int(source_count):,}곳")
        if source_text:
            details.append(f"주요 매체: {source_text}")
        note = "감성분석 없이 뉴스 발생 시점과 시장 반응만 연결한 사건입니다."
        if details:
            note += " " + " · ".join(details)
    else:
        note = "SNS 게시물 원문과 같은 거래일의 시장 데이터를 연결한 사건입니다."
    link_html = f'<div style="margin-top:10px"><a href="{html.escape(source_url)}" target="_blank" style="color:#245fbd;font-weight:720">근거 자료 열기 ↗</a></div>' if source_url else ""
    st.markdown('<div class="detail-card">' f'<div class="badge-row">{"".join(badges)}</div>' f'<div class="detail-title">{html.escape(title)}</div>' f'<div class="detail-text">{html.escape(original)}</div>' f'{link_html}<div class="note-box">{html.escape(note)}</div></div>', unsafe_allow_html=True)
    _render_event_kpis(row, rank)
    st.markdown(f'<div class="summary-box"><b>데이터 요약</b><br>{html.escape(_automatic_summary(row, rank))}</div>', unsafe_allow_html=True)
    provider, model_name = _current_llm()
    config_token = st.session_state.get("active_llm_config_token") or _llm_config_token(provider, model_name)
    config_hash = hashlib.sha256(config_token.encode("utf-8")).hexdigest()[:10]
    translation_key = f"translation_{scope}_{event_id}_{provider}_{model_name}_{config_hash}"
    summary_key = f"summary_{scope}_{event_id}_{provider}_{model_name}_{config_hash}"
    original_is_korean = _looks_korean(original)
    action_cols = st.columns(2)
    if not original_is_korean:
        with action_cols[0]:
            if st.button("원문 한국어로 번역", key=f"translate_btn_{scope}_{event_id}", width="stretch"):
                with st.spinner("한국어 번역을 준비하는 중..."):
                    st.session_state[translation_key] = _cached_translation(
                        original,
                        provider,
                        model_name,
                        bool(st.session_state.get("allow_online_translation", False)),
                        config_token,
                    )
    else:
        with action_cols[0]:
            st.markdown(
                '<div class="note-box" style="margin-top:0;height:100%;">현재 위에 표시된 내용이 이미 한국어라 별도 번역이 필요하지 않습니다.</div>',
                unsafe_allow_html=True,
            )
    with action_cols[1]:
        if st.button("사건 요약 생성", key=f"summary_btn_{scope}_{event_id}", width="stretch", type="primary"):
            ref_tracks = reference.get("track", pd.Series(index=reference.index, dtype=str)).astype(str)
            ref_levels = reference.get("contamination_level", pd.Series(index=reference.index, dtype=str)).astype(str)
            clean = reference[ref_tracks.eq("track1_auto") & ref_levels.eq("CLEAN")]
            same = clean[clean.get("ticker", pd.Series(index=clean.index, dtype=str)).eq(row.get("ticker"))]
            mean_abs = pd.to_numeric(same.get("abnormal_return"), errors="coerce").abs().mean()
            row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            with st.spinner("사건 내용을 정리하는 중..."):
                st.session_state[summary_key] = _cached_event_summary(str(event_id), row_dict, len(same), None if pd.isna(mean_abs) else float(mean_abs), rank["percentile"], original, provider, model_name, config_token)
    translated = st.session_state.get(translation_key)
    if translated and not original_is_korean:
        if translated.get("ok"):
            with st.expander(f"한국어 번역 · {translated.get('engine')}", expanded=True):
                st.write(translated.get("text"))
        else:
            if translated.get("needs_consent"):
                st.warning("상단의 AI 설정에서 AI 서비스를 연결하거나 ‘기본 온라인 번역 허용’을 선택해주세요.")
            else:
                st.warning("번역 서비스에 연결하지 못했습니다. AI 설정이나 인터넷 연결을 확인해주세요.")
    generated_summary = st.session_state.get(summary_key)
    if generated_summary:
        with st.expander("사건 요약", expanded=True):
            st.write(generated_summary)
    st.markdown('<div class="section-title">사건 전후 주가 흐름</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">사건 거래일을 중심으로 앞뒤 15일의 가격을 보여줍니다.</div>', unsafe_allow_html=True)
    _render_price_chart(str(row.get("ticker")), row.get("event_date"), daily_prices, key=f"price_{scope}_{event_id}")


def _render_sentiment_snapshot(events: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">게시물 분위기별 빠른 보기</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">문장의 어조를 긍정·중립·부정으로 나누고, 각 그룹에서 시장 대비 움직임의 중앙값을 함께 보여줍니다.</div>', unsafe_allow_html=True)
    if "sentiment_label" not in events.columns or events["sentiment_label"].dropna().empty:
        st.info("현재 결과 파일에는 게시물 분위기 정보가 없습니다. 감성분석을 포함해 분석을 다시 실행하면 이 영역과 사건 카드에 표시됩니다.")
        return
    work = events.dropna(subset=["sentiment_label"]).copy()
    work["감성"] = work["sentiment_label"].map(_sentiment_label)
    work["_abs"] = pd.to_numeric(work.get("abnormal_return"), errors="coerce").abs()
    cols = st.columns(3)
    for col, label in zip(cols, ["긍정", "중립", "부정"]):
        subset = work[work["감성"].eq(label)]
        median = subset["_abs"].median()
        value = "-" if pd.isna(median) else f"{median * 100:.2f}%"
        with col:
            st.markdown('<div class="sentiment-card">' f'<div class="sentiment-label">{label} 게시물 · {len(subset):,}건</div>' f'<div class="sentiment-value">{value}</div>' '<div class="sentiment-note">시장 대비 움직임 절대값의 중앙값</div></div>', unsafe_allow_html=True)


def render_market_view(events: pd.DataFrame, daily_prices: pd.DataFrame, baseline_events: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">시장 반응 탐색</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">먼저 자료 형태를 고른 뒤 주가 타임라인의 사건 점을 클릭하거나, 아래 최근 사건 카드에서 바로 사건을 선택할 수 있습니다.</div>', unsafe_allow_html=True)
    # 자료 선택지는 현재 날짜 필터 결과가 아니라 전체 데이터 기준으로 유지합니다.
    # 예: 2025-06 뉴스만 있는데 현재 기간이 2025-04까지라 하더라도
    # '뉴스 사건' 선택지가 통째로 사라지지 않고, 선택 시 해당 기간에 데이터가 없음을 안내합니다.
    all_tracks = baseline_events.get("track", pd.Series(index=baseline_events.index, dtype=str)).astype(str)
    source_options = []
    if all_tracks.eq("track1_auto").any():
        source_options.append("SNS 원문 사건")
    if all_tracks.str.startswith("track2").any():
        source_options.append("뉴스 사건")
    if not source_options:
        source_options.append("전체 사건")
    source_choice = st.radio("자료 형태", source_options, horizontal=True, key="home_source_choice")
    scoped = _source_scope(events, source_choice)
    if scoped.empty:
        if source_choice == "뉴스 사건":
            st.info("현재 선택한 분석 기간에는 연결 가능한 뉴스 사건이 없습니다. 분석 기간을 넓히거나 뉴스 수집/가격 범위를 확인해 주세요.")
        else:
            st.info("현재 선택한 분석 기간에는 해당 자료 형태의 사건이 없습니다.")
        return
    st.markdown('<div class="section-title">최근 사건의 시장 반응</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">계기판에는 사건 원문을 길게 넣지 않고, 시장별 최신 사건의 반응 크기와 핵심 정보만 보여줍니다. 전체 내용은 아래 상세 카드에서 확인할 수 있습니다.</div>', unsafe_allow_html=True)
    gauges = compute_ticker_gauges(
        scoped,
        tickers=("QQQ", "SPY", "TSLA"),
        baseline_events=baseline_events,
    )
    first_event = next((state.get("event_id") for state in gauges if state.get("has_data")), None)
    scoped_ids = set(scoped.get("event_id", pd.Series(index=scoped.index, dtype=str)).astype(str))
    current_selection = st.session_state.get("home_selected_event")
    if st.session_state.get("home_selected_source") != source_choice or str(current_selection) not in scoped_ids:
        st.session_state["home_selected_source"] = source_choice
        st.session_state["home_selected_event"] = first_event
    columns = st.columns(3)
    selected_event = None
    for index, (column, state) in enumerate(zip(columns, gauges)):
        with column:
            st.markdown(render_single_gauge_html(state), unsafe_allow_html=True)
            if state.get("has_data") and st.button("이 사건 자세히 보기", key=f"home_gauge_detail_{source_choice}_{index}", width="stretch"):
                selected_event = state.get("event_id")
    if selected_event is not None:
        st.session_state["home_selected_event"] = selected_event
        for key in list(st.session_state):
            if str(key).startswith("timeline_selected_date_"):
                st.session_state.pop(key, None)
    if "home_selected_event" not in st.session_state:
        if first_event is not None:
            st.session_state["home_selected_event"] = first_event

    timeline_selected = _render_event_timeline(scoped, daily_prices, source_choice)
    if timeline_selected is not None:
        st.session_state["home_selected_event"] = timeline_selected
    selected = st.session_state.get("home_selected_event")
    if selected is not None:
        st.markdown('<div class="section-title">선택한 사건 상세</div>', unsafe_allow_html=True)
        _render_event_detail(events, daily_prices, selected, scope="home", baseline_events=baseline_events)
    sns = events[events.get("track", pd.Series(index=events.index, dtype=str)).astype(str).eq("track1_auto")]
    _render_sentiment_snapshot(sns)
    _render_latest_trump_posts()


def render_event_finder(events: pd.DataFrame, daily_prices: pd.DataFrame, baseline_events: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">사건 찾기</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">인물·시장·자료 형태로 사건을 좁혀보세요. 뉴스 사건은 감성분석 없이 시장 반응만 연결합니다.</div>', unsafe_allow_html=True)
    display = _decorate_events(events)
    filters = st.columns(4)
    with filters[0]:
        person = st.selectbox("인물", ["전체"] + sorted(display["인물"].dropna().unique().tolist()), key="finder_person")
    with filters[1]:
        ticker = st.selectbox("시장", ["전체"] + sorted(display["시장"].dropna().unique().tolist()), key="finder_ticker")
    with filters[2]:
        source = st.selectbox("자료 형태", ["전체"] + sorted(display["자료 형태"].dropna().unique().tolist()), key="finder_source")
    with filters[3]:
        sentiment_values = [value for value in ["긍정", "중립", "부정"] if value in set(display["감성"].dropna())]
        sentiment = st.selectbox("게시물 분위기", ["전체"] + sentiment_values, key="finder_sentiment")
    search_cols = st.columns([2, 1])
    with search_cols[0]:
        keyword = st.text_input("내용 검색", placeholder="예: 관세, 중국, 인공지능", key="finder_keyword")
    with search_cols[1]:
        sort = st.selectbox("정렬", ["최근 사건부터", "시장 대비 움직임이 큰 순", "평소 대비 반응이 큰 순"], key="finder_sort")
    filtered = display.copy()
    if person != "전체": filtered = filtered[filtered["인물"].eq(person)]
    if ticker != "전체": filtered = filtered[filtered["시장"].eq(ticker)]
    if source != "전체": filtered = filtered[filtered["자료 형태"].eq(source)]
    if sentiment != "전체": filtered = filtered[filtered["감성"].eq(sentiment)]
    if keyword.strip():
        searchable = filtered["내용"].fillna("") + " " + filtered["주제"].fillna("")
        filtered = filtered[searchable.str.contains(keyword.strip(), case=False, regex=False, na=False)]
    if sort == "시장 대비 움직임이 큰 순":
        filtered = filtered.assign(_sort=filtered["시장 대비 등락률"].abs()).sort_values("_sort", ascending=False)
    elif sort == "평소 대비 반응이 큰 순":
        filtered = filtered.sort_values("반응 강도", ascending=False)
    else:
        filtered = filtered.sort_values("_date", ascending=False)
    if filtered.empty:
        st.info("선택한 조건에 맞는 사건이 없습니다.")
        return
    metric_cols = st.columns(3)
    metric_cols[0].metric("찾은 사건", f"{len(filtered):,}건")
    mean_abs = filtered["시장 대비 등락률"].abs().mean()
    metric_cols[1].metric("평균 시장 대비 움직임", "-" if pd.isna(mean_abs) else f"{mean_abs * 100:.2f}%")
    metric_cols[2].metric("감성 정보가 있는 사건", f"{int(filtered['감성'].isin(['긍정', '중립', '부정']).sum()):,}건")
    table = filtered[["사건", "날짜", "인물", "시장", "주제", "자료 형태", "시장 대비 등락률", "감성", "내용"]].copy()
    table["시장 대비 등락률"] = (table["시장 대비 등락률"] * 100).round(2)
    table = table.rename(columns={"시장 대비 등락률": "시장 대비 등락률(%)"})
    st.dataframe(table, width="stretch", height=390, hide_index=True, column_config={"시장 대비 등락률(%)": st.column_config.NumberColumn(format="%+.2f"), "내용": st.column_config.TextColumn(width="large")})
    ids = filtered["event_id"].astype(str).tolist()
    labels = {str(row["event_id"]): f"{row['사건']} · {row['날짜']} · {row['인물']} · {row['시장']} · {row['주제']}" for _, row in filtered.iterrows()}
    selected = st.selectbox("자세히 볼 사건", ids, format_func=lambda value: labels.get(str(value), str(value)), key="finder_selected")
    if selected:
        _render_event_detail(events, daily_prices, selected, scope="finder", baseline_events=baseline_events)


def _chat_scope(events: pd.DataFrame, question: str) -> tuple[pd.DataFrame, list[str]]:
    work, filters, q = events.copy(), [], question.lower()
    people = {"Musk": ["musk", "머스크", "일론"], "Trump": ["trump", "트럼프", "도널드"]}
    selected_people = [code for code, terms in people.items() if any(term in q for term in terms)]
    if selected_people and "person" in work.columns:
        work = work[work["person"].astype(str).isin(selected_people)]
        filters.append("인물: " + ", ".join(_person_label(code) for code in selected_people))
    tickers = {"TSLA": ["tsla", "테슬라"], "QQQ": ["qqq", "나스닥"], "SPY": ["spy", "s&p", "에스앤피"]}
    selected_tickers = [code for code, terms in tickers.items() if any(term in q for term in terms)]
    if selected_tickers and "ticker" in work.columns:
        work = work[work["ticker"].astype(str).str.upper().isin(selected_tickers)]
        filters.append("시장: " + ", ".join(_ticker_label(code) for code in selected_tickers))
    if "뉴스" in q:
        work = work[work.get("track", pd.Series(index=work.index, dtype=str)).astype(str).str.startswith("track2")]
        filters.append("자료: 뉴스 사건")
    elif "sns" in q or "게시물" in q or "원문" in q:
        work = work[work.get("track", pd.Series(index=work.index, dtype=str)).astype(str).eq("track1_auto")]
        filters.append("자료: SNS 원문 사건")
    for korean, raw in {"긍정": "positive", "부정": "negative", "중립": "neutral"}.items():
        if korean in q and "sentiment_label" in work.columns:
            work = work[work["sentiment_label"].astype(str).str.lower().str.contains(raw, na=False)]
            filters.append(f"분위기: {korean}")
            break
    if "topic" in work.columns:
        for topic in work["topic"].dropna().astype(str).unique():
            if topic.lower() in q or _topic_label(topic).lower() in q:
                work = work[work["topic"].astype(str).eq(topic)]
                filters.append(f"주제: {_topic_label(topic)}")
                break
    return work, filters


def _group_summary(events: pd.DataFrame, column: str, label_func) -> list[dict]:
    if events.empty or column not in events.columns:
        return []
    work = events.copy()
    work["_abs"] = pd.to_numeric(work.get("abnormal_return"), errors="coerce").abs()
    work["_impact"] = pd.to_numeric(work.get("impact_score"), errors="coerce")
    grouped = work.dropna(subset=[column]).groupby(column).agg(사건수=(column, "size"), 평균_시장대비움직임=("_abs", "mean"), 중앙값_시장대비움직임=("_abs", "median"), 평균_반응강도=("_impact", "mean")).reset_index()
    grouped[column] = grouped[column].map(label_func)
    return grouped.to_dict("records")


def _build_chat_context(events: pd.DataFrame, question: str, history: list) -> dict:
    scoped, filters = _chat_scope(events, question)
    display = _decorate_events(scoped)
    top_rows = []
    for _, row in display.sort_values("반응 강도", ascending=False).head(6).iterrows():
        top_rows.append({"사건": row.get("사건"), "날짜": row.get("날짜"), "인물": row.get("인물"), "시장": row.get("시장"), "주제": row.get("주제"), "시장 대비 등락률": _format_pct(row.get("시장 대비 등락률")), "반응 강도": _format_number(row.get("반응 강도"))})
    dates = _parse_event_datetime(scoped.get("event_date"))
    sentiment_rows = []
    if "sentiment_label" in scoped.columns and scoped["sentiment_label"].notna().any():
        sent = scoped.dropna(subset=["sentiment_label"]).copy()
        sent["감성"] = sent["sentiment_label"].map(_sentiment_label)
        sent["_abs"] = pd.to_numeric(sent.get("abnormal_return"), errors="coerce").abs()
        sentiment_rows = sent.groupby("감성").agg(사건수=("감성", "size"), 중앙값_시장대비움직임=("_abs", "median")).reset_index().to_dict("records")
    return {
        "적용한 조건": filters or ["전체 사건"], "사건 수": int(len(scoped)),
        "기간": {"시작": str(dates.min().date()) if dates.notna().any() else None, "종료": str(dates.max().date()) if dates.notna().any() else None},
        "인물별": _group_summary(scoped, "person", _person_label), "시장별": _group_summary(scoped, "ticker", _ticker_label),
        "감성별": sentiment_rows, "반응 상위 사건": top_rows,
        "자료 형태별 사건수": scoped.get("track", pd.Series(dtype=str)).map(_source_label).value_counts().to_dict(),
        "최근 대화": [{"역할": item.get("role"), "내용": item.get("content")} for item in history[-4:]],
    }


def _group_answer(rows: list[dict], group_key: str) -> str:
    if not rows:
        return "비교할 자료가 없습니다."
    return " / ".join(f"{row.get(group_key)} {row.get('사건수', 0):,}건: 평균 {_format_pct(row.get('평균_시장대비움직임'))}, 중앙값 {_format_pct(row.get('중앙값_시장대비움직임'))}" for row in rows)


def _deterministic_answer(question: str, context: dict) -> str:
    q, count = question.lower(), context.get("사건 수", 0)
    filters = ", ".join(context.get("적용한 조건", []))
    if any(term in q for term in ["가장", "상위", "최대", "큰 사건"]):
        rows = context.get("반응 상위 사건", [])
        if not rows:
            return f"{filters} 조건에 맞는 사건이 없습니다."
        lines = [f"{index}. {row['사건']} · {row['날짜']} · {row['인물']} · {row['시장']} · {row['주제']} (반응 강도 {row['반응 강도']})" for index, row in enumerate(rows[:5], start=1)]
        return f"{filters} 조건에서 반응 강도가 컸던 사건입니다.\n\n" + "\n".join(lines)
    if "감성" in q or "긍정" in q or "부정" in q or "분위기" in q:
        rows = context.get("감성별", [])
        if not rows:
            return "현재 결과 파일에는 감성분석 값이 없습니다. 감성분석을 포함해 파이프라인을 다시 실행하면 긍정·중립·부정별 사건 수와 시장 반응을 비교할 수 있습니다."
        parts = [f"{row['감성']} {row['사건수']:,}건의 시장 대비 움직임 중앙값은 {_format_pct(row['중앙값_시장대비움직임'])}" for row in rows]
        return f"{filters} 기준 감성별 결과입니다. " + "; ".join(parts) + "입니다. 문장의 감성과 주가 방향이 같은 의미는 아닙니다."
    if ("머스크" in q or "musk" in q) and ("트럼프" in q or "trump" in q):
        return "인물별 비교 결과입니다. " + _group_answer(context.get("인물별", []), "person") + ". 연결된 시장 구성이 다르므로 숫자만으로 인물 자체의 영향이라고 단정할 수는 없습니다."
    if any(term in q for term in ["시장 비교", "어느 시장", "어디", "세 시장", "시장별"]):
        return "시장별 비교 결과입니다. " + _group_answer(context.get("시장별", []), "ticker") + "."
    if "몇 건" in q or "개수" in q or "기간" in q:
        period = context.get("기간", {})
        return f"{filters} 조건에 해당하는 사건은 {count:,}건이며, 기간은 {period.get('시작')}부터 {period.get('종료')}까지입니다."
    if "뉴스" in q or "sns" in q or "자료" in q:
        counts = context.get("자료 형태별 사건수", {})
        detail = ", ".join(f"{name} {value:,}건" for name, value in counts.items()) or "자료 없음"
        return f"현재 조건의 자료 구성은 {detail}입니다. 뉴스 추가 사례는 SNS 원문과 별도로 뉴스 보도를 통해 추가한 사례입니다."
    if "시장 대비" in q or "초과" in q:
        return "시장 대비 등락률은 해당 종목의 실제 등락률에서 같은 날 시장 전체 움직임을 제외한 값입니다. 양수면 시장보다 더 강했고, 음수면 시장보다 더 약했다는 뜻입니다."
    if "반응 강도" in q or "계기판" in q or "평소" in q:
        return "반응 강도는 가격·거래량·변동성이 평소보다 얼마나 크게 달라졌는지 합친 비교값입니다. 계기판에서는 숫자 자체보다 ‘같은 시장의 과거 사건 중 상위 몇 %인지’를 보여줍니다."
    if any(term in q for term in ["무엇", "뭐 하는", "핵심", "사용법"]):
        return "이 서비스에서는 머스크·트럼프 관련 사건을 고른 뒤 실제 등락률, 시장 전체 움직임을 제외한 등락률, 같은 시장의 과거 대비 반응 크기, 게시물 감성, 전후 주가를 확인할 수 있습니다. 미래 주가를 예측하는 서비스는 아닙니다."
    period = context.get("기간", {})
    top = context.get("반응 상위 사건", [])
    top_text = ""
    if top:
        first = top[0]
        top_text = f" 이 조건에서 반응 강도가 가장 큰 항목은 {first['사건']}({first['날짜']} · {first['시장']} · {first['주제']})입니다."
    return (
        f"{filters} 조건으로 {count:,}건을 찾았습니다. 기간은 {period.get('시작')}부터 {period.get('종료')}까지입니다."
        f"{top_text} 더 구체적으로 ‘가장 큰 사건’, ‘시장별 비교’, ‘감성별 반응’처럼 물으면 해당 조건으로 다시 계산합니다."
    )


@st.cache_data(show_spinner=False)
def _cached_ai_answer(question: str, context: dict, provider: str, model_name: str | None, config_token: str = "") -> str:
    return answer_data_question(question, context, provider=provider, model_name=model_name)


def render_data_chat(events: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">데이터에게 질문하기</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">질문에서 인물·시장·주제·자료 형태를 찾아 실제 사건을 먼저 계산합니다. AI가 연결되지 않아도 자주 묻는 비교 질문은 동작합니다.</div>', unsafe_allow_html=True)
    provider, model_name = _current_llm()
    config_token = st.session_state.get("active_llm_config_token") or _llm_config_token(provider, model_name)
    history = st.session_state.setdefault("service_chat_history", [])
    suggestions = ["반응이 가장 컸던 사건 5개", "일론 머스크와 도널드 트럼프 비교", "세 시장을 비교해줘", "감성별 시장 반응을 알려줘"]
    columns = st.columns(4)
    pending = None
    for column, suggestion in zip(columns, suggestions):
        with column:
            if st.button(suggestion, key=f"chat_suggestion_{suggestion}", width="stretch"):
                pending = suggestion
    status_cols = st.columns([5, 1])
    with status_cols[0]:
        mode = _connection_label(provider, model_name)
        st.caption(f"현재 답변 방식: {mode}")
    with status_cols[1]:
        if st.button("대화 지우기", key="clear_service_chat", width="stretch"):
            history.clear(); st.rerun()
    for item in history[-12:]:
        with st.chat_message(item.get("role", "assistant")):
            st.write(item.get("content", ""))
    typed = st.chat_input("예: 트럼프 관세 사건 중 반응이 가장 컸던 것은?", key="service_chat_input")
    question = typed or pending
    if not question:
        if not history: st.info("위 예시 질문을 누르거나 직접 질문해보세요.")
        return
    history.append({"role": "user", "content": question})
    context = _build_chat_context(events, question, history[:-1])
    if provider == "none":
        answer = _deterministic_answer(question, context)
    else:
        with st.spinner("데이터를 계산하고 답변을 정리하는 중..."):
            answer = _cached_ai_answer(question, context, provider, model_name, config_token)
        if not answer or answer.startswith("판단보류:"):
            answer = _deterministic_answer(question, context)
    history.append({"role": "assistant", "content": answer})
    st.rerun()


def main() -> None:
    _inject_styles()
    data = cached_data()
    events = data.get("events", pd.DataFrame())
    daily_prices = data.get("daily_prices", pd.DataFrame())
    filtered_events = _render_header(events)
    if events.empty:
        st.warning("표시할 사건 데이터가 없습니다. 먼저 일봉 분석 파이프라인을 실행해주세요.")
        with st.expander("실행 방법 확인"):
            st.code(".venv\\Scripts\\python.exe run_daily_pipeline.py --add-sentiment --add-novelty --run-placebo --run-rivn-sensitivity", language="powershell")
        return
    if filtered_events.empty:
        st.warning("선택한 조회 기간에 해당하는 사건이 없습니다. 조회 기간을 넓혀주세요.")
        return
    market_tab, finder_tab, chat_tab = st.tabs(["시장 반응", "사건 찾기", "데이터 질문"])
    with market_tab: render_market_view(filtered_events, daily_prices, baseline_events=events)
    with finder_tab: render_event_finder(filtered_events, daily_prices, baseline_events=events)
    with chat_tab: render_data_chat(filtered_events)


if __name__ == "__main__":
    main()
