import html
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from market_mover.dashboard_data import (
    load_dashboard_data,
)
from market_mover.dashboard_widgets import compute_ticker_gauges, render_single_gauge_html
from market_mover.case_narratives import (
    DEFAULT_LLM_MODELS,
    generate_event_commentary,
    fetch_article_markdown,
    translate_to_korean,
    answer_data_question,
)
from market_mover.load_posts import clean_text
from market_mover.topic_rules import assign_topic, is_market_relevant, map_ticker
from live_monitor import fetch_trump_feed, fetch_post_text
from find_track2_news_candidates import build_query, fetch_google_news_rss


st.set_page_config(
    page_title="Who Moves the Market?",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8fa; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e5e7eb; }
        [data-testid="stSidebar"] * { color: #1f2937; }
        header[data-testid="stHeader"] { background:rgba(247,248,250,.92); }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px;
            padding: 14px 16px; box-shadow: 0 2px 8px rgba(15,23,42,.035);
        }
        [data-testid="stMetricLabel"] { color: #64748b; }
        .block-container { max-width: 1540px; padding-top: 1.35rem; padding-bottom: 3rem; }
        .app-title {
            color:#172033; font-size:2.35rem; font-weight:820; line-height:1.2;
            letter-spacing:-.035em; margin:.15rem 0 1.05rem 0;
        }
        .hero-title { color:#111827; font-size:1.9rem; font-weight:750; line-height:1.3; margin:0 0 .35rem 0; }
        .hero-copy { color:#5f6b7a; font-size:.96rem; max-width:980px; margin-bottom:1.25rem; line-height:1.65; }
        .section-note { color:#64748b; font-size:.84rem; }
        div[data-testid="stExpander"] { background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; }
        .stButton > button { border-radius:10px; font-weight:700; }
        .stTabs [data-baseweb="tab-list"] {
            gap:1.35rem; border-bottom:1px solid #dfe4ea; padding:0 .15rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius:0; padding:.6rem .05rem .72rem; color:#475569;
            font-size:.9rem; font-weight:650; background:transparent;
        }
        .stTabs [aria-selected="true"] { color:#df3e52 !important; }
        .stTabs [data-baseweb="tab-highlight"] { background-color:#df3e52; height:2px; }
        [data-testid="stChatMessage"] {
            background:#ffffff; border:1px solid #e8ebef; border-radius:12px;
            padding:.3rem .55rem; margin-bottom:.45rem;
        }
        .mini-chat-title { color:#172033;font-size:1rem;font-weight:800;margin-bottom:.15rem; }
        .mini-chat-caption { color:#64748b;font-size:.78rem;line-height:1.5;margin-bottom:.65rem; }
        .intro-panel {
            background:#ffffff; border:1px solid #dfe4ea; border-left:4px solid #2563eb;
            border-radius:12px; padding:16px 18px; margin:.2rem 0 1.2rem 0;
            color:#374151; line-height:1.7;
        }
        .source-card {
            background:#ffffff; border:1px solid #dfe4ea; border-radius:14px;
            padding:18px 20px; margin:.5rem 0 1rem 0; color:#1f2937; line-height:1.7;
        }
        .source-card-title { font-size:1.05rem; font-weight:750; color:#111827; margin-bottom:.55rem; }
        .meta-row { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:12px; }
        .meta-item {
            background:#f3f4f6; color:#374151; font-size:.78rem; font-weight:650;
            padding:4px 9px; border-radius:7px;
        }
        .result-card {
            background:#ffffff; border:1px solid #e5e7eb; border-radius:12px;
            padding:16px 18px; margin-bottom:10px;
        }
        .result-question { color:#111827; font-size:1rem; font-weight:750; margin-bottom:7px; }
        .result-status { color:#1d4ed8; font-size:.9rem; font-weight:750; margin-bottom:6px; }
        .result-copy { color:#4b5563; font-size:.9rem; line-height:1.65; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _auto_llm_provider() -> str:
    configured = os.getenv("MARKET_MOVER_LLM_PROVIDER", "").strip().lower()
    if configured in {"gemini", "groq", "ollama", "none"}:
        return configured
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    # 로컬 Ollama가 실제로 실행 중인지 화면 로딩 때 확인하면 그 자체가 지연을 만들 수
    # 있으므로 자동 모드에서는 호출하지 않습니다. Ollama 사용자는 메뉴에서 명시적으로 선택합니다.
    return "none"


def _render_llm_settings() -> tuple[str, str | None]:
    with st.sidebar.expander("AI 요약 설정", expanded=False):
        provider_options = ["자동 선택", "제미나이", "그록", "올라마", "사용하지 않음"]
        stored = st.session_state.get("llm_provider_choice", "자동 선택")
        provider_choice = st.selectbox(
            "AI 서비스",
            provider_options,
            index=provider_options.index(stored) if stored in provider_options else 0,
            key="llm_provider_choice",
        )
        provider_map = {
            "자동 선택": _auto_llm_provider(),
            "제미나이": "gemini",
            "그록": "groq",
            "올라마": "ollama",
            "사용하지 않음": "none",
        }
        provider = provider_map[provider_choice]
        model_default = DEFAULT_LLM_MODELS.get(provider, "")
        custom_model = st.checkbox("사용 모델 직접 지정", value=False, disabled=provider == "none")
        if custom_model:
            model_name = st.text_input(
                "사용 모델",
                value=st.session_state.get(f"model_{provider}", model_default),
                key=f"model_{provider}",
            ).strip() or None
        else:
            model_name = model_default or None
        if provider == "gemini":
            api_key = st.text_input("제미나이 인증 키", type="password", key="gemini_key_input")
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
        elif provider == "groq":
            api_key = st.text_input("그록 인증 키", type="password", key="groq_key_input")
            if api_key:
                os.environ["GROQ_API_KEY"] = api_key
        status = {
            "gemini": "제미나이로 번역과 요약을 생성합니다.",
            "groq": "그록으로 번역과 요약을 생성합니다.",
            "ollama": "컴퓨터에 설치된 올라마를 사용합니다.",
            "none": "AI 기능을 사용하지 않고 계산 결과만 표시합니다.",
        }[provider]
        st.caption(status)
    st.session_state["active_llm_provider"] = provider
    st.session_state["active_llm_model"] = model_name
    return provider, model_name


def _current_llm() -> tuple[str, str | None]:
    return (
        st.session_state.get("active_llm_provider", _auto_llm_provider()),
        st.session_state.get("active_llm_model"),
    )


@st.cache_data(show_spinner=False)
def cached_data() -> dict:
    return load_dashboard_data()


def _as_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _safe_text(*values, fallback="-") -> str:
    """NaN을 화면에 그대로 노출하지 않고 첫 번째 유효 문자열을 반환합니다."""
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


_SCOPE_MARKERS = {
    "live_track2": "\u200b",
    "live_date": "\u200c",
    "live_chart_all": "\u200d",
    "live_chart": "\u2060",
    "explorer": "\u2061",
    "case_study": "\u2062",
    "event": "\u2063",
}


def _scoped_label(label: str, widget_scope: str) -> str:
    """여러 상단 탭에서 같은 사건을 그려도 위젯 ID가 충돌하지 않게 합니다."""
    return label + _SCOPE_MARKERS.get(widget_scope, "\u2064")


def _json_list(value) -> list:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _format_pct(value, digits: int = 2, fallback: str = "-") -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return fallback
    return f"{number * 100:+.{digits}f}%"


def _format_number(value, digits: int = 2, fallback: str = "-") -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return fallback
    return f"{number:.{digits}f}"


PERSON_LABELS = {"Musk": "일론 머스크", "Trump": "도널드 트럼프"}
TICKER_LABELS = {
    "TSLA": "테슬라(TSLA)",
    "SPY": "S&P 500(SPY)",
    "QQQ": "나스닥 100(QQQ)",
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
TRACK_LABELS = {"track1_auto": "SNS 원문 자료", "track2_manual": "2025년 4월 이후 뉴스 자료"}
SESSION_LABELS = {
    "premarket": "장 시작 전",
    "regular_session": "장중",
    "afterhours": "장 마감 후",
    "market_closed": "휴장일",
}
PROVIDER_LABELS = {"gemini": "제미나이", "groq": "그록", "ollama": "올라마", "none": "사용하지 않음"}
TEST_QUESTION_LABELS = {
    "h1_volatility_before_after": "게시물 전후 변동성 차이",
    "h2_topic_difference": "전체 자료의 주제별 차이",
    "h2b_topic_within_ticker_QQQ": "나스닥 100 안에서 주제별 차이",
    "h2b_topic_within_ticker_SPY": "S&P 500 안에서 주제별 차이",
    "h2b_topic_within_ticker_TSLA": "테슬라 안에서 주제별 차이",
    "h3_musk_vs_trump": "전체 자료의 인물별 차이",
    "h3b_musk_vs_trump_within_qqq": "나스닥 100 안에서 인물별 차이",
    "h4_engagement_correlation": "게시물 반응 수와 주가 반응의 관련성",
    "trump_role_difference": "도널드 트럼프 활동 시기별 차이",
    "exploratory_sentiment_positive_vs_negative": "긍정·부정 게시물의 차이",
    "exploratory_sentiment_score_correlation": "감성 점수와 주가 반응의 관련성",
    "exploratory_novelty_correlation": "새로운 내용과 주가 반응의 관련성",
    "exploratory_high_vs_low_novelty": "내용의 새로움 수준별 차이",
}


def _person_label(value) -> str:
    return PERSON_LABELS.get(str(value), _safe_text(value))


def _ticker_label(value) -> str:
    return TICKER_LABELS.get(str(value), _safe_text(value))


def _topic_label(value) -> str:
    return TOPIC_LABELS.get(str(value), _safe_text(value).replace("_", " "))


def _track_label(value) -> str:
    return TRACK_LABELS.get(str(value), _safe_text(value))


def _session_label(value) -> str:
    return SESSION_LABELS.get(str(value), _safe_text(value))


def _localize_known_terms(value) -> str:
    text = _safe_text(value, fallback="")
    replacements = [
        ("Elon Musk", "일론 머스크"),
        ("Donald Trump", "도널드 트럼프"),
        ("Truth Social", "트루스소셜"),
        ("Track1", "기존 SNS 자료"),
        ("Track2", "2025년 4월 이후 뉴스 자료"),
        ("Musk", "일론 머스크"),
        ("Trump", "도널드 트럼프"),
        ("Tesla", "테슬라"),
    ]
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def _short_text(row: pd.Series, limit: int = 90) -> str:
    value = _safe_text(row.get("description"), row.get("text_raw"), row.get("text_clean"), fallback="내용 없음")
    if _looks_korean(value):
        value = _localize_known_terms(value)
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _decorate_events(events: pd.DataFrame) -> pd.DataFrame:
    """내부 식별자와 영문 범주를 화면용 한국어 정보로 바꿉니다."""
    work = events.copy()
    if work.empty:
        return work
    work["_날짜정렬"] = pd.to_datetime(work.get("event_date"), errors="coerce")
    work["_아이디정렬"] = work.get("event_id", pd.Series(index=work.index, dtype=str)).astype(str)
    ordered_index = work.sort_values(["_날짜정렬", "_아이디정렬"], na_position="last").index
    number_map = {row_index: number for number, row_index in enumerate(ordered_index, start=1)}
    width = max(4, len(str(len(work))))
    work["사건 번호"] = [f"사건 {number_map[index]:0{width}d}" for index in work.index]
    work["날짜"] = work.get("event_date", pd.Series(index=work.index, dtype=str)).astype(str)
    work["인물"] = work.get("person", pd.Series(index=work.index, dtype=str)).map(_person_label)
    work["주제"] = work.get("topic", pd.Series(index=work.index, dtype=str)).map(_topic_label)
    work["관련 시장"] = work.get("ticker", pd.Series(index=work.index, dtype=str)).map(_ticker_label)
    work["자료 구분"] = work.get("track", pd.Series(index=work.index, dtype=str)).map(_track_label)
    work["원인 구분"] = work.get("contamination_level", pd.Series(index=work.index, dtype=str)).map(_contam_label)
    work["실제 등락률"] = pd.to_numeric(work.get("stock_return"), errors="coerce")
    work["시장 대비 등락률"] = pd.to_numeric(work.get("abnormal_return"), errors="coerce")
    work["반응 강도"] = pd.to_numeric(work.get("impact_score"), errors="coerce")
    work["내용"] = work.apply(_short_text, axis=1)
    return work

# "CLEAN"은 텍스트가 정제됐다는 뜻이 아니라(그건 preprocess.py가 이미 다른 단계에서 처리함),
# "이 이벤트가 다중게시/매크로/시장충격과 안 섞여서 원인을 특정할 수 있는가"를 뜻하는
# 이벤트 스터디 방법론 용어다(contamination.py, §4-2). 화면에는 오해 없는 표현으로 보여준다.
CONTAM_LABELS = {
    "CLEAN": "원인 비교적 명확",
    "MINOR": "다른 요인 일부 있음",
    "MAJOR": "여러 요인이 겹쳐 원인 구분 어려움",
}


def _contam_label(value) -> str:
    return CONTAM_LABELS.get(str(value), str(value))


def _looks_korean(text: str) -> bool:
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    return hangul >= max(3, len(text) * 0.15)


@st.cache_data(show_spinner=False)
def _cached_translate_preview(text: str, provider: str, model_name: str | None) -> str:
    """게이지 카드의 원문 미리보기(영어 원문)를 한국어로 번역한다. Track2 설명처럼
    이미 한국어인 텍스트는 그대로 두고 불필요한 LLM 호출을 건너뛴다."""
    if not text or _looks_korean(text):
        return text
    translated = translate_to_korean(text, provider=provider, model_name=model_name)
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


def _group_chat_summary(events: pd.DataFrame, column: str) -> list[dict]:
    if events.empty or column not in events.columns:
        return []
    work = events.copy()
    work["abs_abnormal_return"] = pd.to_numeric(work.get("abnormal_return"), errors="coerce").abs()
    work["impact_score_num"] = pd.to_numeric(work.get("impact_score"), errors="coerce")
    grouped = (
        work.dropna(subset=[column])
        .groupby(column, dropna=False)
        .agg(
            사건수=(column, "size"),
            평균_절대초과수익률=("abs_abnormal_return", "mean"),
            중앙값_절대초과수익률=("abs_abnormal_return", "median"),
            평균_반응강도=("impact_score_num", "mean"),
        )
        .reset_index()
    )
    for numeric in ["평균_절대초과수익률", "중앙값_절대초과수익률", "평균_반응강도"]:
        grouped[numeric] = grouped[numeric].round(6)
    if column == "person":
        grouped[column] = grouped[column].map(_person_label)
    elif column == "ticker":
        grouped[column] = grouped[column].map(_ticker_label)
    elif column == "topic":
        grouped[column] = grouped[column].map(_topic_label)
    grouped = grouped.rename(columns={
        "person": "인물",
        "ticker": "관련 시장",
        "topic": "주제",
        "평균_절대초과수익률": "평균 시장 대비 움직임",
        "중앙값_절대초과수익률": "중앙값 시장 대비 움직임",
        "평균_반응강도": "평균 반응 강도",
    })
    return grouped.to_dict("records")


def _chat_scope(events: pd.DataFrame, question: str) -> tuple[pd.DataFrame, list[str]]:
    scoped = events.copy()
    filters = []
    question_lower = question.lower()
    person_terms = {
        "Musk": ["musk", "일론 머스크", "머스크"],
        "Trump": ["trump", "도널드 트럼프", "트럼프"],
    }
    mentioned_people = [person for person, terms in person_terms.items() if any(term in question_lower for term in terms)]
    if mentioned_people and "person" in scoped.columns:
        scoped = scoped[scoped["person"].astype(str).isin(mentioned_people)]
        filters.append("인물: " + ", ".join(_person_label(person) for person in mentioned_people))
    ticker_terms = {
        "TSLA": ["tsla", "테슬라"],
        "QQQ": ["qqq", "나스닥"],
        "SPY": ["spy", "s&p", "에스앤피"],
    }
    mentioned_tickers = [ticker for ticker, terms in ticker_terms.items() if any(term in question_lower for term in terms)]
    if mentioned_tickers and "ticker" in scoped.columns:
        scoped = scoped[scoped["ticker"].astype(str).str.upper().isin(mentioned_tickers)]
        filters.append("관련 시장: " + ", ".join(_ticker_label(ticker) for ticker in mentioned_tickers))
    if any(token in question_lower for token in ["clean", "단일요인", "단일 요인", "원인 명확"]):
        if "contamination_level" in scoped.columns:
            scoped = scoped[scoped["contamination_level"].eq("CLEAN")]
            filters.append("원인 구분: 비교적 명확")
    if "topic" in scoped.columns:
        for topic in scoped["topic"].dropna().astype(str).unique():
            topic_terms = [topic.lower(), _topic_label(topic).lower()]
            if any(term in question_lower for term in topic_terms):
                scoped = scoped[scoped["topic"].astype(str).eq(topic)]
                filters.append(f"주제: {_topic_label(topic)}")
                break
    return scoped, filters


def _build_chat_context(events: pd.DataFrame, stats: dict, question: str, history: list) -> dict:
    display_all = _decorate_events(events)
    scoped, filters = _chat_scope(display_all, question)
    clean = scoped[scoped.get("contamination_level", pd.Series(index=scoped.index, dtype=str)).eq("CLEAN")]
    display_scoped = scoped.copy()
    top_columns = [
        column for column in [
            "사건 번호", "날짜", "인물", "주제", "관련 시장", "시장 대비 등락률",
            "반응 강도", "내용", "source_url",
        ] if column in display_scoped.columns
    ]
    top_events = (
        display_scoped.assign(_impact=pd.to_numeric(display_scoped.get("impact_score"), errors="coerce"))
        .sort_values("_impact", ascending=False)
        .head(8)[top_columns]
        .copy()
    )
    if "source_url" in top_events.columns:
        top_events = top_events.rename(columns={"source_url": "자료 링크"})
    event_dates = pd.to_datetime(scoped.get("event_date"), errors="coerce")
    tests_summary = {}
    for name, value in stats.get("tests", {}).items():
        supported = value.get("reject_fdr_0.05")
        tests_summary[TEST_QUESTION_LABELS.get(name, "추가 연구 질문")] = {
            "결론": "차이 확인" if supported is True else "뚜렷한 차이 없음" if supported is False else "표본 부족",
            "보정 유의확률": value.get("p_value_fdr_bh", value.get("p_value")),
        }
    return {
        "질문에_적용한_필터": filters or ["없음"],
        "필터된_사건수": int(len(scoped)),
        "그중_단일요인_사건수": int(len(clean)),
        "기간": {
            "시작": str(event_dates.min().date()) if event_dates.notna().any() else None,
            "종료": str(event_dates.max().date()) if event_dates.notna().any() else None,
        },
        "인물별_계산": _group_chat_summary(scoped, "person"),
        "관련시장별_계산": _group_chat_summary(scoped, "ticker"),
        "주제별_계산": _group_chat_summary(scoped, "topic"),
        "반응강도_상위사건": top_events.to_dict("records"),
        "가설검정": tests_summary,
        "설명력_참고값": {
            "주제": stats.get("effect_sizes", {}).get("eta_topic"),
            "인물": stats.get("effect_sizes", {}).get("eta_person"),
        },
        "최근대화": [
            {"역할": item.get("role"), "내용": item.get("content")}
            for item in history[-6:]
        ],
    }


def _deterministic_chat_answer(question: str, context: dict) -> str:
    top = context.get("반응강도_상위사건", [])
    question_lower = question.lower()
    if any(token in question_lower for token in ["가장", "상위", "큰 반응", "최대"]):
        if not top:
            return "조건에 맞는 사건이 없습니다. 필터나 질문 범위를 넓혀보세요."
        lines = []
        for index, event in enumerate(top[:5], start=1):
            lines.append(
                f"{index}. {event.get('사건 번호')} · {event.get('날짜')} · {event.get('인물')} · "
                f"{event.get('관련 시장')} · {event.get('주제')}, 반응 강도 {_format_number(event.get('반응 강도'))}"
            )
        return "계산된 반응 강도 상위 사건입니다.\n\n" + "\n".join(lines)
    if (("musk" in question_lower or "머스크" in question_lower)
            and ("trump" in question_lower or "트럼프" in question_lower)):
        rows = context.get("인물별_계산", [])
        return "인물별 계산 결과입니다. " + "; ".join(
            f"{row.get('인물')}: {row.get('사건수')}건, 평균 시장 대비 움직임 {_format_pct(row.get('평균 시장 대비 움직임'))}"
            for row in rows
        ) + ". 다만 연결된 시장 구성이 달라 인물 자체의 효과로 단정할 수 없습니다."
    return (
        f"질문 조건에 맞는 사건은 {context.get('필터된_사건수', 0):,}건이고, "
        f"그중 다른 요인과 덜 섞인 사건은 {context.get('그중_단일요인_사건수', 0):,}건입니다. "
        "인물·관련 시장·주제별 계산과 상위 사건을 근거로 더 구체적으로 질문해보세요."
    )


def render_live_trump_feed(key_prefix: str = "default") -> None:
    with st.expander("도널드 트럼프 최신 게시물 확인", expanded=False):
        st.caption(
            "트루스소셜 공개 피드에서 최근 게시물을 확인합니다. 아직 가격 반응이 끝나지 않은 게시물이므로 "
            "주가 반응 크기는 계산하지 않고, 분석 대상 주제와 관련 시장만 분류합니다. 일론 머스크 게시물은 "
            "안정적인 무료 실시간 자료가 없어 이 기능에서 제공하지 않습니다."
        )
        if st.button("최신 게시물 확인", key=f"{key_prefix}_trump_feed_refresh"):
            with st.spinner("트루스소셜 공개 피드를 확인하는 중..."):
                try:
                    items = fetch_trump_feed(30)
                except Exception as exc:
                    st.error(f"최신 게시물을 불러오지 못했습니다: {exc}")
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
                    st.warning("최근 7일 이내 게시물을 찾지 못했습니다. 공개 피드가 아직 갱신되지 않았을 수 있습니다.")
                    return

                window_label = "24시간" if window_used == 1 else f"{window_used}일"
                st.caption(f"최근 {window_label} 이내 게시물 {len(selected_items)}건")

                rows = []
                for item in selected_items:
                    try:
                        text = fetch_post_text(item["link"])
                    except Exception:
                        text = ""
                    pub_date = item.get("pub_date")
                    korea_time = timezone(timedelta(hours=9))
                    posted = f"{pub_date.astimezone(korea_time):%Y-%m-%d %H:%M}" if pub_date else "-"
                    if not text:
                        rows.append(
                            {"작성자": "도널드 트럼프", "게시 시각(한국)": posted, "시장 관련 여부": "확인 불가", "주제": "-", "관련 시장": "-",
                             "원문 일부": "본문을 확인할 수 없습니다.", "원문 링크": item.get("link")}
                        )
                        continue
                    text_c = clean_text(text)
                    relevant = is_market_relevant(text_c, "Trump")
                    topic = assign_topic(text_c, "Trump")
                    ticker = map_ticker("Trump", topic) if relevant else None
                    rows.append(
                        {
                            "작성자": "도널드 트럼프",
                            "게시 시각(한국)": posted,
                            "시장 관련 여부": "관련 있음" if relevant else "관련 낮음",
                            "주제": _topic_label(topic),
                            "관련 시장": _ticker_label(ticker) if ticker else "-",
                            "원문 일부": text_c[:100],
                            "원문 링크": item.get("link"),
                        }
                    )
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        column_config={
                            "원문 링크": st.column_config.LinkColumn("원문", display_text="열기"),
                        },
                    )
                    st.caption(f"조회 시각(한국): {datetime.now(timezone(timedelta(hours=9))):%Y-%m-%d %H:%M:%S}")
                else:
                    st.info("피드에서 항목을 가져오지 못했습니다.")


@st.cache_data(show_spinner=False)
def _cached_ask_data_answer(
    question: str,
    context: dict,
    provider: str,
    model_name: str | None,
) -> str:
    return answer_data_question(
        question,
        context,
        provider=provider,
        model_name=model_name,
    )


def _render_compact_data_chat(events: pd.DataFrame, stats: dict, gauge_states: list) -> None:
    """실시간 메인 오른쪽에 유지되는 간단한 질의 패널입니다.

    전체 Ask the Data 메뉴와 같은 계산 로직을 사용하되, 현재 계기판 상태를
    추가 근거로 전달합니다. AI가 꺼져 있어도 pandas 계산형 답변은 동작합니다.
    """
    st.markdown('<div class="mini-chat-title">데이터에게 묻기</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mini-chat-caption">현재 화면과 전체 사건 자료를 기준으로 답합니다. '
        '미래 가격을 예측하거나 투자 판단을 대신하지 않습니다.</div>',
        unsafe_allow_html=True,
    )
    provider, model_name = _current_llm()
    history = st.session_state.setdefault("compact_chat_history_v3", [])

    chat_box = st.container(height=390, border=True)
    with chat_box:
        if not history:
            st.caption("예: 최근 세 시장 중 반응이 가장 컸던 곳은 어디야?")
        for item in history[-8:]:
            with st.chat_message(item.get("role", "assistant")):
                st.write(item.get("content", ""))

    quick_cols = st.columns(2)
    pending_question = None
    with quick_cols[0]:
        if st.button("최근 반응 비교", key="compact_compare", use_container_width=True):
            pending_question = "최근 계기판의 세 시장 반응을 비교해줘"
    with quick_cols[1]:
        if st.button("가장 큰 사건", key="compact_top", use_container_width=True):
            pending_question = "반응이 가장 컸던 사건 5개"

    typed_question = st.chat_input(
        "이 화면에 대해 질문하세요",
        key="compact_chat_input",
    )
    question = typed_question or pending_question
    if not question:
        return

    history.append({"role": "user", "content": question})
    context = _build_chat_context(events, stats or {}, question, history[:-1])
    context["현재_계기판"] = [
        {
            "관련 시장": _ticker_label(state.get("ticker")),
            "과거 대비 위치": state.get("percentile"),
            "해당일 실제 등락률": state.get("stock_return"),
            "시장 대비 등락률": state.get("abnormal_return"),
            "최근 사건 인물": _person_label(state.get("person")),
            "최근 사건 주제": _topic_label(state.get("topic")),
        }
        for state in gauge_states if state.get("has_data")
    ]
    if provider == "none":
        answer = _deterministic_chat_answer(question, context)
    else:
        with st.spinner("계산 결과를 정리하는 중..."):
            answer = _cached_ask_data_answer(question, context, provider, model_name)
        if answer.startswith("판단보류:"):
            answer += "\n\n계산 결과만 안내합니다.\n\n" + _deterministic_chat_answer(question, context)
    history.append({"role": "assistant", "content": answer})
    st.rerun()


def _render_analysis_period(events: pd.DataFrame) -> tuple:
    """페이지 맨 위에서 바로 보이는 분석 기간 위젯. (start_date, end_date) 또는
    자료가 없으면 (None, None)을 반환한다 — 사이드바(접혀서 안 보이기 쉬움) 대신
    본문 최상단에 둬서 다른 조작 없이 바로 설정할 수 있게 한다."""
    st.markdown(
        '<div style="background:#ffffff;border:1px solid #dfe4ea;border-left:4px solid #df3e52;'
        'border-radius:12px;padding:14px 18px;margin-bottom:.9rem;">'
        '<div style="font-weight:800;font-size:1.05rem;color:#172033;margin-bottom:2px;">분석 기간</div>'
        '<div style="color:#64748b;font-size:.85rem;">이 화면 전체(계기판·차트·요약)가 아래에서 고른 '
        "기간을 기준으로 계산됩니다.</div></div>",
        unsafe_allow_html=True,
    )
    posted_dates = pd.to_datetime(events.get("posted_at"), errors="coerce").dropna() if not events.empty else pd.Series(dtype="datetime64[ns]")
    if posted_dates.empty:
        st.info("분석할 사건 자료가 아직 없어 기간을 선택할 수 없습니다. `data/raw/`에 원본 CSV를 넣고 파이프라인을 실행하면 선택할 수 있습니다.")
        return None, None

    data_min, data_max = posted_dates.min().date(), posted_dates.max().date()
    col1, col2 = st.columns([2, 1])
    with col1:
        date_range = st.date_input(
            "분석 기간",
            value=(data_min, data_max),
            min_value=data_min,
            max_value=data_max,
            key="analysis_period",
            label_visibility="collapsed",
        )
    with col2:
        st.caption(f"보유 자료 기간: {data_min} ~ {data_max}")
    if isinstance(date_range, tuple) and len(date_range) == 2:
        return date_range
    return data_min, data_max


def render_live_main(
    events: pd.DataFrame,
    daily_prices: pd.DataFrame,
    stats: dict | None = None,
    period_start=None,
    period_end=None,
) -> None:
    st.subheader("최근 연결 사건의 시장 반응")
    st.caption(
        "선택한 기간에서 각 시장과 연결된 가장 최근 사건을 보여줍니다. 해당 사건 뒤의 실제 움직임을 "
        "같은 시장의 과거 사건과 비교한 값이며, 미래 주가를 예측하는 수치가 아닙니다."
    )

    posted_dates_all = pd.to_datetime(events.get("posted_at"), errors="coerce").dropna() if not events.empty else pd.Series(dtype="datetime64[ns]")
    if not posted_dates_all.empty:
        bound_min, bound_max = posted_dates_all.min().date(), posted_dates_all.max().date()
        default_start = period_start if period_start is not None else bound_min
        default_end = period_end if period_end is not None else bound_max
        gcol1, gcol2 = st.columns([2, 1])
        with gcol1:
            local_range = st.date_input(
                "이 구간의 기간",
                value=(default_start, default_end),
                min_value=bound_min,
                max_value=bound_max,
                key="live_gauge_period",
            )
        with gcol2:
            st.caption("기본값은 맨 위에서 고른 분석 기간입니다. 최근 사건이 극단값 위주라면 기간을 좁혀 다른 사건을 볼 수 있어요.")
        if isinstance(local_range, tuple) and len(local_range) == 2:
            gauge_start, gauge_end = local_range
        else:
            gauge_start, gauge_end = default_start, default_end
    else:
        gauge_start, gauge_end = period_start, period_end

    render_live_trump_feed("live_main")

    with st.expander("반응 크기는 어떻게 읽나요?"):
        st.markdown(
            "- **과거 대비 위치**는 같은 시장의 과거 사건 중 이번 반응이 어느 정도로 컸는지 보여줍니다.\n"
            "- **실제 등락률**은 해당 시장이나 종목이 그날 실제로 오른 폭 또는 내린 폭입니다.\n"
            "- **시장 대비 등락률**은 전체 시장의 움직임을 제외하고 남은 차이입니다.\n"
            "- 이 수치는 이미 일어난 일을 비교하는 용도이며 미래 반응을 예측하지 않습니다."
        )

    gauge_states = compute_ticker_gauges(events, tickers=("QQQ", "SPY", "TSLA"), start_date=gauge_start, end_date=gauge_end)
    gauge_cols = st.columns(3)
    for col, gstate in zip(gauge_cols, gauge_states):
        with col:
            st.markdown(render_single_gauge_html(gstate), unsafe_allow_html=True)

    st.divider()
    chart_col, chat_col = st.columns([3.35, 1.15], gap="large")
    with chart_col:
        _render_price_chart_section(events, daily_prices, gauge_states, period_start=gauge_start, period_end=gauge_end)
    with chat_col:
        _render_compact_data_chat(events, stats or {}, gauge_states)


def _render_price_chart_section(
    events: pd.DataFrame,
    daily_prices: pd.DataFrame,
    gauge_states: list,
    period_start=None,
    period_end=None,
) -> None:
    st.subheader("주가 흐름과 게시물 발생 시점")
    st.caption("보라색 점은 일론 머스크, 주황색 점은 도널드 트럼프 사건입니다. 점을 선택하면 원문과 전후 주가를 확인할 수 있습니다.")

    if events.empty or "ticker" not in events.columns or daily_prices.empty:
        st.info("아직 표시할 가격/이벤트 데이터가 없습니다.")
        return

    tickers = sorted(events["ticker"].dropna().unique().tolist())
    if not tickers:
        st.info("종목 정보가 없습니다.")
        return
    latest_ticker = next((g["ticker"] for g in gauge_states if g.get("has_data")), tickers[0])
    default_idx = tickers.index(latest_ticker) if latest_ticker in tickers else 0
    ticker_options = tickers + ["전체 시장 비교"]
    ticker = st.selectbox(
        "관련 시장 선택",
        ticker_options,
        index=default_idx,
        format_func=lambda value: value if value == "전체 시장 비교" else _ticker_label(value),
    )

    persons = sorted(events["person"].dropna().unique().tolist()) if "person" in events.columns else []
    person_choice = st.radio(
        "인물",
        ["전체"] + persons,
        horizontal=True,
        format_func=lambda value: value if value == "전체" else _person_label(value),
    )

    events_f = events if person_choice == "전체" else events[events["person"] == person_choice]

    # 2025-04-14 이후 사건(Track2)은 차트에 딱 6개만 뚝뚝 떨어져 있어서 점을 정확히
    # 클릭하기 어렵다는 피드백이 있었다 — 클릭 대신 목록에서 바로 골라 보는 방법을 추가한다.
    track2 = events[events.get("track", pd.Series(dtype=str)).eq("track2_manual")] if "track" in events.columns else events.iloc[0:0]
    if not track2.empty:
        track2_options = ["(선택 안 함)"] + [
            f"{r.get('event_date')} · {_person_label(r.get('person'))} · {_ticker_label(r.get('ticker'))} · {_topic_label(r.get('topic'))}"
            for _, r in track2.sort_values("event_date").iterrows()
        ]
        track2_ids = [None] + track2.sort_values("event_date")["event_id"].tolist()
        track2_pick = st.selectbox(
            "2025년 4월 이후 뉴스 사건 바로가기",
            track2_options,
        )
        if track2_pick != "(선택 안 함)":
            picked_id = track2_ids[track2_options.index(track2_pick)]
            _render_event_detail(events, picked_id, widget_scope="live_track2")
            return

    # 그 외 날짜(Track1)는 점이 너무 촘촘히 찍혀 있어(종목당 최대 1000개 이상) 정확히
    # 클릭하기가 어렵다 — 날짜를 직접 골라서 그 근처 게시물을 바로 찾아보는 방법도 추가한다.
    if ticker != "전체 시장 비교":
        date_lookup_pool = events_f[events_f["ticker"] == ticker].dropna(subset=["event_date"]).copy()
        date_lookup_pool = date_lookup_pool[_has_real_content(date_lookup_pool)]
        date_lookup_pool["event_date"] = _as_datetime(date_lookup_pool["event_date"])
        date_lookup_pool = date_lookup_pool.dropna(subset=["event_date"])
        if not date_lookup_pool.empty:
            with st.expander("날짜로 사건 찾기"):
                min_d = date_lookup_pool["event_date"].min().date()
                max_d = date_lookup_pool["event_date"].max().date()
                picked_date = st.date_input(
                    "날짜 선택", value=max_d, min_value=min_d, max_value=max_d, key="date_lookup"
                )
                if st.button("이 날짜와 가장 가까운 게시물 보기", key="date_lookup_go"):
                    diffs = (date_lookup_pool["event_date"] - pd.Timestamp(picked_date)).abs()
                    nearest_id = date_lookup_pool.loc[diffs.idxmin(), "event_id"]
                    st.session_state["date_lookup_result"] = nearest_id
        picked_id = st.session_state.get("date_lookup_result")
        if picked_id and str(events.loc[events["event_id"] == picked_id, "ticker"].squeeze()) == ticker:
            _render_event_detail(events, picked_id, widget_scope="live_date")
            return

    fig = go.Figure()

    if ticker == "전체 시장 비교":
        # 종목마다 가격 단위(TSLA 수백 달러 vs QQQ/SPY)가 달라 그대로 겹치면 비교가
        # 안 되므로, 각 종목의 표시 구간 첫날을 100으로 맞춘 정규화 지수로 겹쳐 그린다.
        line_colors = {"QQQ": "#2563eb", "SPY": "#16a34a", "TSLA": "#ef4444"}
        person_colors = {"Musk": "#7c3aed", "Trump": "#f97316"}
        legend_shown = set()
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
                    name=_ticker_label(tkr), hovertemplate=f"{_ticker_label(tkr)} %{{y:.1f}}<extra></extra>",
                )
            )
            tkr_events = events_f[events_f["ticker"] == tkr].dropna(subset=["event_date"]).copy()
            tkr_events = tkr_events[_has_real_content(tkr_events)]
            tkr_events["event_date"] = _as_datetime(tkr_events["event_date"])
            tkr_merged = tkr_events.merge(p[["date", "indexed"]], left_on="event_date", right_on="date", how="left")
            tkr_merged = tkr_merged.dropna(subset=["indexed"])
            if not tkr_merged.empty:
                tkr_merged["_표시인물"] = tkr_merged["person"].map(_person_label)
                tkr_merged["_표시주제"] = tkr_merged["topic"].map(_topic_label)
                for person_code, color in person_colors.items():
                    subset = tkr_merged[tkr_merged["person"] == person_code]
                    if subset.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=subset["event_date"], y=subset["indexed"], mode="markers",
                            marker=dict(size=9, color=color, opacity=0.72, line=dict(width=1, color="#ffffff")),
                            customdata=subset[["event_id"]].values if "event_id" in subset.columns else None,
                            text=(subset["_표시인물"] + " · " + subset["_표시주제"]),
                            hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>선택하면 사건 상세 보기<extra></extra>",
                            name=_person_label(person_code),
                            legendgroup=person_code,
                            showlegend=person_code not in legend_shown,
                        )
                    )
                    legend_shown.add(person_code)
        fig.update_layout(
            yaxis_title="조회 시작일을 100으로 환산",
            legend=dict(
                orientation="h", x=0, y=1.1, xanchor="left", yanchor="bottom",
                font=dict(size=11), bgcolor="rgba(255,255,255,0)",
            ),
        )
        fig.update_xaxes(
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="#94a3b8", spikethickness=1, spikedash="solid",
        )
        if period_start is not None and period_end is not None:
            fig.update_xaxes(range=[pd.Timestamp(period_start), pd.Timestamp(period_end)])
        click = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun", selection_mode="points", key="live_main_chart"
        )
        _render_selected_event(click, events, widget_scope="live_chart_all")
        return

    prices = daily_prices[daily_prices["ticker"] == ticker].copy()
    prices["date"] = _as_datetime(prices["date"])
    prices = prices.dropna(subset=["date", "close"]).sort_values("date")
    if prices.empty:
        st.info(f"{ticker} 가격 데이터가 없습니다.")
        return

    if period_start is not None and period_end is not None:
        windowed = prices[(prices["date"] >= pd.Timestamp(period_start)) & (prices["date"] <= pd.Timestamp(period_end))]
        y_source = windowed["close"] if not windowed.empty else prices["close"]
    else:
        y_source = prices["close"]
    y_min = y_source.min()
    y_max = y_source.max()
    baseline = y_min - (y_max - y_min) * 0.05

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
            fillcolor="rgba(37,99,235,0.14)",
            name=f"{ticker} 종가",
            hovertemplate="%{x|%Y-%m-%d}<br>종가 %{y:.2f}<extra></extra>",
            showlegend=False,
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
        merged["_표시인물"] = merged["person"].map(_person_label)
        merged["_표시주제"] = merged["topic"].map(_topic_label)
        person_colors = {"Musk": "#7c3aed", "Trump": "#f97316"}
        for person_code, color in person_colors.items():
            subset = merged[merged["person"] == person_code]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["event_date"],
                    y=subset["close"],
                    mode="markers",
                    marker=dict(size=10, color=color, opacity=0.76, line=dict(width=1, color="#ffffff")),
                    customdata=subset[["event_id"]].values if "event_id" in subset.columns else None,
                    text=(subset["_표시인물"] + " · " + subset["_표시주제"]),
                    hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>선택하면 사건 상세 보기<extra></extra>",
                    name=_person_label(person_code),
                )
            )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="종가($)",
        plot_bgcolor="white",
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0,
            y=1.1,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0)",
        ),
        hovermode="closest",
    )
    fig.update_yaxes(
        range=[baseline, y_max * 1.03],
        showspikes=False,
    )
    if period_start is not None and period_end is not None:
        x_range = [pd.Timestamp(period_start), pd.Timestamp(period_end)]
    else:
        x_range = [prices["date"].min(), prices["date"].max()]
    fig.update_xaxes(
        range=x_range,
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
    _render_selected_event(click, events, widget_scope="live_chart")


def _render_news_search(row: pd.Series, event_id, widget_scope: str = "event") -> None:
    """Kaggle 원본에는 게시물 URL이 없어서, 대신 그 시점 전후로 실제 관련 뉴스가
    있었는지 Google News RSS로 그 자리에서 자동검색해 보여준다."""
    if st.button("이 시점의 관련 기사 찾기", key=f"{widget_scope}_news_search_{event_id}"):
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
        if pd.isna(event_date):
            st.info("이벤트 날짜가 없어 검색할 수 없습니다.")
            return
        query = build_query(row)
        ticker_name = _ticker_label(row.get("ticker"))
        window_start = event_date - pd.Timedelta(days=2)
        window_end = event_date + pd.Timedelta(days=2)
        with st.spinner("게시 날짜 전후의 관련 기사를 찾는 중..."):
            try:
                articles = fetch_google_news_rss(query, window_start, window_end, limit=5)
            except Exception as exc:
                st.error(f"뉴스 검색 실패: {exc}")
                return
        st.caption(
            f"{_person_label(row.get('person'))}, {_topic_label(row.get('topic'))}, {ticker_name}을 기준으로 "
            f"게시 날짜 전후 2일({window_start.date()} ~ {window_end.date()})의 기사를 찾았습니다. "
            "게시물 문장 전체가 아니라 사건의 인물·주제·시장 정보를 조합해 검색합니다."
        )
        if not articles:
            st.info(f"{event_date.date()} 전후 4일 안에서 관련 기사를 찾지 못했습니다.")
            return
        article_table = pd.DataFrame(articles)[["title", "source", "pub_date", "link"]].rename(
            columns={"title": "기사 제목", "source": "언론사", "pub_date": "게시 시각", "link": "기사 링크"}
        )
        st.dataframe(
            article_table,
            use_container_width=True,
            column_config={"기사 링크": st.column_config.LinkColumn("기사", display_text="열기")},
        )


def _render_selected_event(click, events: pd.DataFrame, widget_scope: str = "event") -> None:
    selected_event_id = None
    points = (click or {}).get("selection", {}).get("points", [])
    for point in points:
        customdata = point.get("customdata")
        if customdata:
            selected_event_id = customdata[0]
            break

    if selected_event_id is None:
        st.caption("차트의 사건 점을 선택하면 아래에 원문과 주가 반응이 표시됩니다.")
        return

    _render_event_detail(events, selected_event_id, widget_scope=widget_scope)


def _render_event_detail(events: pd.DataFrame, selected_event_id, widget_scope: str = "event") -> None:
    if "event_id" not in events.columns:
        st.caption("선택한 사건의 상세 정보를 찾을 수 없습니다.")
        return

    row = events[events["event_id"].astype(str) == str(selected_event_id)]
    if row.empty:
        st.caption("선택한 사건의 상세 정보를 찾을 수 없습니다.")
        return

    row = row.iloc[0]
    display_row = _decorate_events(events)
    display_row = display_row[display_row["event_id"].astype(str).eq(str(selected_event_id))].iloc[0]
    is_track2 = str(row.get("track")) == "track2_manual"
    event_number = html.escape(_safe_text(display_row.get("사건 번호")))
    person = html.escape(_person_label(row.get("person")))
    topic = html.escape(_topic_label(row.get("topic")))
    ticker = html.escape(_ticker_label(row.get("ticker")))
    event_date = html.escape(_safe_text(row.get("event_date")))
    posted_at_et = html.escape(_safe_text(row.get("posted_at_et"), row.get("posted_at")))
    market_session = html.escape(_session_label(row.get("market_session")))
    cluster_size_value = pd.to_numeric(row.get("cluster_size"), errors="coerce")
    cluster_size = int(cluster_size_value) if pd.notna(cluster_size_value) else 1
    source_url = row.get("source_url")
    has_source = isinstance(source_url, str) and source_url.strip()

    # 누가/어떤 분야/어느 종목인지 표에서 바로 안 보여서 헷갈린다는 피드백 반영 —
    # 카드 맨 위에 라벨 붙여서 명확하게 정리해둔다.
    meta = (
        '<div class="meta-row">'
        f'<span class="meta-item">{event_number}</span>'
        f'<span class="meta-item">{person}</span>'
        f'<span class="meta-item">{topic}</span>'
        f'<span class="meta-item">{ticker}</span>'
        f'<span class="meta-item">{event_date}</span>'
        f'<span class="meta-item">미국 동부시간 {posted_at_et} · {market_session}</span>'
        + (
            f'<span class="meta-item">연속 게시물 {cluster_size}개를 하나의 사건으로 묶음</span>'
            if cluster_size > 1
            else ""
        )
        + "</div>"
    )

    if is_track2:
        # Musk 원본 게시물 데이터는 2025-04-13까지만 있어(캐글 수집 범위), 그 이후
        # 사건은 게시물 원문이 아니라 뉴스 보도를 근거로 수동 등록했다. 원문인 것처럼
        # 오인되지 않도록 라벨을 분리하고 실제 기사 링크를 맨 위로 올린다.
        body = (
            f'<div class="source-card-title">{event_number} · 뉴스로 확인한 사건</div>'
            + meta
            + '<div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:9px;'
            'padding:10px 12px;margin-bottom:10px;color:#4b5563;font-size:.86rem;">'
            "2025년 4월 이후 자료는 SNS 원문이 아니라 당시 뉴스 보도를 바탕으로 정리했습니다.</div>"
        )
        if has_source:
            body += (
                f'<div style="margin-bottom:10px;"><a href="{html.escape(source_url)}" target="_blank" '
                f'style="color:#1d4ed8;font-weight:650;">근거 기사 열기</a></div>'
            )
        body += f"<div><b>사건 설명</b><br>{html.escape(_localize_known_terms(row.get('description')) or '(설명 없음)')}</div>"
    else:
        original_text = _safe_text(row.get("text_raw"), row.get("text_clean"), row.get("description"), fallback="(원문 없음)")
        body = (
            f'<div class="source-card-title">{event_number} · SNS 게시물</div>'
            + meta
            + f"<div><b>게시물 원문</b><br>{html.escape(original_text)}</div>"
        )
        if has_source:
            body += (
                f'<div style="margin-top:10px;"><a href="{html.escape(source_url)}" target="_blank" '
                f'style="color:#1d4ed8;">원문 또는 관련 자료 열기</a></div>'
            )
        else:
            # 일부 원본 파일에는 URL 컬럼이 없을 수 있습니다. 없는 링크를 추측해
            # 만들어내지 않고 정직하게 이유를 밝힙니다.
            body += (
                '<div style="margin-top:10px;font-size:12px;color:#6b7280;">'
                "원본 자료에 게시물 주소가 포함되지 않아 링크를 제공할 수 없습니다.</div>"
            )
    manual_narrative = _safe_text(row.get("narrative_reviewed"), row.get("narrative"), fallback="")
    if isinstance(manual_narrative, str) and manual_narrative.strip():
        manual_narrative = _localize_known_terms(manual_narrative)
        body += (
            f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid #e2e8f0;'
            f'font-size:13px;color:#475569;"><b>검수된 사건 설명</b><br>'
            f"{html.escape(manual_narrative)}</div>"
        )
    st.markdown(f'<div class="source-card">{body}</div>', unsafe_allow_html=True)
    _render_cluster_members(row, selected_event_id, widget_scope=widget_scope)

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "해당일 실제 등락률",
        _format_pct(row.get("stock_return")),
        help="게시물과 연결된 거래일에 해당 종목이나 지수가 실제로 움직인 폭입니다.",
    )
    metric_cols[1].metric(
        "시장 대비 등락률",
        _format_pct(row.get("abnormal_return")),
        help="해당 종목의 등락률에서 기준 시장의 움직임을 뺀 값입니다.",
    )
    metric_cols[2].metric(
        "반응 강도 점수",
        _format_number(row.get("impact_score")),
        help="가격·거래량·변동성의 평소 대비 변화 정도를 합쳐 비교한 점수입니다. 방향이 아니라 크기를 나타냅니다.",
    )
    contam_value = row.get("contamination_level")
    contam_display = "뉴스 기반 수동 정리" if is_track2 or pd.isna(contam_value) else _contam_label(contam_value)
    with metric_cols[3]:
        # 이 값은 "다른 요인 일부 있음"처럼 문장형 텍스트라 st.metric의 큰 숫자용
        # 폰트로 넣으면 잘려서 글자 하나만 확대된 것처럼 보인다 — 직접 작은 카드로 그린다.
        st.markdown(
            f'<div title="같은 시점에 다른 게시물·경제 일정·시장 충격이 함께 있었는지를 반영합니다." '
            'style="padding-top:2px;">'
            '<div style="font-size:.8rem;color:rgb(49,51,63);opacity:.75;">원인 구분 가능성</div>'
            f'<div style="font-size:1rem;font-weight:600;color:#172033;margin-top:6px;line-height:1.35;">'
            f'{html.escape(contam_display)}</div></div>',
            unsafe_allow_html=True,
        )

    provider, model_name = _current_llm()
    original_for_translation = _safe_text(
        row.get("cluster_text_raw"), row.get("text_raw"), row.get("text_clean"), fallback=""
    )
    if not is_track2 and original_for_translation and not _looks_korean(original_for_translation):
        translation_key = f"event_translation_{selected_event_id}_{provider}_{model_name}"
        if st.button(
            "한국어 번역",
            key=f"{widget_scope}_translate_event_button_{selected_event_id}",
            disabled=provider == "none",
        ):
            with st.spinner("원문을 한국어로 번역하는 중..."):
                st.session_state[translation_key] = _cached_translate_preview(
                    original_for_translation[:4000], provider, model_name
                )
        if provider == "none":
            st.caption("이 버튼은 AI 서비스를 선택해야 눌립니다 — 왼쪽 사이드바 'AI 요약 설정'에서 제미나이·그록·올라마 중 하나를 선택해주세요.")
        if translation_key in st.session_state:
            with st.expander("한국어 번역", expanded=True):
                st.write(st.session_state[translation_key])

    article_text = None
    if is_track2 and has_source:
        # 2025-04-14 이후 사건(Track2)은 원본 게시물이 아니라 뉴스 기반이라, 링크만
        # 던져두지 않고 기사 본문을 바로 그 자리에서 가져와 보여준다.
        article_text = _render_full_article(source_url, selected_event_id, widget_scope=widget_scope)
    elif not is_track2 and not has_source:
        _render_news_search(row, selected_event_id, widget_scope=widget_scope)

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
    row_dict["person"] = _person_label(row.get("person"))
    row_dict["topic"] = _topic_label(row.get("topic"))
    row_dict["ticker"] = _ticker_label(row.get("ticker"))

    commentary_source = (
        article_text
        or _safe_text(row.get("cluster_text_raw"), row.get("text_raw"), row.get("text_clean"), fallback="")
    )
    pct_text = "비교 표본 부족" if pct is None else f"같은 종목 과거 사건 중 상위 {(1-pct)*100:.0f}% 수준"
    st.caption(f"비교 기준: 같은 시장에서 원인을 비교적 구분할 수 있었던 사건 {n}건 · {pct_text}")
    commentary_key = f"event_commentary_{selected_event_id}_{provider}_{model_name}"
    if st.button(
        "AI 요약·분석 생성",
        key=f"{widget_scope}_commentary_button_{selected_event_id}",
        type="primary",
        disabled=provider == "none",
    ):
        with st.spinner("선택한 사건을 요약하고 실제 가격 반응과 비교하는 중..."):
            st.session_state[commentary_key] = _cached_event_commentary(
                str(selected_event_id),
                row_dict,
                n,
                None if pd.isna(mean_abs_ar) else float(mean_abs_ar),
                pct,
                topic_mean_abs_ar,
                commentary_source,
                provider,
                model_name,
            )
    commentary = st.session_state.get(commentary_key)
    if commentary:
        summary_part, analysis_part = _split_commentary(commentary)
        if summary_part:
            st.markdown(f"**요약**\n\n{summary_part}")
        st.info(f"**분석**\n\n{analysis_part}")
    elif provider == "none":
        st.info(
            "위 버튼은 AI 서비스를 선택해야 눌립니다 — 왼쪽 사이드바 'AI 요약 설정'에서 제미나이·그록·올라마 중 "
            "하나를 선택해주세요. 지금은 AI 요약 없이 원본 분석 결과에서 바로 계산한 수치만 보여주는 상태입니다."
        )

    _render_price_context_chart(row_ticker, row.get("event_date"), widget_scope=widget_scope)


def _render_cluster_members(row: pd.Series, selected_event_id, widget_scope: str = "event") -> None:
    size_value = pd.to_numeric(row.get("cluster_size"), errors="coerce")
    size = int(size_value) if pd.notna(size_value) else 1
    if size <= 1:
        return
    texts = _json_list(row.get("member_texts_raw_json"))
    urls = _json_list(row.get("member_source_urls_json"))
    times = _json_list(row.get("member_posted_at_et_json"))
    with st.expander(_scoped_label(f"연속 게시물 {size}개 확인", widget_scope), expanded=False):
        st.caption("같은 인물·관련 시장·주제에 해당하고 첫 게시물 이후 6시간 안에 올라온 글을 하나의 사건으로 묶었습니다.")
        for index in range(size):
            timestamp = _safe_text(times[index] if index < len(times) else None, fallback="시각 미상")
            text_value = _safe_text(texts[index] if index < len(texts) else None, fallback="(원문 없음)")
            st.markdown(f"**게시물 {index + 1} · {timestamp}**")
            st.write(text_value)
            url = urls[index] if index < len(urls) else None
            if isinstance(url, str) and url.strip():
                st.markdown(f"[원문 열기]({url.strip()})")
            if index < size - 1:
                st.divider()


@st.cache_data(show_spinner=False)
def _cached_article_markdown(url: str) -> str:
    return fetch_article_markdown(url)


@st.cache_data(show_spinner=False)
def _cached_translate_article(text: str, provider: str, model_name: str | None) -> str:
    return translate_to_korean(text, provider=provider, model_name=model_name)


def _render_full_article(source_url: str, event_id, widget_scope: str = "event"):
    article_key = f"article_text_{event_id}"
    if st.button("기사 본문 가져오기", key=f"{widget_scope}_fetch_article_{event_id}"):
        with st.spinner("기사 원문을 불러오는 중..."):
            try:
                st.session_state[article_key] = _cached_article_markdown(source_url)
            except Exception as exc:
                st.warning(f"기사 원문을 불러오지 못했습니다: {exc}")
                st.session_state.pop(article_key, None)
    article_md = st.session_state.get(article_key, "")
    if not article_md:
        st.caption("기사 수집은 자동 실행하지 않습니다. 필요할 때만 버튼을 눌러 불러옵니다.")
        return None
    if not article_md.strip():
        st.warning("기사 원문을 가져오지 못했습니다. 위 링크로 직접 열어보세요.")
        return None
    provider, model_name = _current_llm()
    with st.expander(_scoped_label("기사 원문 확인", widget_scope), expanded=True):
        st.caption("기사 링크에서 가져온 본문입니다. 사이트의 수집 제한에 따라 일부만 표시될 수 있습니다.")
        preview = article_md[:4000]
        st.markdown(preview + ("..." if len(article_md) > 4000 else ""))
        translated_key = f"article_translation_{event_id}_{provider}_{model_name}"
        if st.button(
            "기사 한국어 번역",
            key=f"{widget_scope}_translate_article_{event_id}",
            disabled=provider == "none",
        ):
            with st.spinner("기사를 한국어로 번역하는 중... (원문이 길면 시간이 걸립니다)"):
                st.session_state[translated_key] = _cached_translate_article(preview, provider, model_name)
        if translated_key in st.session_state:
            st.markdown("**한국어 번역**")
            st.markdown(st.session_state[translated_key])
    return article_md


def _render_price_context_chart(ticker: str, event_date, widget_scope: str = "event") -> None:
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
            go.Scatter(x=window["date"], y=window["ma5"], mode="lines", line=dict(color="#475569", width=1.5), name="5일 이동평균")
        )
    fig.add_vline(x=event_date, line=dict(color="#2563eb", width=1.5, dash="dot"))
    fig.add_annotation(x=event_date, y=1, yref="paper", text="사건 거래일", showarrow=False, yshift=10, font=dict(color="#1d4ed8", size=11))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="가격($)",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#334155"),
    )
    fig.update_xaxes(gridcolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb")

    with st.expander(_scoped_label(f"{_ticker_label(ticker)} 전후 주가 흐름", widget_scope), expanded=True):
        typical_text = f"약 ±{typical_move:.1f}%" if typical_move is not None else "계산 불가"
        st.caption(
            f"빨간 봉은 상승, 파란 봉은 하락입니다. 점선은 사건과 연결된 거래일, 회색 선은 5일 이동평균입니다. "
            f"직전 60거래일을 기준으로 하루 움직임의 중앙값은 {typical_text}입니다."
        )
        chart_date = pd.Timestamp(event_date).strftime("%Y%m%d")
        st.plotly_chart(fig, use_container_width=True, key=f"{widget_scope}_price_context_{ticker}_{chart_date}")


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
    event_id: str,
    row_dict: dict,
    n: int,
    mean_abs_ar,
    pct,
    topic_mean_abs_ar=None,
    article_text=None,
    provider: str = "ollama",
    model_name: str | None = None,
) -> str:
    ticker_stats = {
        "n": n,
        "mean_abs_ar": f"{mean_abs_ar:.2%}" if mean_abs_ar is not None else "-",
        "percentile": pct,
        "topic_mean_abs_ar": f"{topic_mean_abs_ar:.2%}" if topic_mean_abs_ar is not None else None,
    }
    return generate_event_commentary(
        row_dict,
        ticker_stats,
        content_text=article_text,
        provider=provider,
        model_name=model_name,
    )


def render_overview(events: pd.DataFrame, stats: dict, placebo_summary: dict) -> None:
    clean = events[events.get("contamination_level", pd.Series(dtype=str)).eq("CLEAN")] if not events.empty else events
    event_dates = pd.to_datetime(events.get("event_date"), errors="coerce") if not events.empty else pd.Series(dtype="datetime64[ns]")
    cols = st.columns(4)
    cols[0].metric("분석한 사건", f"{len(events):,}건")
    cols[1].metric("원인 비교적 명확", f"{len(clean):,}건")
    cols[2].metric("관련 시장", f"{events['ticker'].nunique():,}개" if "ticker" in events else "0개")
    period_text = "-"
    if event_dates.notna().any():
        period_text = f"{event_dates.min():%Y.%m}~{event_dates.max():%Y.%m}"
    cols[3].metric("분석 기간", period_text)

    if events.empty:
        st.info("분석할 사건 자료가 없습니다. 먼저 데이터 분석을 실행해야 합니다.")
        return

    timeline = _decorate_events(events).dropna(subset=["event_date", "impact_score"]).copy()
    if not timeline.empty:
        timeline["날짜"] = _as_datetime(timeline["날짜"])
        timeline["시장 대비 움직임"] = timeline.get("시장 대비 등락률", 0).abs()
        fig = px.scatter(
            timeline,
            x="날짜",
            y="반응 강도",
            color="인물",
            size="시장 대비 움직임",
            hover_name="사건 번호",
            hover_data={
                "주제": True,
                "관련 시장": True,
                "원인 구분": True,
                "시장 대비 등락률": ":+.2%",
                "내용": True,
                "시장 대비 움직임": False,
            },
            labels={
                "날짜": "사건 거래일",
                "반응 강도": "반응 강도 점수",
                "시장 대비 등락률": "시장 대비 등락률",
            },
            color_discrete_map={"일론 머스크": "#7c3aed", "도널드 트럼프": "#f97316"},
        )
        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend_title_text="인물",
        )
        st.subheader("사건별 시장 반응 분포")
        st.caption("점이 위에 있을수록 가격·거래량·변동성이 평소보다 크게 움직인 사건입니다. 점의 크기는 시장 대비 움직임의 절대 크기를 나타냅니다.")
        st.plotly_chart(fig, use_container_width=True)

    if placebo_summary:
        with st.expander("무관한 날짜와 비교한 점검 결과"):
            st.caption("실제 사건 날짜의 반응이 임의로 고른 날짜보다 특별했는지 확인하는 보조 점검입니다.")
            check_cols = st.columns(3)
            check_cols[0].metric("반복 비교 횟수", f"{int(placebo_summary.get('iterations', 0)):,}회")
            topic_share = pd.to_numeric(placebo_summary.get("placebo_h2_share_as_or_more_significant"), errors="coerce")
            person_share = pd.to_numeric(placebo_summary.get("placebo_h3_share_as_or_more_significant"), errors="coerce")
            check_cols[1].metric("주제 결과가 우연히 나온 비율", "-" if pd.isna(topic_share) else f"{topic_share * 100:.1f}%")
            check_cols[2].metric("인물 결과가 우연히 나온 비율", "-" if pd.isna(person_share) else f"{person_share * 100:.1f}%")
            st.caption("비율이 낮을수록 실제 사건 날짜의 결과가 임의 날짜에서는 드물게 나타났다는 뜻입니다.")

    top_table = _decorate_events(events).sort_values("반응 강도", ascending=False).head(15)
    top_table = top_table[["사건 번호", "날짜", "인물", "관련 시장", "주제", "시장 대비 등락률", "반응 강도", "원인 구분", "내용"]].copy()
    top_table["시장 대비 등락률"] = (top_table["시장 대비 등락률"] * 100).round(2)
    top_table = top_table.rename(columns={"시장 대비 등락률": "시장 대비 등락률(%)"})
    st.subheader("반응 강도가 컸던 사건")
    st.caption("가격·거래량·변동성이 평소보다 크게 움직인 사건 순서입니다. 점수가 높아도 해당 발언이 원인이라는 뜻은 아닙니다.")
    st.dataframe(top_table, use_container_width=True, hide_index=True, height=420)


def render_event_explorer(events: pd.DataFrame) -> None:
    if events.empty:
        st.info("분석할 사건 자료가 없습니다.")
        return
    st.markdown(
        '<div class="intro-panel"><b>이 페이지에서 확인할 수 있는 내용</b><br>'
        "원하는 인물·주제·관련 시장을 골라 사건을 좁힌 뒤, 각 게시물 이후 실제 등락률과 "
        "시장 대비 움직임을 비교할 수 있습니다. 목록에서 사건을 선택하면 원문과 전후 주가가 이어서 표시됩니다.</div>",
        unsafe_allow_html=True,
    )

    display = _decorate_events(events)
    filter_cols = st.columns(4)
    people = ["전체"] + sorted(display["person"].dropna().astype(str).unique().tolist())
    tickers = ["전체"] + sorted(display["ticker"].dropna().astype(str).unique().tolist())
    topics = ["전체"] + sorted(display["topic"].dropna().astype(str).unique().tolist())
    causes = ["전체"] + sorted(display["contamination_level"].dropna().astype(str).unique().tolist())
    with filter_cols[0]:
        person_choice = st.selectbox("인물", people, format_func=lambda value: value if value == "전체" else _person_label(value))
    with filter_cols[1]:
        ticker_choice = st.selectbox("관련 시장", tickers, format_func=lambda value: value if value == "전체" else _ticker_label(value))
    with filter_cols[2]:
        topic_choice = st.selectbox("게시물 주제", topics, format_func=lambda value: value if value == "전체" else _topic_label(value))
    with filter_cols[3]:
        cause_choice = st.selectbox("원인 구분 가능성", causes, format_func=lambda value: value if value == "전체" else _contam_label(value))

    filtered = display.copy()
    if person_choice != "전체":
        filtered = filtered[filtered["person"].astype(str).eq(person_choice)]
    if ticker_choice != "전체":
        filtered = filtered[filtered["ticker"].astype(str).eq(ticker_choice)]
    if topic_choice != "전체":
        filtered = filtered[filtered["topic"].astype(str).eq(topic_choice)]
    if cause_choice != "전체":
        filtered = filtered[filtered["contamination_level"].astype(str).eq(cause_choice)]

    search_cols = st.columns([2, 1])
    with search_cols[0]:
        keyword = st.text_input("게시물 내용 검색", placeholder="찾고 싶은 단어나 문구를 입력하세요")
    with search_cols[1]:
        sort_choice = st.selectbox("정렬 기준", ["최근 사건부터", "시장 대비 움직임이 큰 순", "반응 강도가 큰 순"])
    if keyword.strip():
        searchable = (
            filtered.get("text_raw", pd.Series(index=filtered.index, dtype=str)).fillna("").astype(str)
            + " " + filtered.get("text_clean", pd.Series(index=filtered.index, dtype=str)).fillna("").astype(str)
            + " " + filtered.get("description", pd.Series(index=filtered.index, dtype=str)).fillna("").astype(str)
        )
        filtered = filtered[searchable.str.contains(keyword.strip(), case=False, na=False, regex=False)]

    if sort_choice == "시장 대비 움직임이 큰 순":
        filtered = filtered.assign(_sort=filtered["시장 대비 등락률"].abs()).sort_values("_sort", ascending=False)
    elif sort_choice == "반응 강도가 큰 순":
        filtered = filtered.sort_values("반응 강도", ascending=False)
    else:
        filtered = filtered.sort_values("_날짜정렬", ascending=False)

    clear_count = int(filtered.get("contamination_level", pd.Series(dtype=str)).eq("CLEAN").sum())
    mean_abs = pd.to_numeric(filtered.get("시장 대비 등락률"), errors="coerce").abs().mean()
    metric_cols = st.columns(3)
    metric_cols[0].metric("검색된 사건", f"{len(filtered):,}건")
    metric_cols[1].metric("원인 비교적 명확", f"{clear_count:,}건")
    metric_cols[2].metric("평균 시장 대비 움직임", "-" if pd.isna(mean_abs) else f"{mean_abs * 100:.2f}%")

    if filtered.empty:
        st.info("선택한 조건에 해당하는 사건이 없습니다. 조건을 줄이거나 검색어를 바꿔보세요.")
        return

    table = filtered[[
        "사건 번호", "날짜", "인물", "관련 시장", "주제", "실제 등락률",
        "시장 대비 등락률", "반응 강도", "원인 구분", "내용",
    ]].copy()
    table["실제 등락률"] = (table["실제 등락률"] * 100).round(2)
    table["시장 대비 등락률"] = (table["시장 대비 등락률"] * 100).round(2)
    table = table.rename(columns={"실제 등락률": "실제 등락률(%)", "시장 대비 등락률": "시장 대비 등락률(%)"})
    st.dataframe(
        table,
        use_container_width=True,
        height=470,
        hide_index=True,
        column_config={
            "내용": st.column_config.TextColumn("게시물 또는 사건 내용", width="large"),
            "반응 강도": st.column_config.NumberColumn("반응 강도", format="%.2f"),
            "실제 등락률(%)": st.column_config.NumberColumn("실제 등락률(%)", format="%+.2f"),
            "시장 대비 등락률(%)": st.column_config.NumberColumn("시장 대비 등락률(%)", format="%+.2f"),
        },
    )

    event_ids = filtered["event_id"].astype(str).tolist()
    label_map = {
        str(row["event_id"]): (
            f"{row['사건 번호']} | {row['날짜']} | {row['인물']} | {row['관련 시장']} | "
            f"{row['주제']} | 시장 대비 {_format_pct(row['시장 대비 등락률'])}"
        )
        for _, row in filtered.iterrows()
    }
    selected_id = st.selectbox("자세히 볼 사건", event_ids, format_func=lambda value: label_map.get(str(value), str(value)))
    if selected_id:
        _render_event_detail(events, selected_id, widget_scope="explorer")


def _adjusted_p_value(stats: dict, key: str):
    test = stats.get("tests", {}).get(key, {})
    return test.get("p_value_fdr_bh", test.get("p_value"))


def _test_supported(stats: dict, key: str):
    test = stats.get("tests", {}).get(key, {})
    return test.get("reject_fdr_0.05")


def _p_value_text(value) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "계산하지 못함"
    return "0.001 미만" if number < 0.001 else f"{number:.3f}"


def _result_card(question: str, status: str, explanation: str) -> None:
    st.markdown(
        '<div class="result-card">'
        f'<div class="result-question">{html.escape(question)}</div>'
        f'<div class="result-status">{html.escape(status)}</div>'
        f'<div class="result-copy">{html.escape(explanation)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_hypotheses(stats: dict, events: pd.DataFrame) -> None:
    if not stats.get("tests"):
        st.info("연구 질문을 확인할 통계 결과가 없습니다.")
        return

    st.markdown(
        '<div class="intro-panel"><b>먼저 확인할 핵심 결론</b><br>'
        "전체 자료만 보면 인물과 주제에 따라 반응 차이가 있는 것처럼 보입니다. 그러나 같은 시장 안에서 다시 비교하면 "
        "그 차이가 뚜렷하지 않았습니다. 따라서 발언한 사람이나 주제보다, 어떤 시장과 연결된 사건인지가 결과 차이에 "
        "더 크게 작용했을 가능성이 있습니다.</div>",
        unsafe_allow_html=True,
    )

    h1 = _test_supported(stats, "h1_volatility_before_after")
    _result_card(
        "게시물 전후로 주가 변동성이 달라졌나?",
        "뚜렷한 변화를 확인하지 못함" if h1 is False else "변화가 확인됨" if h1 else "판단할 자료 부족",
        "현재 표본에서는 게시물 전후의 변동성이 일관되게 달라졌다고 보기 어렵습니다."
        if h1 is False else "게시물 전후의 변동성 차이가 우연만으로 보기 어려운 수준이었습니다.",
    )

    topic_all = _test_supported(stats, "h2_topic_difference")
    topic_within = [
        _test_supported(stats, key) for key in [
            "h2b_topic_within_ticker_QQQ", "h2b_topic_within_ticker_SPY", "h2b_topic_within_ticker_TSLA"
        ]
    ]
    if topic_all and not any(value is True for value in topic_within):
        topic_status = "전체에서는 차이가 보였지만, 같은 시장 안에서는 뚜렷하지 않음"
        topic_copy = "주제별 차이로 보였던 결과가 실제로는 테슬라·나스닥 100·S&P 500의 원래 변동성 차이를 반영했을 수 있습니다."
    else:
        topic_status = "주제별 차이가 확인됨" if topic_all else "주제별 차이를 확인하지 못함"
        topic_copy = "같은 시장 안에서도 주제에 따른 차이가 반복되는지 함께 확인한 결과입니다."
    _result_card("게시물 주제에 따라 시장 반응 크기가 달랐나?", topic_status, topic_copy)

    person_all = _test_supported(stats, "h3_musk_vs_trump")
    person_same_market = _test_supported(stats, "h3b_musk_vs_trump_within_qqq")
    if person_all and person_same_market is False:
        person_status = "전체에서는 차이가 보였지만, 같은 시장에서는 뚜렷하지 않음"
        person_copy = "일론 머스크 사건은 테슬라와 연결된 비중이 높아 전체 비교만으로 인물 자체의 영향이라고 단정하기 어렵습니다."
    else:
        person_status = "인물별 차이가 확인됨" if person_all else "인물별 차이를 확인하지 못함"
        person_copy = "두 인물을 같은 시장에서 비교한 결과까지 함께 반영했습니다."
    _result_card("일론 머스크와 도널드 트럼프의 반응 크기가 달랐나?", person_status, person_copy)

    engagement = _test_supported(stats, "h4_engagement_correlation")
    _result_card(
        "반응 수가 많은 게시물일수록 주가도 크게 움직였나?",
        "뚜렷한 관련성을 확인하지 못함" if engagement is False else "관련성이 확인됨" if engagement else "판단할 자료 부족",
        "좋아요·공유 등 게시물 반응 수가 많다는 이유만으로 시장 반응도 컸다고 보기는 어렵습니다.",
    )

    role = _test_supported(stats, "trump_role_difference")
    _result_card(
        "도널드 트럼프의 후보·시민 시기와 대통령 시기가 달랐나?",
        "뚜렷한 차이를 확인하지 못함" if role is False else "시기별 차이가 확인됨" if role else "판단할 자료 부족",
        "대통령 시기 표본이 적기 때문에 차이가 없다고 확정하기보다, 현재 자료에서는 확인하지 못했다고 해석해야 합니다.",
    )

    clean = events[events.get("contamination_level", pd.Series(index=events.index, dtype=str)).eq("CLEAN")].copy()
    if not clean.empty and {"topic", "abnormal_return"}.issubset(clean.columns):
        topic_summary = (
            clean.assign(_abs_move=pd.to_numeric(clean["abnormal_return"], errors="coerce").abs() * 100)
            .groupby("topic")
            .agg(사건수=("topic", "size"), 중앙값=("_abs_move", "median"))
            .reset_index()
        )
        topic_summary = topic_summary[topic_summary["사건수"] >= 5].sort_values("중앙값")
        if not topic_summary.empty:
            topic_summary["주제"] = topic_summary["topic"].map(_topic_label)
            fig = px.bar(topic_summary, x="중앙값", y="주제", orientation="h", text="사건수")
            fig.update_traces(marker_color="#3b82f6", texttemplate="%{text}건", textposition="outside")
            fig.update_layout(
                height=max(330, len(topic_summary) * 34),
                margin=dict(l=10, r=40, t=20, b=10),
                xaxis_title="시장 대비 움직임의 중앙값(%)",
                yaxis_title=None,
                showlegend=False,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
            )
            st.subheader("주제별 시장 반응 크기")
            st.caption("다른 사건과 비교적 덜 겹친 사례만 사용한 단순 비교입니다. 관련 시장 구성이 다르므로 이 그래프만으로 주제 효과를 단정하면 안 됩니다.")
            st.plotly_chart(fig, use_container_width=True)

    detail_rows = []
    core_keys = [
        "h1_volatility_before_after",
        "h2_topic_difference",
        "h2b_topic_within_ticker_QQQ",
        "h2b_topic_within_ticker_SPY",
        "h2b_topic_within_ticker_TSLA",
        "h3_musk_vs_trump",
        "h3b_musk_vs_trump_within_qqq",
        "h4_engagement_correlation",
        "trump_role_difference",
    ]
    for key in core_keys:
        label = TEST_QUESTION_LABELS[key]
        value = stats.get("tests", {}).get(key, {})
        supported = value.get("reject_fdr_0.05")
        detail_rows.append({
            "확인한 질문": label,
            "결론": "차이 확인" if supported is True else "뚜렷한 차이 없음" if supported is False else "표본 부족",
            "보정 유의확률": _p_value_text(_adjusted_p_value(stats, key)),
        })
    with st.expander("검정 수치 자세히 보기"):
        st.caption("여러 질문을 동시에 확인할 때 우연히 차이가 있다고 나올 가능성을 보정한 값을 사용했습니다. 일반적으로 0.05보다 작으면 우연만으로 보기 어려운 차이로 판단합니다.")
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)


def render_method_checks(data: dict, events: pd.DataFrame) -> None:
    st.markdown(
        '<div class="intro-panel"><b>왜 원인 구분이 필요한가?</b><br>'
        "게시물과 같은 시점에 다른 게시물, 경제 일정, 시장 전체 충격이 함께 있었다면 주가 움직임을 한 발언의 영향으로 "
        "해석하기 어렵습니다. 아래 분류는 각 사건을 얼마나 조심해서 읽어야 하는지 보여줍니다.</div>",
        unsafe_allow_html=True,
    )
    if not events.empty and "contamination_level" in events:
        counts = events["contamination_level"].fillna("정보 없음").map(
            lambda value: _contam_label(value) if value != "정보 없음" else value
        ).value_counts().reset_index()
        counts.columns = ["원인 구분 가능성", "사건 수"]
        fig = px.bar(counts, x="원인 구분 가능성", y="사건 수", text="사건 수")
        fig.update_traces(marker_color="#3b82f6", textposition="outside")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    explanation_cols = st.columns(3)
    explanation_cols[0].markdown("**원인 비교적 명확**\n\n같은 시점의 다른 주요 요인이 적어 사건과 시장 반응을 비교하기 상대적으로 수월합니다.")
    explanation_cols[1].markdown("**다른 요인 일부 있음**\n\n다른 게시물이나 일정이 일부 겹쳐 결과 해석에 주의가 필요합니다.")
    explanation_cols[2].markdown("**원인 구분 어려움**\n\n여러 요인이 함께 발생해 특정 게시물의 영향이라고 보기 어렵습니다.")

    available_checks = []
    if not data.get("topic_audit_summary", pd.DataFrame()).empty:
        available_checks.append("게시물 주제 분류 정확도 점검")
    if not data.get("rivn_sensitivity", pd.DataFrame()).empty:
        available_checks.append("비교 종목을 바꾼 민감도 점검")
    if not data.get("placebo_results", pd.DataFrame()).empty:
        available_checks.append("무관한 날짜와 비교한 반복 점검")
    if available_checks:
        st.subheader("추가로 수행된 점검")
        for check in available_checks:
            st.markdown(f"- {check}")
        st.caption("세부 수치는 분석 결과 파일에 저장되어 있으며, 이 화면에서는 초보자가 바로 해석할 수 있는 핵심 결과만 보여줍니다.")


def render_ask_data_design(stats: dict, events: pd.DataFrame) -> None:
    st.subheader("분석 결과에 질문하기")
    st.caption(
        "질문에 포함된 인물·관련 시장·주제·원인 구분 조건을 먼저 찾아 실제 사건 자료를 계산합니다. "
        "AI는 계산된 결과를 읽기 쉽게 설명하며 미래 주가 예측이나 투자 조언은 하지 않습니다."
    )
    provider, model_name = _current_llm()
    history = st.session_state.setdefault("data_chat_history_v2", [])

    prompt_columns = st.columns(4)
    suggestions = [
        "반응이 가장 컸던 사건 5개",
        "일론 머스크와 도널드 트럼프 비교",
        "테슬라에서 원인이 비교적 명확한 사건 요약",
        "이 프로젝트의 핵심 결론",
    ]
    pending_question = None
    for column, suggestion in zip(prompt_columns, suggestions):
        with column:
            if st.button(suggestion, key=f"suggestion_{suggestion}", use_container_width=True):
                pending_question = suggestion

    top_bar_left, top_bar_right = st.columns([5, 1])
    with top_bar_left:
        st.caption(f"AI 설명 기능: {PROVIDER_LABELS.get(provider, provider)}")
    with top_bar_right:
        if st.button("대화 지우기", use_container_width=True):
            history.clear()
            st.rerun()

    for item in history:
        with st.chat_message(item.get("role", "assistant")):
            st.write(item.get("content", ""))
            evidence = item.get("evidence")
            if item.get("role") == "assistant" and evidence:
                with st.expander("답변에 사용한 계산 근거"):
                    st.caption(" · ".join(evidence.get("filters", [])) or "전체 데이터")
                    rows = evidence.get("top_events", [])
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    typed_question = st.chat_input(
        "예: 도널드 트럼프의 관세 관련 사건 중 반응이 가장 컸던 것은?",
        key="full_data_chat_input",
    )
    question = typed_question or pending_question
    if not question:
        if not history:
            st.info("위 예시 질문을 누르거나 직접 질문해보세요.")
        return

    history.append({"role": "user", "content": question})
    context = _build_chat_context(events, stats, question, history[:-1])
    if provider == "none":
        answer = _deterministic_chat_answer(question, context)
    else:
        with st.spinner("데이터를 필터링하고 답변을 생성하는 중..."):
            answer = _cached_ask_data_answer(question, context, provider, model_name)
        if answer.startswith("판단보류:"):
            answer += "\n\nAI 연결 없이 계산 결과만 안내합니다.\n\n" + _deterministic_chat_answer(question, context)
    history.append(
        {
            "role": "assistant",
            "content": answer,
            "evidence": {
                "filters": context.get("질문에_적용한_필터", []),
                "top_events": context.get("반응강도_상위사건", [])[:5],
            },
        }
    )
    st.rerun()


def _render_page_header(title: str, description: str) -> None:
    st.markdown(
        f'<div class="hero-title">{html.escape(title)}</div>'
        f'<div class="hero-copy">{html.escape(description)}</div>',
        unsafe_allow_html=True,
    )


def render_track2_news_page(events: pd.DataFrame) -> None:
    _render_page_header(
        "2025년 4월 이후 뉴스 사건",
        "기존 SNS 게시물 자료가 끝난 이후의 주요 사건을 뉴스 근거와 함께 확인하는 화면입니다. 기사 본문·한국어 번역·AI 분석은 필요한 사건에서만 실행됩니다.",
    )
    render_live_trump_feed("case_study")
    track2 = events[
        events.get("track", pd.Series(index=events.index, dtype=str)).astype(str).eq("track2_manual")
    ].copy()
    if track2.empty:
        st.info("등록된 2025년 4월 이후 뉴스 사건이 없습니다.")
        return
    decorated = _decorate_events(events)
    track2 = decorated[decorated.get("track", pd.Series(index=decorated.index, dtype=str)).astype(str).eq("track2_manual")].copy()
    track2["event_date_sort"] = pd.to_datetime(track2["event_date"], errors="coerce")
    track2 = track2.sort_values("event_date_sort", ascending=False)
    labels = [
        f"{row.get('사건 번호')} | {row.get('날짜')} | {row.get('인물')} | {row.get('관련 시장')} | {row.get('주제')}"
        for _, row in track2.iterrows()
    ]
    selected_label = st.selectbox("분석할 뉴스 사건", labels)
    selected_event_id = track2.iloc[labels.index(selected_label)]["event_id"]
    _render_event_detail(events, selected_event_id, widget_scope="case_study")


def render_case_studies(events: pd.DataFrame, figures: list) -> None:
    """원본 케이스 스터디 기능과 새 뉴스 사건 분석을 한 화면에 유지합니다."""
    news_tab, intraday_tab = st.tabs(["2025년 4월 이후 뉴스", "장중 분석 자료"])
    with news_tab:
        render_track2_news_page(events)
    with intraday_tab:
        _render_page_header(
            "장중 케이스 분석",
            "대표 사건의 분봉 분석 결과가 생성되어 있으면 여기에서 확인합니다. 일봉으로 원인을 구분하기 어려운 장중 게시물은 분봉 결과를 우선 참고해야 합니다.",
        )
        intraday = [path for path in figures if "intraday" in path.name.lower()]
        if intraday:
            for path in intraday:
                st.markdown(f"- [{path.name}]({path.as_posix()})")
        else:
            st.info("아직 생성된 장중 케이스 분석 파일이 없습니다. `run_intraday_case_study.py`를 실행하면 결과가 여기에 표시됩니다.")


def render_statistics_page(data: dict, events: pd.DataFrame, stats: dict) -> None:
    _render_page_header(
        "연구 결과",
        "이 프로젝트가 확인하려 했던 질문과 그 결론을 쉬운 표현으로 정리했습니다. 전체 자료의 차이와 같은 시장 안에서의 차이를 구분해 과도한 해석을 피합니다.",
    )
    overview_tab, hypothesis_tab, method_tab = st.tabs(["전체 결과", "연구 질문별 결론", "분석 신뢰도"])
    with overview_tab:
        render_overview(events, stats, data["placebo_summary"])
    with hypothesis_tab:
        render_hypotheses(stats, events)
    with method_tab:
        render_method_checks(data, events)


def main() -> None:
    _inject_app_styles()
    data = cached_data()
    events = data["events"]
    stats = data["stats"]

    with st.sidebar:
        st.markdown("### AI 기능 설정")
        st.caption("번역·요약·질문 답변에 사용할 서비스를 선택합니다.")
    _render_llm_settings()

    st.markdown('<div class="app-title">Who Moves the Market?</div>', unsafe_allow_html=True)
    tabs = st.tabs(
        ["실시간 메인", "개요", "이벤트 탐색기", "가설 검증", "방법론 점검", "케이스 스터디", "Ask the Data"]
    )

    with tabs[0]:
        _render_page_header(
            "SNS 발언 이후 시장은 어떻게 움직였을까?",
            "일론 머스크와 도널드 트럼프의 게시물·발언 이후 테슬라, 나스닥 100, S&P 500이 실제로 얼마나 움직였는지 확인합니다.",
        )
        period_start, period_end = _render_analysis_period(events)
        if period_start is not None and not events.empty:
            event_dates_all = pd.to_datetime(events.get("event_date"), errors="coerce")
            events_scope = events.loc[(event_dates_all.dt.date >= period_start) & (event_dates_all.dt.date <= period_end)]
        else:
            events_scope = events
        st.markdown(
            '<div class="intro-panel"><b>이 프로그램은 무엇을 보여주나요?</b><br>'
            "게시물이나 발언이 나온 시점과 다음 거래일의 가격·거래량·변동성을 연결해, 평소보다 큰 움직임이 있었는지 비교합니다. "
            "같은 시점의 다른 뉴스도 주가에 영향을 줄 수 있으므로 발언이 주가를 움직였다고 단정하지 않으며, 미래 주가도 예측하지 않습니다.</div>",
            unsafe_allow_html=True,
        )
        summary_cols = st.columns(4)
        clean_count = int(events_scope.get("contamination_level", pd.Series(dtype=str)).eq("CLEAN").sum()) if not events_scope.empty else 0
        period_text = f"{period_start:%Y.%m.%d}~{period_end:%Y.%m.%d}" if period_start is not None else "-"
        summary_cols[0].metric("분석한 사건", f"{len(events_scope):,}건")
        summary_cols[1].metric("분석 기간", period_text)
        summary_cols[2].metric("관련 시장", f"{events_scope['ticker'].nunique() if 'ticker' in events_scope else 0:,}개")
        summary_cols[3].metric("원인 비교적 명확", f"{clean_count:,}건")
        render_live_main(events, data["daily_prices"], stats, period_start=period_start, period_end=period_end)

    with tabs[1]:
        _render_page_header(
            "전체 분석 결과",
            "분석한 사건의 범위와 시장 반응 분포, 반응 강도가 컸던 사건을 먼저 살펴봅니다.",
        )
        render_overview(events, stats, data["placebo_summary"])

    with tabs[2]:
        _render_page_header(
            "사건별 주가 반응 찾기",
            "관심 있는 인물·주제·관련 시장을 선택해 사건을 찾고, 각 사건의 원문과 실제 주가 반응을 자세히 확인합니다.",
        )
        render_event_explorer(events)

    with tabs[3]:
        _render_page_header(
            "연구 질문별 결론",
            "가설 번호와 통계표부터 보여주지 않고, 이 프로젝트가 실제로 확인한 질문과 결론을 먼저 설명합니다.",
        )
        render_hypotheses(stats, events)

    with tabs[4]:
        _render_page_header(
            "분석 신뢰도 점검",
            "같은 시점의 다른 게시물과 경제 일정이 겹쳤는지 확인하고, 결과를 어느 정도까지 해석할 수 있는지 보여줍니다.",
        )
        render_method_checks(data, events)

    with tabs[5]:
        render_case_studies(events, data["figures"])

    with tabs[6]:
        _render_page_header(
            "분석 결과에 질문하기",
            "질문에서 조건을 찾아 실제 사건 자료를 먼저 계산한 뒤, 그 결과를 근거와 함께 설명합니다.",
        )
        render_ask_data_design(stats, events)


if __name__ == "__main__":
    main()
