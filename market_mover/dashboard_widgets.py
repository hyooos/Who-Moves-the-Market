"""대시보드 메인 비주얼(실시간 반응강도 게이지) 렌더링 유틸.

이 게이지는 가격을 예측하지 않는다. §5-3에서 topic/person 내용이 반응 크기를
거의 설명하지 못하고, ticker(종목)가 반응 크기 분산의 대부분을 설명한다는 걸
확인했기 때문에, 바늘은 "이 게시물이 매핑된 종목이 과거 CLEAN 이벤트에서
평균적으로 얼마나 크게 반응해왔는가"라는 회고적(retrospective) 사실만
가리킨다 — live_monitor.py와 동일한 설계 철학이다.

게이지 좌표계: t는 0(왼쪽 끝, 9시 방향)~1(오른쪽 끝, 3시 방향)의 위치를 나타내는
분수다. 표준 수학 각도로는 theta = 180*(1-t)도(0도=오른쪽, 90도=위, 180도=왼쪽)이며,
화면 좌표는 y축이 아래로 증가하므로 y = cy - r*sin(theta)로 뒤집어 그린다.

캐리커처는 실제 인물 사진이 아니라 직접 그린 SVG 일러스트다 — 실존 인물(공인) 사진을
합성·조작해서 움직이는 것처럼 보이게 하는 건 오인 소지가 있어 피하고, 대신 만화적
캐리커처로 같은 연출을 구현한다.
"""

import base64
import html
import math
from pathlib import Path

import pandas as pd

_CX, _CY, _R = 110, 110, 82

# 4등분 감정 캐리커처(사용자가 직접 그려서 제공한 만화 일러스트, 실제 사진 아님).
_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_EMOTION_FILES = {
    "Trump": ["trump_calm.png", "trump_annoyed.png", "trump_angry.png", "trump_furious.png"],
    "Musk": ["musk_calm.png", "musk_annoyed.png", "musk_angry.png", "musk_furious.png"],
}
_emotion_cache: dict = {}


def _emotion_data_uri(person: str, quadrant: int):
    files = _EMOTION_FILES.get(person)
    if not files:
        return None
    quadrant = max(0, min(3, quadrant))
    cache_key = (person, quadrant)
    if cache_key not in _emotion_cache:
        path = _ASSETS_DIR / files[quadrant]
        if not path.exists():
            _emotion_cache[cache_key] = None
        else:
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            _emotion_cache[cache_key] = f"data:image/png;base64,{b64}"
    return _emotion_cache[cache_key]


def compute_ticker_gauges(events: pd.DataFrame, tickers=("QQQ", "SPY", "TSLA"), start_date=None, end_date=None) -> list:
    """QQQ/SPY/TSLA 각각에 대해, 지정한 기간(start_date~end_date) 안에서 "그 종목에
    매핑된 가장 최근 게시물"의 영향점수가 그 종목의 과거 CLEAN 표본에서 몇 번째
    백분위인지 계산한다. 종목별로 독립적으로 계산하므로 세 게이지를 나란히 놓고
    바로 비교할 수 있다.

    Track2(수동 등록 사건)는 "impact_score가 상위권인 사건"만 골라 넣은 표본이라
    (§4-3 채택 기준) 항상 극단값에 가깝다 — 백분위 자체는 그대로 계산하되, 뽑힌
    "최신"이 Track2일 경우 결과에 is_curated=True를 표시해 대시보드가 "이건 미리
    알려진 대형 사건이라 일반적인 크기가 아니다"라는 경고를 붙일 수 있게 한다.
    """
    results = []
    if events.empty or "ticker" not in events.columns:
        return [_empty_gauge_state(t) for t in tickers]

    clean = events[events.get("contamination_level", pd.Series(dtype=str)).eq("CLEAN")]

    candidates = events.copy()
    if start_date is not None or end_date is not None:
        posted = pd.to_datetime(candidates["posted_at"], errors="coerce")
        if start_date is not None:
            candidates = candidates[posted >= pd.Timestamp(start_date)]
            posted = pd.to_datetime(candidates["posted_at"], errors="coerce")
        if end_date is not None:
            candidates = candidates[posted <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    for ticker in tickers:
        same_ticker_all = candidates[candidates["ticker"] == ticker].dropna(subset=["posted_at"]).copy()
        if same_ticker_all.empty:
            results.append(_empty_gauge_state(ticker))
            continue
        same_ticker_all["posted_at"] = pd.to_datetime(same_ticker_all["posted_at"], errors="coerce")
        same_ticker_all = same_ticker_all.dropna(subset=["posted_at"]).sort_values("posted_at")
        if same_ticker_all.empty:
            results.append(_empty_gauge_state(ticker))
            continue
        latest = same_ticker_all.iloc[-1]
        is_curated = str(latest.get("track")) == "track2_manual"

        clean_ticker = clean[clean["ticker"] == ticker] if "ticker" in clean.columns else clean.iloc[0:0]
        same_ticker_scores = clean_ticker["impact_score"].dropna() if "impact_score" in clean_ticker else pd.Series(dtype=float)
        impact_score = latest.get("impact_score")
        if len(same_ticker_scores) and pd.notna(impact_score):
            pct = float((same_ticker_scores < impact_score).mean())
        else:
            pct = None

        text_preview = str(latest.get("text_clean") or latest.get("description") or "")
        abnormal_return = latest.get("abnormal_return")
        results.append(
            {
                "has_data": True,
                "ticker": ticker,
                "t": pct if pct is not None else 0.5,
                "percentile": pct,
                "impact_score": impact_score,
                "abnormal_return": abnormal_return,
                "person": latest.get("person"),
                "topic": latest.get("topic"),
                "posted_at": str(latest.get("posted_at")),
                "text_preview": text_preview,
                "clean_n": len(clean_ticker),
                "is_curated": is_curated,
            }
        )
    return results


def _empty_gauge_state(ticker: str) -> dict:
    return {
        "has_data": False,
        "ticker": ticker,
        "t": 0.5,
        "percentile": None,
        "impact_score": None,
        "abnormal_return": None,
        "person": None,
        "topic": None,
        "posted_at": None,
        "text_preview": "",
        "clean_n": 0,
        "is_curated": False,
    }


def _flatten(html_str: str) -> str:
    """줄마다 4칸 이상 들여쓰기가 남으면 마크다운이 <pre> 코드블록으로 오인해서
    HTML을 그대로 렌더링하지 못한다(Streamlit st.markdown이 내부적으로 이 규칙을 따름).
    그래서 각 줄의 선행 공백을 제거하고 한 줄로 합쳐 안전하게 만든다."""
    return " ".join(line.strip() for line in html_str.splitlines() if line.strip())


def _point(theta_deg: float, r: float, cx: float = _CX, cy: float = _CY) -> tuple:
    theta = math.radians(theta_deg)
    return cx + r * math.cos(theta), cy - r * math.sin(theta)


# 감정 캐리커처(calm/annoyed/angry/furious)와 맞춰 게이지도 4등분한다.
_GAUGE_ZONES = [
    (0.00, "#3b82f6", "낮음"),
    (0.25, "#facc15", "보통"),
    (0.50, "#f97316", "큰 편"),
    (0.75, "#ef4444", "매우 큼"),
]


def render_single_gauge_html(state: dict) -> str:
    """종목 1개짜리 소형 게이지. 세 개를 나란히 놓고 한눈에 비교하는 용도."""
    ticker = html.escape(str(state.get("ticker") or "-"))

    if not state.get("has_data"):
        return _flatten(f"""
        <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border-radius:14px;
                    padding:16px;color:#94a3b8;text-align:center;font-size:13px;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
          <div style="font-weight:700;color:#f1f5f9;margin-bottom:6px;">{ticker}</div>
          선택하신 기간 안에는 이 종목({ticker})에 매핑된 게시물이 없습니다 — 기간을 넓혀보세요.
        </div>
        """)

    t = state["t"]
    zones_svg = []
    bounds = _GAUGE_ZONES + [(1.0, None, None)]
    for i in range(len(_GAUGE_ZONES)):
        t0, color, zone_label = bounds[i]
        t1 = bounds[i + 1][0]
        theta_start = 180 * (1 - t0)
        theta_end = 180 * (1 - t1)
        x1, y1 = _point(theta_start, _R)
        x2, y2 = _point(theta_end, _R)
        zones_svg.append(
            f'<path d="M {_CX} {_CY} L {x1:.1f} {y1:.1f} A {_R} {_R} 0 0 1 {x2:.1f} {y2:.1f} Z" '
            f'fill="{color}" opacity="0.32"/>'
        )

    ticks_svg = []
    for i in range(13):
        theta = 180 - (180 * i / 12)
        is_major = i % 3 == 0
        r1 = _R + 2
        r2 = _R + (10 if is_major else 5)
        x1, y1 = _point(theta, r1)
        x2, y2 = _point(theta, r2)
        ticks_svg.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{"#334155" if is_major else "#94a3b8"}" stroke-width="{2 if is_major else 1}"/>'
        )

    rotate_deg = 180 * t - 90
    needle_len = _R - 12

    person = html.escape(str(state.get("person") or "-"))
    topic = html.escape(str(state.get("topic") or "-"))
    posted_at = html.escape(str(state.get("posted_at") or "-"))
    text_preview = html.escape(state.get("text_preview") or "")
    pct = state.get("percentile")
    pct_txt = f"{pct*100:.0f}%" if isinstance(pct, (int, float)) and pd.notna(pct) else "-"
    clean_n = state.get("clean_n", 0)

    # 게이지는 "얼마나 크게 반응했는가"(크기)만 나타내고 방향(상승/하락)은 안 담고
    # 있어서 헷갈린다는 피드백 반영 — 방향은 실제 등락률 부호로 별도 배지를 만든다.
    ar = state.get("abnormal_return")
    if isinstance(ar, (int, float)) and pd.notna(ar):
        if ar > 0:
            direction_badge = (
                f'<span style="background:#7f1d1d;color:#fecaca;font-size:13px;font-weight:700;'
                f'padding:3px 10px;border-radius:999px;">📈 상승 +{ar*100:.1f}%</span>'
            )
        elif ar < 0:
            direction_badge = (
                f'<span style="background:#1e3a8a;color:#bfdbfe;font-size:13px;font-weight:700;'
                f'padding:3px 10px;border-radius:999px;">📉 하락 {ar*100:.1f}%</span>'
            )
        else:
            direction_badge = '<span style="color:#94a3b8;font-size:13px;">보합</span>'
    else:
        direction_badge = '<span style="color:#94a3b8;font-size:12px;">방향 계산 불가</span>'
    curated_badge = (
        '<div style="margin-top:6px;font-size:11px;background:#fef3c7;color:#92400e;'
        'padding:3px 8px;border-radius:6px;">⚠ 미리 알려진 대형 사건(수동 선정) — '
        "일반적인 크기가 아닐 수 있음</div>"
        if state.get("is_curated")
        else ""
    )

    # 게이지를 4등분해서(0~25/25~50/50~75/75~100%) 그 게시물 인물의 감정 캐리커처로
    # 지금 어떤 구간인지 한눈에 보이게 한다.
    emotion_html = ""
    person_name = str(state.get("person"))
    if person_name in ("Trump", "Musk") and isinstance(pct, (int, float)) and pd.notna(pct):
        quadrant = min(3, int(pct * 4))
        uri = _emotion_data_uri(person_name, quadrant)
        if uri:
            emotion_html = f'<img src="{uri}" width="40" height="40" style="border-radius:50%;border:2px solid #334155;" />'

    svg_h = _CY + 46

    return _flatten(f"""
<div style="background:linear-gradient(135deg,#1e293b,#0f172a);border-radius:14px;
            padding:16px;color:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <div style="display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:2px;">
    <div style="font-weight:800;font-size:16px;">{ticker}</div>
    {emotion_html}
  </div>
  <div style="position:relative;width:{_CX*2}px;height:{svg_h}px;margin:0 auto;">
    <svg viewBox="0 0 {_CX*2} {svg_h}" width="{_CX*2}" height="{svg_h}">
      <path d="M {_CX-_R-14} {_CY} A {_R+14} {_R+14} 0 0 1 {_CX+_R+14} {_CY} L {_CX+_R+14} {_CY+4}
               L {_CX-_R-14} {_CY+4} Z" fill="#f8fafc" opacity="0.06"/>
      {''.join(zones_svg)}
      {''.join(ticks_svg)}
      <circle cx="{_CX}" cy="{_CY}" r="{_R+2}" fill="none" stroke="#334155" stroke-width="1.5"/>
      <g style="transform-origin:{_CX}px {_CY}px;animation:needle-live 3.2s ease-in-out infinite;">
        <g style="transform-origin:{_CX}px {_CY}px;transform:rotate({rotate_deg:.1f}deg);">
          <polygon points="{_CX-3},{_CY} {_CX+3},{_CY} {_CX+1},{_CY-needle_len} {_CX-1},{_CY-needle_len}"
                   fill="#f87171"/>
        </g>
      </g>
      <circle cx="{_CX}" cy="{_CY}" r="7" fill="#1e293b" stroke="#f87171" stroke-width="2"/>
      <text x="{_CX}" y="{_CY+28}" text-anchor="middle" font-size="19" font-weight="800"
            fill="#f1f5f9">{pct_txt}</text>
    </svg>
  </div>
  <div style="font-size:11px;color:#94a3b8;text-align:center;margin-top:2px;">
    반응 크기가 과거 {clean_n}건 중 상위 {pct_txt}에 해당</div>
  <div style="text-align:center;margin-top:6px;">{direction_badge}</div>
  {curated_badge}
  <div style="font-size:12px;color:#cbd5e1;margin-top:8px;line-height:1.5;">
    <b>{person}</b> · {topic}<br/><span style="color:#94a3b8;">{posted_at}</span>
  </div>
  <div style="font-size:10.5px;color:#64748b;margin-top:8px;">📝 이 게시물 원문(번역, 요약 아님)</div>
  <div style="margin-top:2px;padding:8px 10px;background:#0f172a;border-radius:8px;
              font-size:12px;color:#e2e8f0;border:1px solid #334155;word-break:break-word;
              line-height:1.5;height:110px;overflow-y:auto;">
    {text_preview or '(원문 없음)'}
  </div>
</div>
<style>
@keyframes needle-live {{
  0%, 100% {{ transform: rotate(-1.5deg); }}
  50% {{ transform: rotate(1.5deg); }}
}}
</style>
""")


def render_mac_window_html(title: str, body_html: str) -> str:
    """맥OS 창(빨/노/초 트래픽라이트) 스타일 카드. body_html은 이미 안전하게
    이스케이프된 상태로 전달해야 한다(호출부에서 html.escape 처리)."""
    title_safe = html.escape(str(title or ""))
    return _flatten(f"""
    <div style="border-radius:14px;overflow:hidden;border:1px solid #e2e8f0;
                box-shadow:0 8px 24px rgba(0,0,0,0.08);background:#ffffff;margin-top:8px;">
      <div style="display:flex;align-items:center;gap:16px;padding:10px 14px;
                  background:linear-gradient(#f8fafc,#eef1f5);border-bottom:1px solid #e2e8f0;">
        <div style="display:flex;gap:7px;">
          <span style="width:12px;height:12px;border-radius:50%;background:#ff5f57;
                       display:inline-block;"></span>
          <span style="width:12px;height:12px;border-radius:50%;background:#febc2e;
                       display:inline-block;"></span>
          <span style="width:12px;height:12px;border-radius:50%;background:#28c840;
                       display:inline-block;"></span>
        </div>
        <div style="flex:1;text-align:center;font-size:12.5px;color:#64748b;
                    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                    padding-right:52px;">{title_safe}</div>
      </div>
      <div style="padding:18px 20px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                  color:#0f172a;font-size:14.5px;line-height:1.7;">
        {body_html}
      </div>
    </div>
    """)
