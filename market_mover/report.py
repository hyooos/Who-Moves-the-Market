from pathlib import Path

import pandas as pd
from jinja2 import Template

from . import config


REPORT_TEMPLATE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Market Mover 분석 리포트</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; line-height: 1.5; color: #202124; }
    h1, h2 { margin-bottom: 8px; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 14px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
    th { background: #f6f8fa; }
    .muted { color: #666; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    .metric { border: 1px solid #ddd; padding: 14px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>Market Mover 분석 리포트</h1>
  <p class="muted">분석 기준일: {{ freeze_date }}</p>

  <div class="grid">
    <div class="metric"><strong>전체 이벤트 수</strong><br>{{ n_total }}</div>
    <div class="metric"><strong>오염이 적은 이벤트 수</strong><br>{{ n_clean }}</div>
    <div class="metric"><strong>분석 방식</strong><br>일봉 Track 1 + 선택적 Track 2</div>
  </div>

  <h2>그룹별 표본 수</h2>
  {{ group_sizes_html }}

  <h2>통계검정 결과</h2>
  <p class="muted">FDR 보정: {{ fdr_rationale }}</p>
  {{ tests_html }}

  <h2>영향 점수 상위 이벤트</h2>
  {{ top_events_html }}

  <h2>Track 2 케이스 내러티브</h2>
  {{ narratives_html }}

  <h2>시각화</h2>
  <ul>
    {% for figure in figures %}
      <li><a href="../figures/{{ figure }}">{{ figure }}</a></li>
    {% endfor %}
  </ul>
</body>
</html>
"""


def _dict_table(data: dict) -> str:
    rows = []
    for group, values in data.items():
        if not values:
            rows.append({"그룹": _label(group), "구분": "(없음)", "표본 수": 0})
        else:
            for level, n in values.items():
                rows.append({"그룹": _label(group), "구분": _label(level), "표본 수": n})
    return pd.DataFrame(rows).to_html(index=False)


def _label(value) -> str:
    labels = {
        "person": "인물",
        "topic": "토픽",
        "ticker": "종목",
        "trump_role": "트럼프 지위",
        "candidate_or_citizen": "후보/시민",
        "president": "대통령",
        "CLEAN": "오염 낮음",
        "MINOR": "오염 일부",
        "MAJOR": "오염 큼",
        "impact_score": "영향 점수",
        "abnormal_return": "초과수익률",
        "text_clean": "게시물 내용",
        "narrative": "케이스 설명(LLM 원문)",
        "narrative_reviewed": "케이스 설명(사람 검수)",
        "description": "수동 설명",
        "sentiment_label": "감성 라벨",
        "sentiment_score": "감성 점수",
        "sentiment_confidence": "감성 신뢰도",
        "sentiment_model": "감성 모델",
        "novelty_score": "Novelty 점수",
        "max_prior_similarity": "최대 과거 유사도",
        "event_id": "이벤트 ID",
        "person": "인물",
        "event_date": "이벤트 거래일",
        "topic": "토픽",
        "ticker": "종목",
    }
    return labels.get(value, str(value))


def _tests_table(stats_results: dict) -> str:
    rows = []
    for key, value in stats_results.get("tests", {}).items():
        rows.append(
            {
                "가설": key,
                "검정": value.get("test"),
                "통계량": value.get("statistic"),
                "p-value": value.get("p_value"),
                "FDR 보정 p-value": value.get("p_value_fdr_bh"),
                "FDR 0.05 유의": value.get("reject_fdr_0.05"),
                "상태": value.get("status"),
            }
        )
    return pd.DataFrame(rows).to_html(index=False)


def write_report(
    events: pd.DataFrame,
    stats_results: dict,
    figure_dir: Path,
    report_path: Path,
    freeze_date: str = None,
) -> None:
    top_cols = [
        "event_id",
        "person",
        "event_date",
        "topic",
        "ticker",
        "sentiment_label",
        "sentiment_score",
        "novelty_score",
        "impact_score",
        "abnormal_return",
        "text_clean",
    ]
    top_cols = [col for col in top_cols if col in events.columns]
    top_events = events.dropna(subset=["impact_score"]).nlargest(20, "impact_score")[top_cols]
    top_events = top_events.rename(columns={col: _label(col) for col in top_events.columns})
    narrative_cols = ["event_id", "person", "posted_at", "topic", "ticker", "description", "narrative", "narrative_reviewed"]
    narrative_cols = [col for col in narrative_cols if col in events.columns]
    narratives = events[events["track"].eq("track2_manual")][narrative_cols].copy() if narrative_cols else pd.DataFrame()
    narratives = narratives.rename(columns={col: _label(col) for col in narratives.columns})
    figures = sorted(path.name for path in figure_dir.glob("*.html"))
    html = Template(REPORT_TEMPLATE).render(
        freeze_date=freeze_date or config.FREEZE_DATE,
        n_total=stats_results.get("n_total", 0),
        n_clean=stats_results.get("n_clean", 0),
        group_sizes_html=_dict_table(stats_results.get("group_sizes", {})),
        tests_html=_tests_table(stats_results),
        fdr_rationale=stats_results.get("multiple_testing_rationale", ""),
        top_events_html=top_events.to_html(index=False),
        narratives_html=narratives.to_html(index=False) if not narratives.empty else "<p>Track 2 내러티브가 아직 없습니다.</p>",
        figures=figures,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
