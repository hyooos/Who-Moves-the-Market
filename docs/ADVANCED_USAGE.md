# 세부 실행 가이드

감성분석 / novelty / placebo / RIVN 민감도 / Track2 LLM 내러티브 / Track2 백필 / 장중 케이스 / 실시간 모니터링 — [README.md](../README.md)의 Quick Start에 안 들어간 선택 옵션들의 실행 방법을 모았습니다.

## 실행 준비

이 저장소에는 가상환경(`.venv/`)이 포함돼 있지 않습니다. clone한 뒤 각자 새로 만들어야 합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

설치와 데이터가 준비됐는지 빠르게 확인하려면 스모크 테스트를 돌려보세요.

```bash
PYTHONPATH=. .venv/bin/python tests/smoke_test.py
```

## 일봉 전체 분석 실행

기본값은 2-pass 방식입니다. 먼저 전체 가격 기준으로 이벤트와 오염 수준을 만든 뒤, `CLEAN` 이벤트 날짜만 robust z-score baseline에서 제외해 impact score를 다시 계산합니다.

```bash
.venv/bin/python run_daily_pipeline.py --baseline-mode plain   # 기존 방식과 비교하고 싶을 때
.venv/bin/python run_daily_pipeline.py --freeze-date 2026-08-11  # 리포트 기준일 직접 지정
.venv/bin/python run_daily_pipeline.py --skip-prices             # 게시물 CSV 정제까지만
```

## 수동 입력 파일

`data/manual/fomc_calendar.csv`

```csv
date,event_type
2023-02-01,FOMC
```

`data/manual/track2_curated_events.csv`

```csv
event_id,person,posted_at,platform,topic,ticker,source_url,description,narrative_reviewed
tk2_001,Musk,2025-06-04T14:32:00,X,doge_budget_feud,TSLA,https://example.com,사건 설명,
```

## 감성분석: Twitter-RoBERTa

Track 2 설명을 써주는 LLM이 아니라, 게시물 텍스트를 `positive`/`neutral`/`negative`로 분류하는 감성분석 모델입니다(FinBERT의 대체재, Gemini/Groq/Ollama의 대체재 아님).

```bash
.venv/bin/pip install -r requirements-optional.txt
.venv/bin/python run_daily_pipeline.py --add-sentiment
```

결과 이벤트 테이블에 `sentiment_label`, `sentiment_score`, `sentiment_confidence`, `sentiment_model` 컬럼이 추가되고, 통계검정(Mann-Whitney, Spearman, η²)에도 자동 반영됩니다. 각 그룹 표본이 5개 미만이면 검정은 자동 생략됩니다.

## Novelty score

같은 인물이 최근 30일 안에 비슷한 말을 반복했는지 Jaccard 유사도로 계산합니다.

```bash
.venv/bin/python run_daily_pipeline.py --add-novelty
```

## Placebo 검정

실제 게시물 날짜 대신 무작위 거래일을 뽑아 같은 통계검정을 반복합니다(기본 200회).

```bash
.venv/bin/python run_daily_pipeline.py --run-placebo --placebo-iterations 200
```

결과: `outputs/tables/placebo_results.csv`, `outputs/tables/placebo_summary.json`

## RIVN 민감도 분석

TSLA abnormal return은 기본적으로 `GM`, `F`, `RIVN` peer 평균 대비로 계산합니다. RIVN 포함 여부가 결과를 바꾸는지 확인합니다.

```bash
.venv/bin/python run_daily_pipeline.py --run-rivn-sensitivity
```

결과: `outputs/tables/rivn_sensitivity.csv`

## Track2 LLM 내러티브

```bash
# Ollama 로컬 모델(권장 — 무료, 반복 실행에 안전)
ollama pull qwen2.5:7b
.venv/bin/python run_daily_pipeline.py --build-narratives --llm-provider ollama --llm-model qwen2.5:7b

# 또는 클라우드 API
export GEMINI_API_KEY="본인_API_키"
.venv/bin/python run_daily_pipeline.py --build-narratives --llm-provider gemini
```

프롬프트는 인과관계를 단정하지 않도록 설계되어 있습니다("~때문에 하락했다"가 아니라 "직후 이례적 반응이 관측됐다"). 기사 본문이나 가격 맥락이 부족하면 `판단보류`가 표시됩니다.

## Track2 자동 백필 후보 (twscrape, 선택·1회성)

```bash
.venv/bin/pip install -r requirements-optional.txt
export TWSCRAPE_USERNAME="계정" TWSCRAPE_PASSWORD="비밀번호" TWSCRAPE_EMAIL="이메일"
.venv/bin/python backfill_track2_musk_twscrape.py
```

결과는 `data/manual/track2_musk_backfill_candidates.csv`에 저장되는 "후보 목록"이며, 실제 채택할 사건만 확인해서 `track2_curated_events.csv`로 옮깁니다.

## 장중 케이스 스터디

```bash
.venv/bin/python run_intraday_case_study.py \
  --event-id tk1_000001 \
  --intraday-csv data/manual/example_intraday.csv
```

장중 CSV 컬럼: `datetime,ticker,open,high,low,close,volume`

## 실시간 모니터링

```bash
PYTHONPATH=. .venv/bin/python live_monitor.py --person Trump
PYTHONPATH=. .venv/bin/python live_monitor.py --person Musk --paste "텍스트 직접 입력"
```

가격을 예측하지 않고, topic→ticker 분류 규칙을 새 게시물에 그대로 적용해 보여주는 필터입니다.
