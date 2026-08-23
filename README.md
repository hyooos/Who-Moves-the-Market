# 📈 Who Moves the Market?

### 누가 시장을 움직이는가? : 트럼프, 일론머스크 SNS 데이터 기반 주가 반응 분석
> 도널드 트럼프와 일론 머스크의 트위터 게시물과 금융시장 데이터를 같은 시간축에 놓고 어떤 게시물 전후에 가격·거래량·변동성이 평소보다 크게 반응했는지 분석하는 프로젝트

---

## Overview

본 프로젝트는 Trump·Musk의 SNS 게시물(Truth Social/X)과 TSLA·QQQ·SPY 등 종목·지수의 일봉 가격 데이터를 결합해, **robust z-score 기반 이벤트 스터디**로 "게시물 직후 시장이 이례적으로 반응했는가"를 검증합니다.

Track1(2023-01~2025-04, 통계적 가설검정)과 Track2(2025-06 결별·2025-10 관세 등 캐글 범위 밖 개별 사건 케이스 스터디)를 이원 구조로 병행하며, placebo 순열검정·FDR 다중검정 보정·ticker 교란 통제 보조검정까지 코드 레벨에서 방법론을 스스로 검증하도록 설계했습니다. 검증 과정에서 발견한 두 개의 핵심 버그(SPY 자기 자신을 시장 벤치마크로 쓴 오류, topic 키워드 단어 경계 누락)를 직접 고치고 그 전후 결과를 투명하게 공개하는 것도 이 프로젝트의 일부입니다.

---

## Key Features

- Trump(Truth Social)·Musk(X) SNS 게시물 12만 건+ 정제 및 topic 룰베이스 분류
- Robust z-score(median/MAD) 기반 이벤트 스터디, 2-pass baseline(CLEAN 이벤트 날짜 제외)
- 다중게시·매크로(FOMC)·시장충격 3중 오염 분류 → CLEAN 표본만 정식 가설검정
- H1~H6 가설검정(Wilcoxon/Kruskal-Wallis/Mann-Whitney/Spearman) + Benjamini-Hochberg FDR 보정
- Placebo 순열검정(200회) + RIVN peer 민감도 분석으로 방법론 자체 검증
- ticker 교란 통제 보조검정 + η² 효과크기 분해로 "진짜 원인" 재검증
- topic 분류 정확도 직접 라벨링·평가(`audit_topics.py`) — 키워드 룰 기반 분류의 한계를 수치로 공개
- Track2 케이스 스터디: 로컬 LLM(Ollama+Qwen2.5:7b)으로 사실 기반 내러티브 생성 + 사람 검수 레이어
- Streamlit 대시보드: 실시간 반응강도 게이지, 가격 그라데이션 차트 클릭 조회, 가설검증표
- 실시간 게시물 필터(`live_monitor.py`) — 가격 예측이 아닌 규칙 기반 관심 알림
- UTC 원본 시각 보존 → 미국 동부시간(ET) 변환 → 장 마감 후/휴장일은 다음 실제 거래일로 정렬
- 원본 게시물 ID·URL을 최종 이벤트까지 보존해 대시보드에서 원문으로 바로 이동
- 같은 화자·ticker·topic·반응 거래일의 연속 게시물을 첫 글 기준 6시간 고정 창으로 묶어 pseudo-replication 방지

---

## Tech Stack

#### Language

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" height="28"/>
</p>

---

#### Data Processing & Analysis

<p>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/statsmodels-3776AB?style=flat-square" height="28"/>
</p>

---

#### Machine Learning & NLP

<p>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" height="28"/>
  <img src="https://img.shields.io/badge/Twitter--RoBERTa-1DA1F2?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/Ollama-000000?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/Qwen2.5-6A1B9A?style=flat-square" height="28"/>
</p>

---

#### Market Data & Statistics

<p>
  <img src="https://img.shields.io/badge/yfinance-8A2BE2?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/Robust%20Z--Score-1E88E5?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/FDR%20(Benjamini--Hochberg)-43A047?style=flat-square" height="28"/>
  <img src="https://img.shields.io/badge/Placebo%20Test-5E35B1?style=flat-square" height="28"/>
</p>

---

#### Visualization & Dashboard

<p>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" height="28"/>
  <img src="https://img.shields.io/badge/Jinja2-B41717?style=flat-square&logo=jinja&logoColor=white" height="28"/>
</p>

---

## Pipeline

1. `load_posts.py`: 서로 다른 Musk/Trump CSV 스키마를 공통 형식으로 바꾸고 원본 ID·URL·UTC/ET 시각을 보존합니다.
2. `preprocess.py` + `topic_rules.py`: 분석 기간과 시장 관련 키워드를 적용하고 topic·연결 ticker를 정합니다.
3. `event_windows.py`: ET 기준 09:30 이전은 당일, 09:30~16:00은 당일(부분 일봉 경고), 16:00 이후·주말·휴장일은 다음 거래일에 연결합니다.
4. `event_clustering.py`: 같은 화자·ticker·topic·반응 거래일의 글을 첫 게시물 기준 6시간 고정 창으로 묶고 모든 원문·URL을 보존합니다.
5. `impact.py` + `contamination.py`: 사건별 가격 반응 점수와 별도 사건/FOMC/시장충격 중첩 여부를 계산합니다.
6. `stats.py`: CLEAN 사건 표본으로 가설검정과 FDR 보정을 수행합니다.
7. `plots.py` + `report.py` + `dashboard_app.py`: 차트·HTML 리포트·클릭 가능한 Streamlit 화면을 만듭니다.

세부 파일 역할과 고도화 순서는 [`PROJECT_GUIDE_KO.md`](PROJECT_GUIDE_KO.md)를 참고하세요.

### 시간 정렬 원칙

| ET 게시 시점 | 일봉 이벤트 거래일 | 해석 품질 |
| --- | --- | --- |
| 거래일 09:30 이전 | 같은 거래일 | 장 시작 후 반응을 대체로 포함 |
| 거래일 09:30~16:00 | 같은 거래일 | 게시 전 당일 움직임이 섞여 분봉 분석 권장 |
| 거래일 16:00 이후 | 다음 거래일 | 다음 정규장 반응에 연결 |
| 주말·휴장일 | 다음 거래일 | 다음 정규장 반응에 연결 |

`posted_at`은 기존 코드 호환용 timezone-naive UTC이며, 의미가 명확한 `posted_at_utc`, `posted_at_et`, `market_session`, `event_date_rule`, `daily_alignment_quality`를 함께 저장합니다. 미국 조기폐장일은 현재 16:00 마감을 사용하므로 후속 버전에서 거래소 캘린더 기반으로 보완할 예정입니다.

### 사건 clustering 원칙

일봉 하나에 같은 캠페인의 게시물 여러 개가 연결되면 같은 가격 반응을 여러 번 세는 문제가 생깁니다. 기본 분석은 `person + ticker + topic + event_date`가 같고 첫 게시물로부터 6시간 이내인 글을 하나의 사건으로 묶습니다. 직전 글과의 간격이 아니라 첫 글을 기준으로 잡아 cluster가 연쇄적으로 며칠까지 늘어나는 것을 막습니다. `--no-cluster-posts`로 기존 게시물 단위 결과를 재현하거나 `--cluster-hours`로 민감도를 비교할 수 있습니다.

---

## Key Findings

아래 표는 시간대·clustering 수정 전 게시물 단위 CLEAN 표본(359건)에서 나온 역사적 baseline입니다. 새 6시간 사건 clustering 결과로 전체 파이프라인을 다시 실행한 뒤 수치를 교체해야 합니다.
전체 판정 근거와 재해석 과정은 [`docs/final_report.md`](../docs/final_report.md) §6-2를 참고하세요.

| 가설 | 내용 | 최종 판정 |
| --- | --- | --- |
| H1 | 게시 전후 변동성 증가 | ❌ 기각 |
| H2 | topic별 반응 차이 | ❌ 기각(ticker 교란으로 재해석) |
| H3 | 직결/비직결 기업 반응 차이 | ❌ 기각(ticker 교란으로 재해석, 문제의식은 유지) |
| H4 | 감성(긍정/부정) 비대칭 | ❌ 기각(raw p<0.05, FDR 보정 후 비유의) |
| H5 | 단기 집중·소멸 | ⏸ 판단 보류(일봉 해상도 한계) |
| H6 | 정치 권력 확정 여부 | ❌ 기각(대통령 시기 표본 n=8로 검정력 부족) |

### Main Findings

- ticker(TSLA/QQQ/SPY) 하나가 절대 초과수익률 분산의 27.2%를 설명 — topic이 추가로 설명하는 몫은 **0.3%p**뿐
- SPY를 SPY 자기 자신과 비교하던 벤치마크 버그를 발견·수정 — 이후 통계 결과가 훨씬 보수적인 방향으로 수렴
- topic 키워드가 단어 경계 없이 매칭되던 버그 발견(`ai`가 "said" 안에서 88% 오탐) — 수정 후 market-relevant 게시물 6,185건 → 3,015건으로 절반 가까이 감소, 오히려 CLEAN 표본은 268→359건으로 증가
- 자체 topic 분류 정확도를 직접 라벨링해 검증 — 전체 정확도 56%, Musk 쪽 topic(precision 100%) vs Trump 쪽 topic(precision 0~40%대)의 격차를 수치로 공개
- 다중게시를 "신호"로 볼지 "오염"으로 볼지 두 관점을 직접 실험 — 신호 관점은 pseudo-replication(같은 날짜 관측치 최대 42회 중복)으로 허위 유의성을 양산함을 확인
- Track2 케이스 스터디에서 2025-10-10 관세 발표가 확장 기간 전체 최대 반응(SPY z=-5.49)으로 확인됨

---

## Directory Structure

```bash
.
├── market_mover/                  # 핵심 로직 패키지
│   ├── config.py                  # 경로·기간·티커 등 전역 설정
│   ├── load_posts.py              # SNS 게시물 로딩·정제
│   ├── load_prices.py             # yfinance 가격 다운로드 + 캐시 폴백
│   ├── preprocess.py              # market-relevant 필터링
│   ├── topic_rules.py             # 키워드 기반 topic 분류·종목 매핑
│   ├── event_windows.py           # 이벤트 정렬(Track1/Track2)
│   ├── event_clustering.py        # 연속 게시물 6시간 사건 clustering
│   ├── impact.py                  # robust z-score, impact_score
│   ├── contamination.py           # 오염 분류(CLEAN/MINOR/MAJOR)
│   ├── novelty.py / sentiment.py  # novelty score / 감성분석(선택)
│   ├── stats.py                   # H1~H6 통계검정 + FDR 보정
│   ├── placebo.py / sensitivity.py# 순열검정 / RIVN 민감도
│   ├── case_narratives.py         # Track2 LLM 내러티브(Ollama+Qwen2.5:7b)
│   ├── plots.py / report.py       # Plotly 차트 / HTML 리포트
│   └── dashboard_data.py / dashboard_widgets.py  # 대시보드 로더·게이지
├── run_daily_pipeline.py          # 메인 실행 파일
├── audit_topics.py                # topic 분류 정확도 검증
├── live_monitor.py                # 실시간 게시물 필터
├── find_track2_news_candidates.py # Track2 뉴스 발견 도구
├── run_intraday_case_study.py     # 분봉 케이스 스터디
├── dashboard_app.py                # Streamlit 대시보드
├── data/
│   ├── raw/                       # Kaggle 원본 CSV (직접 채워야 함, 미포함)
│   ├── manual/                    # Track2 수동 이벤트, FOMC 캘린더 등
│   ├── interim/ processed/        # 파이프라인 중간·최종 산출물
├── outputs/{tables,figures,reports}/  # 통계·차트·리포트 산출물
├── requirements.txt / requirements-optional.txt
└── README.md
```

---

## Run

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

감성분석·대시보드 등 선택 기능까지 쓰려면:

```bash
pip install -r requirements-optional.txt
```

Kaggle에서 받은 게시물 CSV를 `data/raw/`에 넣어주세요(파일명에 `musk`/`elon`/`trump`/`donald`가 포함되면 자동 인식).

### 전체 파이프라인 실행

```bash
PYTHONPATH=. .venv/bin/python run_daily_pipeline.py \
  --add-sentiment --add-novelty --run-placebo --run-rivn-sensitivity \
  --build-narratives --llm-provider ollama --llm-model qwen2.5:7b
```

옵션 없이 최소 실행만 하려면:

```bash
.venv/bin/python run_daily_pipeline.py
```

### Streamlit Dashboard

```bash
PYTHONPATH=. .venv/bin/streamlit run dashboard_app.py
```

대시보드는 원본의 상단 메뉴 구성을 살린 `실시간 메인 / 개요 / 이벤트 탐색기 / 가설 검증 / 방법론 점검 / 케이스 스터디 / Ask the Data`로 구성됩니다. 실시간 메인에는 계기판과 일론 머스크·도널드 트럼프 캐릭터를 유지하고, 주가 차트 오른쪽에 작은 챗봇을 함께 배치했습니다. 전체 Ask the Data 메뉴도 별도로 유지합니다. 내부 식별자는 숨기고 화면용 사건 번호와 한국어 범주명을 사용합니다. 사건 점을 선택하면 원문, 한국어 번역, 실제 가격 반응, 전후 캔들 차트, AI 요약·분석을 확인할 수 있습니다.

번역·기사 수집·AI 분석은 화면 진입 시 자동 실행하지 않습니다. 필요한 버튼을 눌렀을 때만 한 번 실행하고 같은 입력은 캐시하므로, 기존처럼 사건을 클릭할 때마다 모든 작업을 다시 호출하지 않습니다.

AI provider는 왼쪽 `AI 설정`에서 선택합니다.

```bash
# Gemini
export GEMINI_API_KEY="..."

# Groq
export GROQ_API_KEY="..."

# 로컬 Ollama
ollama serve
ollama pull qwen2.5:7b
```

API key가 없더라도 가격 차트·필터·집계·상위 사건 조회는 동작합니다. 데이터 챗봇은 pandas가 먼저 계산하며, LLM 연결이 없으면 계산형 기본 답변으로 전환됩니다.

### Topic 분류 검증

```bash
.venv/bin/python audit_topics.py --sample-size 50
# data/manual/topic_audit_sample.csv의 manual_topic을 채운 뒤
.venv/bin/python audit_topics.py --evaluate
```

각 옵션(감성분석/novelty/placebo/RIVN 민감도/Track2 내러티브/장중 케이스)의 상세 사용법은 아래 섹션과 [`docs/codebase_reference.md`](../docs/codebase_reference.md)를 참고하세요.

---

## Conclusion

이벤트 스터디 인프라(정렬·오염 분류·robust z-score)와 placebo 기반 방법론은 견고하게 작동했습니다. 다만 원래 유의했던 topic(H2)·인물(H3) 효과는 재검증 결과 종목(ticker) 자체의 변동성 차이로 설명되는 교란이었고, 이는 "SNS 발언 내용 자체"보다 "그 발언이 어떤 종목에 연결되는가"가 시장 반응 크기를 결정하는 더 근본적인 요인일 수 있음을 시사합니다.

이 결과가 "SNS 데이터가 시장과 무관하다"는 뜻은 아닙니다 — CLEAN 이벤트 정의 자체가 게시물 직후의 이례적 반응이라는 시간적 연관 위에 서 있고, Track2 케이스 스터디(2025-06 결별, 2025-10 관세 발표)는 그 극단적 사례를 구체적으로 보여줍니다. 부정된 것은 "그 반응의 크기가 발언 내용·화자에 따라 체계적으로 달라진다"는 더 좁고 구체적인 주장이며, 이 프로젝트는 그 주장을 검증하는 과정에서 스스로 두 개의 핵심 버그(벤치마크 오류, 키워드 매칭 오류)를 찾아 고쳤다는 점에서 방법론 검증 장치가 의도대로 작동했다고 판단합니다.

자세한 배경·선행연구·전체 결과는 [`docs/final_report.md`](../docs/final_report.md)에서 확인할 수 있습니다.

---

## Notice

본 프로젝트는 투자 조언이 아닙니다. 여기서 다루는 상관관계는 인과관계를 의미하지 않으며, 실제 투자 판단에 사용해서는 안 됩니다.

---

<details>
<summary><b>세부 실행 가이드 (감성분석 / novelty / placebo / RIVN / Track2 백필 / 장중 케이스)</b></summary>

## 실행 준비

이 폴더에는 가상환경(`.venv/`)이 포함돼 있지 않습니다. 받은 뒤 각자 새로 만들어야 합니다.

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

</details>
