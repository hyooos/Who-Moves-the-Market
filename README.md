# Market Mover 분석 파이프라인

머스크와 트럼프의 게시물이 주식/지수 움직임과 어떤 관련이 있는지 Python에서만 분석하는 이벤트 스터디 프로젝트입니다. Tableau로 옮기지 않고, CSV 표와 Plotly HTML 차트, 최종 HTML 리포트를 모두 Python에서 생성합니다.

## 구현 범위

- Track 1: 전체 기간 게시물 기반 일봉 이벤트 스터디
- Track 2: 주요 사건을 직접 정리한 케이스 스터디 이벤트 병합
- Track 2 선택 백필: `twscrape`로 Musk 2025-04-14 이후 후보 이벤트를 한 번만 보조 수집
- Track 2 선택 내러티브: Jina Reader와 Gemini/Groq/Ollama 중 선택한 LLM으로 뉴스 근거 기반 설명 생성
- 선택 감성분석: FinBERT 대신 SNS 문체에 더 맞는 Twitter-RoBERTa 사용
- 장중 분석: 선택한 사건에 대해 분봉/시간봉 데이터가 있을 때 발언 전후 움직임 시각화
- 가격 데이터 폴백: yfinance가 실패하면 `data/manual/price_cache.csv` 사용
- Placebo 검정: 실제 이벤트 날짜 대신 무작위 거래일을 넣어 방법론이 가짜 신호에도 반응하는지 확인
- Topic 분류 검증: 수동 라벨링 표본 CSV 생성 및 룰베이스 분류 정확도 평가
- 이벤트 제외 baseline: 1차 이벤트/오염 분류 후 CLEAN 이벤트 날짜를 robust z-score 60일 baseline에서 제외하는 2-pass 구조
- 전체 가설 FDR 보정: 개별 p-value와 Benjamini-Hochberg 보정 p-value를 함께 저장
- H2 사후검정: Kruskal-Wallis 이후 topic pairwise Dunn's test(FDR 보정)
- RIVN 민감도 분석: TSLA peer 그룹에서 RIVN 포함/제외 결과 비교
- Novelty 검정: 반복 발언이 아닌 새로운 발언일수록 시장 반응이 다른지 검정
- 일봉 감쇠곡선: 이벤트 전후 `D-3 ~ D+3` 평균 절대 초과수익률 시각화
- 결과물: 분석 테이블, 통계검정 결과, HTML 차트, 최종 HTML 리포트

## 폴더 구조

이 폴더는 원본 프로젝트에서 **실행에 필요한 파일만 뽑아온 공유용 사본**입니다. `.venv/`(가상환경)와 `data/raw/*.csv`(90MB+ 원본, 용량 문제로 미포함), 설계 문서(`docs/`)는 빠져 있습니다. 원본 CSV와 설계 문서가 필요하면 상위 폴더(`market_mover/`)를 함께 받으세요.

```text
market_mover/       핵심 로직 패키지 (config, stats, impact 등 17개 모듈)
run_daily_pipeline.py       메인 실행 파일
run_intraday_case_study.py  장중 케이스 스크립트
audit_topics.py             topic 분류 검증 스크립트
backfill_track2_musk_twscrape.py  Track 2 후보 보조 수집 스크립트
dashboard_app.py            Streamlit 대시보드
tests/                      스모크 테스트
requirements.txt / requirements-optional.txt  패키지 목록
data/raw/                   Kaggle 원본 게시물 CSV를 직접 넣어야 하는 자리(비어 있음)
data/manual/                FOMC 일정, Track 2 수동 사건 CSV 템플릿
data/interim/                정제된 게시물과 중간 산출물 (실행 후 생성)
data/processed/             최종 점수화된 이벤트 테이블 (실행 후 생성)
outputs/tables/             분석용 CSV/JSON 결과 (실행 후 생성)
outputs/figures/            Plotly HTML 차트 (실행 후 생성)
outputs/reports/            최종 HTML 리포트 (실행 후 생성)
```

## 파일 구성

처음 이 코드를 보는 조원용 요약입니다.

### 실행 스크립트 (루트)

| 파일 | 역할 |
| --- | --- |
| `run_daily_pipeline.py` | 메인 실행 파일. 게시물 로딩부터 리포트 생성까지 전체 파이프라인을 실행합니다. |
| `run_intraday_case_study.py` | 이벤트 1건을 골라 분봉 데이터로 장중 반응 차트를 만듭니다(선택). |
| `audit_topics.py` | 룰베이스 topic 분류가 얼마나 정확한지 수동 라벨링으로 검증합니다(선택). |
| `backfill_track2_musk_twscrape.py` | twscrape로 Musk 최신 게시물 후보를 보조 수집합니다(선택, 1회성). |
| `dashboard_app.py` | Streamlit 대시보드 실행 파일입니다. |
| `tests/smoke_test.py` | 합성 데이터로 파이프라인이 안 깨지는지 확인하는 테스트입니다. |

### `market_mover/` 패키지 (핵심 로직)

| 파일 | 역할 |
| --- | --- |
| `config.py` | 경로, 분석 기간, 대상 티커 등 전역 설정값 |
| `load_posts.py` | Kaggle 게시물 CSV 로딩·정제·인물/플랫폼 자동 인식 |
| `load_prices.py` | yfinance 가격 다운로드 + 수동 캐시 폴백 |
| `preprocess.py` | 분석 기간 필터링 + topic 분류 + market-relevant 필터링 |
| `topic_rules.py` | 키워드 기반 topic 분류기와 topic→종목 매핑 |
| `event_windows.py` | 게시물을 다음 거래일 이벤트로 정렬, 장중 윈도우 생성 |
| `impact.py` | robust z-score, 초과수익률, impact_score 계산 |
| `contamination.py` | 다중게시/FOMC/시장충격 오염 플래그 및 등급 분류 |
| `novelty.py` | 최근 30일 대비 발언 novelty(Jaccard 유사도) 점수 |
| `sentiment.py` | (선택) Twitter-RoBERTa 감성분석 |
| `stats.py` | H1~H4 가설 통계검정, FDR 보정, 사후검정, 효과크기 |
| `placebo.py` | 무작위 날짜로 재검정하는 placebo(순열) 테스트 |
| `sensitivity.py` | TSLA peer(RIVN 포함/제외) 민감도 분석 |
| `case_narratives.py` | (선택) Track 2 이벤트 LLM 근거 내러티브 생성 |
| `plots.py` | Plotly HTML 차트 생성 |
| `report.py` | Jinja2 기반 최종 HTML 리포트 생성 |
| `dashboard_data.py` | Streamlit 대시보드용 산출물(CSV/JSON) 로더 |

## 실행 준비

이 폴더에는 가상환경(`.venv/`)이 포함돼 있지 않습니다. 받은 뒤 각자 새로 만들어야 합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kaggle에서 받은 CSV 파일을 `data/raw/` 폴더에 넣어주세요(현재 비어 있음). 파일명에는 `musk`, `elon`, `trump`, `donald` 중 하나가 들어가면 자동으로 인식합니다.

설치와 데이터가 준비됐는지 빠르게 확인하려면 스모크 테스트를 돌려보세요. `tests/smoke_test.py`는 직접 실행하면 `market_mover` 모듈을 못 찾을 수 있으니 프로젝트 루트를 `PYTHONPATH`에 넣어서 실행합니다.

```bash
PYTHONPATH=. .venv/bin/python tests/smoke_test.py
```

## 일봉 전체 분석 실행

```bash
.venv/bin/python run_daily_pipeline.py
```

기본값은 2-pass 방식입니다. 먼저 전체 가격 기준으로 이벤트와 오염 수준을 만든 뒤, `CLEAN` 이벤트 날짜만 robust z-score baseline에서 제외해 impact score를 다시 계산합니다. 기존 방식과 비교하고 싶으면:

```bash
.venv/bin/python run_daily_pipeline.py --baseline-mode plain
```

리포트에 표시할 분석 기준일을 직접 지정하려면:

```bash
.venv/bin/python run_daily_pipeline.py --freeze-date 2026-08-11
```

Track 2 수동 사건에 대해 뉴스 기반 설명까지 생성하려면 Gemini API 키를 환경변수로 넣고 실행합니다.

```bash
export GEMINI_API_KEY="본인_API_키"
.venv/bin/python run_daily_pipeline.py --build-narratives
```

Groq를 쓰고 싶다면:

```bash
export GROQ_API_KEY="본인_API_키"
.venv/bin/python run_daily_pipeline.py --build-narratives --llm-provider groq
```

Ollama 로컬 모델을 쓰고 싶다면 먼저 로컬에서 모델을 띄운 뒤 실행합니다.

```bash
ollama run llama3.1:8b
.venv/bin/python run_daily_pipeline.py --build-narratives --llm-provider ollama
```

가격 데이터를 받기 전에 게시물 CSV 컬럼만 먼저 확인하고 싶다면:

```bash
.venv/bin/python run_daily_pipeline.py --skip-prices
```

## 수동 입력 파일

`data/manual/fomc_calendar.csv`

```csv
date,event_type
2023-02-01,FOMC
```

`data/manual/track2_curated_events.csv`

```csv
event_id,person,posted_at,platform,topic,ticker,source_url,description
tk2_001,Musk,2025-06-04T14:32:00,X,doge_budget_feud,TSLA,https://example.com,사건 설명
```

## 생성되는 결과물

- `data/interim/posts_cleaned.csv`: 정제 후 market-relevant로 남은 게시물
- `data/interim/events_daily.csv`: 일봉 기준 이벤트 테이블
- `data/processed/events_scored.csv`: impact score까지 붙은 최종 이벤트 테이블
- `outputs/tables/stats_results.json`: 통계검정 결과
- `outputs/tables/case_narratives.csv`: Track 2 케이스 설명
- `outputs/figures/*.html`: Plotly 시각화
- `outputs/reports/market_mover_report.html`: 최종 분석 리포트

## Python 대시보드

Tableau 없이 Python 안에서 결과를 탐색하려면 Streamlit 대시보드를 사용합니다. 대시보드 설계 의도와 화면 구성은 원본 프로젝트(`market_mover/docs/dashboard_blueprint.md`)에 정리되어 있습니다.

먼저 분석 결과를 생성합니다.

```bash
.venv/bin/python run_daily_pipeline.py --add-novelty --run-placebo --run-rivn-sensitivity
```

감성분석까지 포함하려면 선택 패키지 설치 후 `--add-sentiment`를 추가합니다.

```bash
.venv/bin/pip install -r requirements-optional.txt
.venv/bin/python run_daily_pipeline.py --add-sentiment --add-novelty --run-placebo --run-rivn-sensitivity
```

대시보드를 실행합니다.

```bash
.venv/bin/streamlit run dashboard_app.py
```

대시보드는 아래 화면으로 구성됩니다.

- 개요: 이벤트 수, CLEAN 이벤트 수, impact timeline, 상위 이벤트
- 이벤트 탐색기: 인물/topic/종목/오염/감성/track 필터와 개별 이벤트 상세
- 가설 검증: p-value, FDR 보정 p-value, H2 Dunn's 사후검정, 효과크기
- 방법론 점검: contamination 분포, placebo, topic audit, RIVN 민감도
- 케이스 스터디: Track 2 수동 사건과 LLM narrative
- Ask the Data: 추후 Groq/Ollama 연결을 위한 질의응답 자리

## 가격 데이터 폴백

기본적으로 yfinance에서 일봉 가격을 받습니다. 네트워크 문제나 yfinance 장애로 실패하면 `data/manual/price_cache.csv`를 자동으로 찾습니다.

수동 캐시 파일은 아래 컬럼을 가져야 합니다.

```csv
date,ticker,open,high,low,close,volume
2023-01-03,TSLA,100,105,99,104,1234567
```

## 감성분석: Twitter-RoBERTa

Twitter-RoBERTa는 Track 2 설명을 써주는 LLM이 아니라, 게시물 텍스트를 `positive`, `neutral`, `negative`로 분류하는 감성분석 모델입니다. 즉 Gemini/Groq/Ollama의 대체재가 아니라 FinBERT의 대체재입니다.

이 프로젝트의 원문은 금융 뉴스가 아니라 트럼프/머스크의 SNS 게시물이므로, 감성분석을 추가한다면 FinBERT보다 `cardiffnlp/twitter-roberta-base-sentiment-latest`를 우선 사용하도록 구성했습니다.

선택 패키지를 설치한 뒤:

```bash
.venv/bin/pip install -r requirements-optional.txt
```

감성분석까지 붙여서 실행합니다.

```bash
.venv/bin/python run_daily_pipeline.py --add-sentiment
```

결과 이벤트 테이블에는 `sentiment_label`, `sentiment_score`, `sentiment_confidence`, `sentiment_model` 컬럼이 추가됩니다.

감성분석을 켜면 통계검정에도 자동으로 반영됩니다.

- `positive` vs `negative` 그룹의 절대 초과수익률 차이: Mann-Whitney 검정
- `sentiment_score`와 절대 초과수익률의 관계: Spearman 상관
- 감성 라벨별 설명력: `eta_sentiment_label`

단, 각 그룹 표본이 5개 미만이면 검정은 자동 생략되고 리포트의 상태 컬럼에 표본 부족 사유가 표시됩니다.

## Novelty score

같은 인물이 최근 30일 안에 비슷한 말을 반복했는지 Jaccard 유사도로 계산합니다. `1.0`에 가까울수록 최근 반복 발언과 덜 비슷한 새 발언입니다.

```bash
.venv/bin/python run_daily_pipeline.py --add-novelty
```

Novelty를 켜면 통계검정에 아래 항목이 추가됩니다.

- `novelty_score`와 절대 초과수익률의 Spearman 상관
- 높은 novelty 그룹과 낮은 novelty 그룹의 Mann-Whitney 검정

## Placebo 검정

실제 게시물 날짜 대신 같은 종목/토픽 구조를 유지한 채 무작위 거래일을 뽑아 같은 통계검정을 반복합니다. 실제 H2 p-value보다 같거나 더 유의한 placebo 결과가 자주 나오면, topic 효과 해석은 보수적으로 해야 합니다.

```bash
.venv/bin/python run_daily_pipeline.py --run-placebo --placebo-iterations 200
```

결과:

- `outputs/tables/placebo_results.csv`
- `outputs/tables/placebo_summary.json`

요약 파일에는 H2(topic 차이)뿐 아니라 H3(Musk vs Trump)도 실제 p-value 대비 placebo가 얼마나 자주 같거나 더 유의했는지 함께 저장됩니다.

## RIVN 민감도 분석

TSLA abnormal return은 기본적으로 `GM`, `F`, `RIVN` peer 평균 대비로 계산합니다. RIVN은 상장 기간과 변동성이 달라 peer 포함 여부가 결과에 영향을 줄 수 있으므로, 아래 옵션으로 `GM,F,RIVN`과 `GM,F` 버전을 비교합니다.

```bash
.venv/bin/python run_daily_pipeline.py --run-rivn-sensitivity
```

결과:

- `outputs/tables/rivn_sensitivity.csv`

## Topic 분류 검증

룰베이스 topic 분류가 얼마나 맞는지 확인하려면 먼저 수동 라벨링용 표본을 만듭니다.

```bash
.venv/bin/python audit_topics.py --sample-size 50
```

`data/manual/topic_audit_sample.csv`의 `manual_topic` 컬럼을 직접 채운 뒤 평가합니다.

```bash
.venv/bin/python audit_topics.py --evaluate
```

결과:

- `outputs/tables/topic_audit_summary.csv`
- `outputs/tables/topic_audit_confusion_matrix.csv`
- `outputs/tables/topic_audit_precision_recall.csv`

이 검증은 H2(topic별 반응 차이)의 전제 조건을 점검하기 위한 것입니다. topic 분류 정확도가 낮으면 H2의 통계적 유의성은 보수적으로 해석해야 합니다.

## 오염 플래그 기준

다중 게시 오염은 기본적으로 `person` 기준 24시간 이내 게시물을 보수적으로 플래깅합니다. 즉 같은 인물이 같은 날 서로 다른 topic을 올려도 이벤트 간섭 가능성이 있다고 보고 `multi_post_flag`를 세웁니다.

## Track 2 자동 백필 후보 만들기

Musk의 2025-04-14 이후 게시물은 `twscrape`로 후보를 보조 수집할 수 있습니다. 이 도구는 비공식 X 접근 방식이므로 지속 자동화가 아니라 1회성 보조 수집으로만 쓰는 것을 권장합니다. 막히거나 계정 문제가 생기면 바로 수동 큐레이션으로 돌아가면 됩니다.

선택 패키지 설치:

```bash
.venv/bin/pip install -r requirements-optional.txt
```

계정 정보는 코드에 쓰지 말고 환경변수로 넣습니다.

```bash
export TWSCRAPE_USERNAME="계정"
export TWSCRAPE_PASSWORD="비밀번호"
export TWSCRAPE_EMAIL="이메일"
export TWSCRAPE_EMAIL_PASSWORD="이메일_비밀번호_또는_빈값"

.venv/bin/python backfill_track2_musk_twscrape.py
```

결과는 기본적으로 `data/manual/track2_musk_backfill_candidates.csv`에 저장됩니다. 이 파일은 “후보 목록”이고, 실제 케이스 스터디에 쓸 사건만 확인해서 `data/manual/track2_curated_events.csv`에 옮기는 방식이 좋습니다.

## Track 2 내러티브 원칙

자동 설명은 감성분석 모델이 아니라 생성형 LLM이 담당합니다. 현재 선택지는 `gemini`, `groq`, `ollama`, `none`입니다.

설명 생성 프롬프트는 인과관계를 단정하지 않도록 설계되어 있습니다. 예를 들어 “이 발언 때문에 주가가 하락했다”가 아니라 “해당 발언 직후 이례적인 가격 반응이 관측됐다”처럼 관찰 중심으로 씁니다. 기사 본문이나 가격 맥락이 부족하면 리포트에 `판단보류`가 표시됩니다.

## 장중 케이스 스터디 실행

분봉 또는 시간봉 CSV가 있을 때 특정 이벤트 하나를 골라 발언 전후 가격 움직임을 봅니다.

```bash
.venv/bin/python run_intraday_case_study.py \
  --event-id tk1_000001 \
  --intraday-csv data/manual/example_intraday.csv
```

장중 CSV는 아래 컬럼을 가져야 합니다.

```csv
datetime,ticker,open,high,low,close,volume
2025-06-04 14:30:00,TSLA,100,101,99,100.5,123456
```
