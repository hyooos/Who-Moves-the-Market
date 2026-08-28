# data/ 폴더 설명

`market_mover` 파이프라인이 다루는 데이터를 4단계로 나눠서 관리합니다: `raw`(원본) → `manual`(사람이 직접 준비) → `interim`(중간 산출물) → `processed`(최종 결과). `outputs/`는 데이터가 아니라 분석 결과물(테이블·차트·리포트)이라 별도 폴더입니다.

```text
data/
├── raw/         원본 게시물 CSV — 사람이 직접 다운받아 넣음
├── manual/      FOMC 캘린더, Track 2 사건 목록 — 사람이 직접 입력
├── interim/     정제된 게시물, 가격 캐시 — 파이프라인이 자동 생성
└── processed/   최종 이벤트 테이블 — 파이프라인이 자동 생성
```

**손으로 만지는 곳은 `raw/`와 `manual/` 둘뿐입니다.** `interim/`, `processed/`는 `run_daily_pipeline.py`가 매번 새로 만드니 직접 편집하면 안 됩니다.

---

## `data/raw/` — 원본 게시물 데이터

**어디서 왔나**: Kaggle에서 받은 CSV 3개(Apify 트위터 스크래퍼 형식 포함) + 팀원이 직접 진행한 EDA 노트북 2개. 전부 사람이 직접 준비한 원본입니다.

이 저장소의 `data/raw/`에 CSV 3개와 두 EDA 노트북을 그대로 넣어두면 됩니다. `.gitignore`로 제외돼 있어 git에는 올라가지 않으니, 새로 clone했다면 각자 다시 채워야 합니다.

| 파일 | 행 수 | 무슨 데이터인가 | 핵심 컬럼 |
| --- | --- | --- | --- |
| `Kaggle Trump_2009_2025.csv` | 90,343 | Trump의 X(트위터)+Truth Social 게시물 전체(2009~2025). 삭제된 게시물도 아카이브에 남아 있음(`deleted_flag`) | `date`, `platform`, `text`, `favorite_count`, `repost_count`, `deleted_flag` |
| `all_musk_posts.csv` | 55,099 | Musk의 X 게시물 전체. Apify 스크래퍼 형식이라 컬럼명이 카멜케이스 | `createdAt`, `fullText`, `likeCount`, `retweetCount`, `replyCount`, `isReply` |
| `musk_quote_tweets.csv` | 7,273 | Musk가 인용(quote)한 트윗들의 상세 정보(원본 트윗 + Musk가 덧붙인 코멘트). **`all_musk_posts.csv`와 id 기준 100% 중복**이라 파이프라인 로딩 단계에서 자동으로 건너뜁니다(`load_posts.py`) — 뉴스 검색 등 참고용으로만 남겨둠 | `musk_quote_tweet`, `musk_quote_created_at`, `orig_tweet_text` |
| `musk_raw_eda.ipynb` | - | Musk 데이터 EDA 노트북(결측치, 중복, engagement 정규화 방법 비교 등) | - |
| `trump_raw_eda.ipynb` | - | Trump 데이터 EDA 노트북(플랫폼별 스케일 차이, 삭제 게시물 처리 방향 등) | - |

두 EDA 노트북에서 나온 결론(quote-tweet 중복 제거, engagement 결측을 0으로 채우지 않기, 답글 필터링 기준 등)은 실제 코드(`load_posts.py`)에 반영돼 있습니다 — 자세한 내용은 `docs/implementation_status.md` §3.2, §3.5를 참고하세요.

---

## `data/manual/` — 사람이 직접 준비/입력하는 파일

| 파일 | 현재 상태 | 무슨 데이터인가 |
| --- | --- | --- |
| `fomc_calendar.csv` | 일부만 채워짐 | 연준(FOMC) 회의 날짜 목록(`date,event_type`). 게시물이 FOMC 회의 전후 1일 이내면 "오염(macro_overlap_flag)"으로 분류하는 데 씀 |
| `track2_curated_events.csv` | **0건(빈 파일)** | Track 2(사람이 직접 고른 대표 사건)를 채우는 파일. 컬럼: `event_id,person,posted_at,platform,topic,ticker,source_url,description`. `outputs/tables/track2_news_candidates.csv`(자동 생성된 뉴스 후보 목록)에서 골라 옮겨 담으면 됨 |
| `price_cache.csv` | 없음(안 씀) | yfinance 실시간 다운로드가 실패할 때만 쓰는 수동 가격 캐시. 지금까지는 yfinance가 잘 됐어서 필요 없었음 |

---

## `data/interim/` — 파이프라인 중간 산출물 (자동 생성)

`run_daily_pipeline.py`를 돌릴 때마다 새로 만들어집니다.

| 파일 | 무슨 데이터인가 |
| --- | --- |
| `posts_cleaned.csv` | `data/raw/`의 원본 게시물 중 분석 기간(2023-01-01~2025-04-13)에 해당하고, topic 분류상 market-relevant로 판정된 게시물만 남긴 것. 현재 기준 3,015건이며 원본 ID·URL·UTC/ET 시각을 함께 보존 |
| `daily_prices.csv` | yfinance에서 받은 TSLA/SPY/QQQ/GM/F/RIVN 일봉 가격 원본 캐시 |
| `daily_prices_scored.csv` | 위 가격에 robust z-score, 초과수익률, impact_score 등을 계산해 붙인 것(2-pass baseline 적용 후 최종 버전) |
| `events_posts_aligned.csv` | 3,015개 게시물을 ET 장 시간과 실제 가격 거래일에 연결한 게시물 단위 추적 테이블 |
| `events_daily.csv` | 같은 화자·ticker·topic·반응 거래일의 글을 6시간 고정 창으로 묶은 사건 단위 테이블. 묶인 모든 원문·URL은 `member_*_json` 컬럼에 보존하며 장중 사건은 `PARTIAL_DAY_INTRADAY_PREFERRED`로 표시 |

---

## `data/processed/` — 최종 결과 (자동 생성)

| 파일 | 무슨 데이터인가 |
| --- | --- |
| `events_scored.csv` | **이 프로젝트의 최종 산출물.** 게시물 + 원문 URL + UTC/ET 시각 + 시장 세션 + 가격 반응(초과수익률, impact_score) + 오염 분류(CLEAN/MINOR/MAJOR) + 선택적 novelty score가 붙은 이벤트 테이블. Streamlit 대시보드와 `market_mover_report.html`이 최종적으로 읽는 파일 |

---

## `outputs/` — 데이터가 아니라 "분석 결과물" (자동 생성)

데이터 자체는 아니고 위 `events_scored.csv`를 분석한 결과입니다. 참고로 같이 정리합니다.

| 폴더/파일 | 내용 |
| --- | --- |
| `outputs/tables/stats_results.json` | 모든 통계검정 결과(H1~H4, ticker 교란 통제 보조검정, novelty/sentiment exploratory 검정, FDR 보정) |
| `outputs/tables/placebo_results.csv`, `placebo_summary.json` | placebo(무작위 날짜) 순열검정 200회 결과 |
| `outputs/tables/rivn_sensitivity.csv` | TSLA peer에 RIVN 포함/제외 민감도 비교 |
| `outputs/tables/filtering_summary.csv` | 인물별 원본 게시물 수 대비 필터링 비율 |
| `outputs/tables/track2_news_candidates.csv` | `scripts/track2/find_track2_news_candidates.py`가 찾은 Track 2 후보 사건 + 관련 뉴스 목록 |
| `outputs/figures/*.html` | Plotly 인터랙티브 차트 4종 |
| `outputs/reports/market_mover_report.html` | 정적 HTML 최종 리포트 |

---

## 요약: "내가 수집한 데이터 어디 있어?"

- **Kaggle에서 받은 원본 CSV, 직접 만든 EDA 노트북** → `data/raw/`에 원본 그대로 있습니다. 사람이 넣은 것 그대로, 코드가 건드리지 않습니다.
- **분석 결과(이벤트 테이블, 통계, 차트)** → `data/interim/`, `data/processed/`, `outputs/`에 자동 생성됩니다. 사람이 직접 만드는 게 아니라 `run_daily_pipeline.py`가 만듭니다.
- **아직 사람 손을 더 타야 하는 것** → `data/manual/track2_curated_events.csv`(0건, Track 2 큐레이션 필요)와 `fomc_calendar.csv`(일부만 채워짐) 두 개뿐입니다.
