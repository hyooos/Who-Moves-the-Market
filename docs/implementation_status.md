# Who Moves the Market? — 구현 현황 (진행상황 문서)

최종 갱신: 2026-08-25. 이 문서는 계속 갱신되는 **살아있는 상태 문서**입니다 — 뭘 실행했고, 뭐가 됐고, 뭐가 아직 안 됐는지를 그때그때 남깁니다. 처음 여는 사람은 "지금 뭐가 되고 뭐가 안 되는지"를 이 문서 하나로 파악할 수 있습니다.

> **가장 먼저 읽어야 할 것**: 이 저장소는 코드는 오래전에 완성돼 있었지만, **`data/raw/`·`data/processed/`·`outputs/`가 전부 비어 있어 파이프라인이 한 번도 끝까지 실행된 적이 없는 상태**였습니다(§1). 2026-08-25 세션에서 실제 Kaggle CSV로 처음 끝까지 실행했고(§2), 그 결과 H2(topic 차이)·H3(Musk vs Trump 차이)가 원래는 극도로 유의했지만 **종목(ticker) 교란이었다는 걸 확인**했습니다(§2.3) — 옆 폴더 `market_mover/docs/implementation_status.md`(같은 코드베이스의 이전 스냅샷)에서 이미 한 번 발견됐던 것과 동일한 패턴이 이 저장소에서도 재현된 것입니다. 대시보드 UX 개선도 별도로 진행했습니다(§3).

---

## 0. 이 프로젝트가 뭘 하는 프로젝트인가

Trump와 Musk의 SNS 게시물(Kaggle CSV)과 실제 주가(TSLA/SPY/QQQ/GM/F/RIVN)를 이벤트 스터디(event study) 방법론으로 연결해, "이 사람이 이런 주제로 글을 올린 날 시장이 통계적으로 이례적인 반응을 보였는가?"를 검증하는 Python 파이프라인 + Streamlit 대시보드입니다. robust z-score, 오염(contamination) 분류, placebo 순열검정, FDR 보정, ticker 교란 통제 보조검정 등 "예쁜 결과"가 아니라 "방어 가능한 결과"를 위한 장치가 코드 레벨에 내장돼 있습니다. 자세한 설계는 [`README.md`](../README.md), 파일별 역할은 [`../market_mover/docs/codebase_reference.md`](../../market_mover/docs/codebase_reference.md) 참고(같은 코드베이스의 이전 스냅샷 기준 — 이 저장소와 파일 구성이 거의 동일합니다).

---

## 1. 세션 시작 시점 상태 — "0건" 문제

사용자가 대시보드를 열었을 때 "분석한 사건 0건 · 분석 기간 – · 관련 시장 0개"만 보이는 상태였습니다. 원인을 추적한 결과:

- `data/raw/`, `data/interim/`, `data/processed/`, `outputs/tables/`가 전부 `.gitkeep`만 있는 빈 폴더 — `.gitignore`에 원본·산출물이 커밋 대상에서 빠지도록 설정돼 있어(의도된 설계), 새로 clone하면 항상 빈 상태로 시작합니다.
- 게다가 로컬 가상환경(`.venv`)에 `requirements.txt`가 완전히 설치돼 있지 않아(`yfinance`, `scipy`, `statsmodels`, `scikit-posthocs` 등 누락) `run_daily_pipeline.py` 자체가 애초에 실행 불가능한 상태였습니다.

`dashboard_app.py`의 `market_mover/dashboard_data.py`는 파일이 없으면 빈 DataFrame을 반환하도록 방어적으로 짜여 있어 에러 없이 뜨긴 하지만, 그래서 "왜 안 되지?"가 잘 안 보이는 게 문제였습니다.

---

## 2. 파이프라인 첫 실전 실행 (2026-08-25)

### 2.1 실행한 명령

```bash
pip install -r requirements.txt          # yfinance, scipy, statsmodels, scikit-posthocs 등 설치
PYTHONPATH=. .venv/bin/python run_daily_pipeline.py   # 옵션 없이 최소 실행
```

사용자가 `data/raw/`에 실제 Kaggle CSV 3개(`Kaggle Trump_2009_2025.csv`, `all_musk_posts.csv`, `musk_quote_tweets.csv`)를 직접 넣은 뒤 실행했습니다.

### 2.2 실행 결과

```text
원본 게시물: 127,246건 (musk_quote_tweets.csv는 100% 중복이라 자동 제외, Musk 짧은 감탄사형 답글 18,196건 제외 후)
  - Trump: 90,343행
  - Musk: 36,903행 (55,099행에서 답글 필터링 후)
market-relevant 필터링: 44,775 → 3,015건 (6.7%)
Track1 클러스터링: 3,015개 게시물 → 2,173개 사건(같은 화자·ticker·topic·거래일, 첫 글 기준 6시간 창)
Track2(수동 큐레이션) 사건: 6건 (이미 data/manual/track2_curated_events.csv에 채워져 있었음)
최종 이벤트 테이블: 2,179건 (Track1 2,173 + Track2 6)
  - 인물: Musk 1,135 / Trump 1,044
  - 종목: QQQ 886 / SPY 698 / TSLA 595
  - 오염 분류: CLEAN 429 / MINOR 1,502 / MAJOR 242
분석 기간: 2023-01-03 ~ 2025-10-23
```

산출물: `data/processed/events_scored.csv`, `data/interim/daily_prices_scored.csv`, `outputs/tables/stats_results.json`, `outputs/tables/filtering_summary.csv`, `outputs/figures/*.html`(4종), `outputs/reports/market_mover_report.html` — 전부 생성 확인.

이번 실행은 `--add-sentiment`, `--add-novelty`, `--run-placebo`, `--run-rivn-sensitivity`, `--build-narratives` 없이 **최소 옵션으로만** 돌린 상태입니다(§4의 다음 단계 참고).

### 2.3 통계 검정 결과 — H2/H3는 유의하지만 ticker 교란

`stats.py`의 통계 검정 결과(`outputs/tables/stats_results.json`, CLEAN 429건 기준):

| 검정 | 내용 | p-value | FDR 보정 p-value |
| --- | --- | --- | --- |
| H1 | 게시 전후 변동성 차이 (Wilcoxon) | 0.225 | 0.406 (미유의) |
| H2 | topic별 절대 초과수익률 차이 (Kruskal-Wallis) | **1.38e-29** | **1.24e-28** (매우 유의) |
| H2b | ↳ QQQ 내부만 고정 | 0.057 | 0.172 (미유의) |
| H2b | ↳ SPY 내부만 고정 | 0.152 | 0.341 (미유의) |
| H2b | ↳ TSLA 내부만 고정 | 0.807 | 0.807 (미유의) |
| H3 | Musk vs Trump 반응 크기 (Mann-Whitney) | **4.62e-11** | **2.08e-10** (매우 유의) |
| H3b | ↳ QQQ로 ticker 고정 후 재비교 | 0.561 | 0.662 (미유의) |
| H4 | 참여도-반응 크기 상관 (Spearman) | 0.589 | 0.662 (미유의) |
| trump_role | 후보/시민 시기 vs 대통령 시기 | 0.383 | 0.574 (미유의) |

**읽는 법**: 전체 표본에서는 topic(H2)과 인물(H3) 차이가 극도로 유의하게 나오지만, 종목(ticker)을 고정한 채 다시 비교하면(H2b, H3b) 유의성이 거의 완전히 사라집니다. 이건 `../market_mover/docs/implementation_status.md` §3.6에서 이미 한 번 발견된 것과 정확히 같은 패턴 — "topic·인물 효과처럼 보였던 게 사실은 종목별 변동성 차이(ticker confound)"라는 결론이 이 저장소의 실제 데이터로도 재현됐습니다. sentiment·novelty 관련 exploratory 검정은 이번 실행에 `--add-sentiment`/`--add-novelty`를 안 켜서 표본 0건으로 전부 생략 상태입니다.

### 2.4 아직 안 돌린 선택 기능

| 기능 | 플래그 | 상태 |
| --- | --- | --- |
| 감성분석(Twitter-RoBERTa) | `--add-sentiment` | 미실행 |
| Novelty score | `--add-novelty` | 미실행 |
| Placebo 순열검정(200회) | `--run-placebo` | 미실행 |
| RIVN peer 민감도 분석 | `--run-rivn-sensitivity` | 미실행 |
| Track2 LLM 내러티브 생성 | `--build-narratives --llm-provider ...` | 미실행 (narrative 컬럼은 비어 있음) |
| topic 분류 정확도 검증 | `audit_topics.py --evaluate` | **표본은 이미 채워져 있음**(`data/manual/topic_audit_sample.csv` 50건 `manual_topic` 전부 기입됨), `--evaluate`만 실행하면 바로 정확도가 나옴 |

---

## 3. 대시보드 UX 개선 (별도 브랜치, 아직 미병합)

`yejin-update`에 있던 대시보드를 실제로 써보면서 나온 피드백을 반영해 `dashboard_app.py`를 수정했습니다. **이 작업은 `dashboard-period-fixes` 브랜치에만 있고, `yejin-update`/`main`에는 아직 merge되지 않았습니다** (§5 참고 — 한 번 merge했다가 사용자 요청으로 revert했습니다).

1. **분석 기간 컨트롤 위치**: 처음엔 사이드바(자동으로 접혀 있어 안 보임)에 넣었다가, 페이지 최상단(제목 바로 아래)으로 옮기고, "최근 연결 사건의 시장 반응" 섹션에는 그 기간을 덮어쓸 수 있는 로컬 컨트롤도 추가함 — Track2(뉴스 기반 수동 등록) 사건은 원래 극단값만 골라 넣은 표본이라 기간을 안 좁히면 게이지가 항상 100%대로만 나오는 문제를 해결.
2. **가격 차트가 선택한 기간을 실제로 반영**: x축뿐 아니라 y축도 선택한 기간의 실제 가격 범위로 다시 계산하도록 수정(전에는 x축만 확대되고 y축은 전체 기간 기준이라 눌린 것처럼 보였음).
3. **차트 범례 추가**: 보라색(Musk)/주황색(Trump) 점 색깔이 뭘 의미하는지 왼쪽 위에 작은 범례로 표시.
4. **"원인 구분 가능성" 폰트 버그 수정**: 문장형 텍스트("다른 요인 일부 있음")를 숫자용 `st.metric` 큰 폰트에 넣어서 글자 하나만 잘려 확대된 것처럼 보이던 버그를 일반 텍스트 크기 카드로 교체.
5. **번역/AI 요약 버튼 비활성 안내**: AI 서비스(사이드바)를 선택 안 하면 버튼이 잠기는 게 의도된 동작인데 이유를 안내하는 문구가 전혀 없었음 — 버튼 아래에 "왼쪽 사이드바 AI 요약 설정에서 선택하세요" 안내를 추가.

---

## 4. 다음 단계 로드맵

우선순위 순서로 정리:

1. **`dashboard-period-fixes` → `yejin-update` 병합 여부 결정** — 지금은 별도 브랜치로만 올라가 있고 PR #1은 merge됐다가 revert된 상태(§5). 검토 후 다시 merge할지 결정 필요.
2. **`audit_topics.py --evaluate` 실행** — 표본(50건)이 이미 채워져 있어서 바로 실행 가능. topic 분류(키워드 룰) 정확도를 수치로 확인할 수 있음.
3. **Placebo 검정 + RIVN 민감도 실행** — `run_daily_pipeline.py --run-placebo --run-rivn-sensitivity`. §2.3의 ticker 교란 발견이 우연이 아니라는 걸 순열검정으로 한 번 더 방어하려면 필요.
4. **Track2 LLM 내러티브 생성** — Ollama(로컬, 무료) 또는 Gemini/Groq API 키로 `--build-narratives` 실행하면 케이스 스터디 탭에 근거 기반 설명이 채워짐.
5. **감성분석·novelty 추가** — `--add-sentiment --add-novelty`. 지금은 표본 0건이라 관련 exploratory 검정이 전부 생략된 상태.

---

## 5. Git 브랜치 상태 메모

- `dashboard-period-fixes` 브랜치를 만들어 §3의 대시보드 수정 사항을 커밋(`4b847e1`)하고 origin에 push함.
- PR #1(`dashboard-period-fixes` → `yejin-update`)을 만들고 곧바로 merge(`d5354ed`)했으나, **`yejin-update`는 그대로 두고 별도 브랜치로만 올리길 원했다는 걸 뒤늦게 확인**해서 merge 커밋을 revert(`ddd9533`)함 — `yejin-update`의 `dashboard_app.py`는 현재 merge 이전과 100% 동일한 내용으로 복원돼 있음(diff 없음 확인).
- `dashboard-period-fixes` 브랜치는 원래 커밋(`4b847e1`)을 그대로 가리키도록 다시 만들어 origin에 push해둔 상태 — merge는 안 한 채로 남아 있음.
- 이 파이프라인 실행(§2)과 데이터 산출물(`data/processed/`, `outputs/`)은 `.gitignore` 대상이라 git 이력과 무관하게 로컬에만 존재함 — 새로 clone하면 다시 §1 상태로 돌아가므로, 다른 팀원 컴퓨터에서 대시보드를 채우려면 §2.1의 두 명령을 그대로 실행하면 됨.
