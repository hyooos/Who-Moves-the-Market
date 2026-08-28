# baseline_results (2) 검토 결과

## 실행 무결성

새 결과 ZIP은 정상입니다.

- Track1 게시물 이벤트 3,015건, Track2 6건
- 정확한 중복 0건
- `source_post_id`, `source_url`, `posted_at_utc`, `posted_at_et` 모두 Track1 3,015건 전부 보존
- 이벤트 가격 반응 결측 0건
- 실제 세션 분류: 휴장 896, 장후 834, 정규장 768, 장전 517
- contamination: CLEAN 357, MINOR 2,304, MAJOR 354

ZIP의 `baseline_summary.json`에 남아 있던 `before timezone and clustering fixes` 문구와 단순 평일 기반 `session_counts_approx`는 노트북의 오래된 진단 코드였습니다. 실제 `events_daily.csv`에는 시간대 수정이 정상 반영돼 있으며, 다음 노트북 버전에서는 이 요약 문구와 진단 방식도 수정했습니다.

## 시간대 수정 전후 가설검정

| 검정 | 수정 전 p | 시간대 수정 후 p | FDR 후 결론 |
| --- | ---: | ---: | --- |
| H1 게시 전후 변동성 | 0.2730 | 0.1744 | 유의하지 않음 |
| H2 전체 topic 차이 | 5.95e-21 | 1.63e-23 | 유의함 |
| H2 QQQ 내부 topic | 0.0800 | 0.0705 | 유의하지 않음 |
| H2 SPY 내부 topic | 0.5560 | 0.0562 | 유의하지 않음 |
| H2 TSLA 내부 topic | 0.9980 | 0.9356 | 유의하지 않음 |
| H3 전체 Musk > Trump | 6.55e-8 | 9.31e-11 | 유의함 |
| H3 QQQ 내부 Musk > Trump | 0.8208 | 0.3722 | 유의하지 않음 |
| 참여도 상관 | 0.9670 | 0.6789 | 유의하지 않음 |
| Trump 지위 차이 | 0.7180 | 0.9417 | 유의하지 않음 |

시간대 수정으로 개별 p-value는 달라졌지만 핵심 해석은 바뀌지 않았습니다. 전체 topic 차이와 전체 Musk/Trump 차이는 유의하지만, ticker를 고정하면 모두 유의하지 않습니다. 따라서 pooled 결과를 “어떤 내용 또는 어떤 인물이 시장을 더 움직였다”라고 해석하면 안 되고, TSLA·QQQ·SPY의 원래 변동성 차이가 섞인 결과로 보는 것이 타당합니다.

## 남아 있던 구조적 문제와 수정

시간대 수정 결과의 3,015개 행은 여전히 게시물 단위였습니다. 이 중 2,608건(86.5%)이 24시간 내 다른 게시물과 겹쳐 다중게시로 표시됐습니다. 같은 가격 반응을 여러 번 세지 않도록 새 코드에서는 같은 `person + ticker + topic + event_date`의 게시물을 첫 글 기준 6시간 고정 창으로 묶습니다.

전체 데이터 검증에서는 3,015개 게시물이 2,173개 사건으로 정리됐고 모든 원문 ID·본문·URL이 보존됐습니다. 예상 contamination은 CLEAN 429, MINOR 1,502, MAJOR 242입니다. 이는 clustering까지만 적용해 계산한 표본 수이며, H1~H6는 새 Colab 결과로 다시 확정해야 합니다.

Track2의 자정 시각도 별도 검토가 필요했습니다. `00:00:00`은 날짜만 알려진 사건의 자리표시일 가능성이 높으므로 새 코드에서는 `unknown_date_only`와 `DATE_ONLY_MANUAL_REVIEW`로 표시합니다. 나머지 수동 시각도 출처에서 정확한 시각과 timezone을 확인하기 전에는 `MANUAL_TIME_UNVERIFIED`로 남깁니다.
