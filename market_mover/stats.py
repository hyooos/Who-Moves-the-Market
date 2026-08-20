import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests


def eta_squared(groups):
    groups = [np.asarray(group, dtype=float) for group in groups if len(group) > 0]
    if len(groups) < 2:
        return np.nan
    all_values = np.concatenate(groups)
    grand_mean = np.nanmean(all_values)
    ss_between = sum(len(group) * (np.nanmean(group) - grand_mean) ** 2 for group in groups)
    ss_total = np.nansum((all_values - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total else np.nan


def _safe_test(name, fn):
    try:
        stat, p_value = fn()
        return {"test": name, "statistic": float(stat), "p_value": float(p_value), "status": "완료"}
    except Exception as exc:
        return {"test": name, "statistic": None, "p_value": None, "status": f"생략: {exc}"}


def _require_n(series: pd.Series, min_n: int, label: str) -> pd.Series:
    values = series.dropna()
    if len(values) < min_n:
        raise ValueError(f"{label} 표본 부족 n={len(values)} < {min_n}")
    return values


def _require_frame(df: pd.DataFrame, columns: list, min_n: int, label: str) -> pd.DataFrame:
    values = df.dropna(subset=columns)
    if len(values) < min_n:
        raise ValueError(f"{label} 표본 부족 n={len(values)} < {min_n}")
    return values


def _add_fdr_correction(results: dict) -> None:
    keys = []
    p_values = []
    for key, value in results.get("tests", {}).items():
        p_value = value.get("p_value")
        if p_value is not None and not pd.isna(p_value):
            keys.append(key)
            p_values.append(float(p_value))
    if not p_values:
        results["multiple_testing"] = {"method": "fdr_bh", "n_tests": 0, "adjusted": {}}
        return
    reject, p_adj, _, _ = multipletests(p_values, method="fdr_bh")
    adjusted = {}
    for key, original, adjusted_p, is_reject in zip(keys, p_values, p_adj, reject):
        adjusted[key] = {
            "p_value": original,
            "p_value_fdr_bh": float(adjusted_p),
            "reject_fdr_0.05": bool(is_reject),
        }
        results["tests"][key]["p_value_fdr_bh"] = float(adjusted_p)
        results["tests"][key]["reject_fdr_0.05"] = bool(is_reject)
    results["multiple_testing"] = {"method": "fdr_bh", "n_tests": len(p_values), "adjusted": adjusted}


def _topic_pairwise_dunn(clean: pd.DataFrame, min_group_n: int = 5) -> dict:
    groups = {
        topic: group["abs_abnormal_return"].dropna()
        for topic, group in clean.groupby("topic")
        if len(group["abs_abnormal_return"].dropna()) >= min_group_n
    }
    topics = sorted(groups)
    if len(topics) < 2:
        return {"method": "dunn_fdr_bh", "status": "생략: 표본 충분한 topic 쌍이 없습니다.", "comparisons": []}
    try:
        import scikit_posthocs as sp
    except ImportError:
        return {"method": "dunn_fdr_bh", "status": "생략: scikit-posthocs가 설치되어 있지 않습니다.", "comparisons": []}

    dunn_df = clean[clean["topic"].isin(topics)].dropna(subset=["abs_abnormal_return", "topic"])
    matrix = sp.posthoc_dunn(dunn_df, val_col="abs_abnormal_return", group_col="topic", p_adjust="fdr_bh")
    rows = []
    for i, topic_a in enumerate(topics):
        for topic_b in topics[i + 1:]:
            rows.append(
                {
                    "topic_a": topic_a,
                    "topic_b": topic_b,
                    "n_a": int(len(groups[topic_a])),
                    "n_b": int(len(groups[topic_b])),
                    "p_value_fdr_bh": float(matrix.loc[topic_a, topic_b]),
                    "reject_fdr_0.05": bool(matrix.loc[topic_a, topic_b] < 0.05),
                }
            )
    return {"method": "dunn_fdr_bh", "status": "완료", "comparisons": rows}


def _topic_within_ticker_tests(clean: pd.DataFrame, min_group_n: int = 5) -> dict:
    """topic은 topic_rules.map_ticker()로 사실상 ticker와 1:1에 가깝게 매핑되므로,
    ticker를 섞은 채로 topic 효과를 검정하면 실제로는 종목별 변동성 차이(예: TSLA가
    QQQ/SPY보다 원래 변동성이 큼)를 topic 효과로 오인할 위험이 있다. 이 함수는 같은
    ticker 안에서만 topic을 비교해 그 교란을 통제한 보조(exploratory) 검정이다."""
    per_ticker = {}
    for ticker, ticker_group in clean.groupby("ticker"):
        topic_groups = [
            g["abs_abnormal_return"].dropna().values
            for _, g in ticker_group.groupby("topic")
            if len(g["abs_abnormal_return"].dropna()) >= min_group_n
        ]
        group_sizes = {
            topic: int(len(g["abs_abnormal_return"].dropna()))
            for topic, g in ticker_group.groupby("topic")
            if len(g["abs_abnormal_return"].dropna()) >= min_group_n
        }
        if len(topic_groups) < 2:
            per_ticker[ticker] = {
                "test": f"Kruskal-Wallis 검정: {ticker} 내부 topic별 절대 초과수익률 차이 (ticker 고정)",
                "statistic": None,
                "p_value": None,
                "status": "생략: 표본 충분한(>=5) topic이 2개 미만입니다.",
                "group_sizes": group_sizes,
            }
            continue
        result = _safe_test(
            f"Kruskal-Wallis 검정: {ticker} 내부 topic별 절대 초과수익률 차이 (ticker 고정)",
            lambda groups=topic_groups: kruskal(*groups),
        )
        result["group_sizes"] = group_sizes
        per_ticker[ticker] = result
    return per_ticker


def run_stats(events_scored: pd.DataFrame, fdr_scope: str = "all_tests") -> dict:
    clean = events_scored[events_scored["contamination_level"] == "CLEAN"].copy()
    clean["abs_abnormal_return"] = clean["abnormal_return"].abs()

    results = {
        "n_total": int(len(events_scored)),
        "n_clean": int(len(clean)),
        "group_sizes": {
            "person": clean.groupby("person").size().to_dict(),
            "topic": clean.groupby("topic").size().to_dict(),
            "ticker": clean.groupby("ticker").size().to_dict(),
            "trump_role": clean.dropna(subset=["trump_role"]).groupby("trump_role").size().to_dict(),
        },
        "tests": {},
        "effect_sizes": {},
        "warnings": [],
    }
    results["multiple_testing_rationale"] = (
        "기본값은 탐색적 분석에서 false discovery를 보수적으로 통제하기 위해 "
        "p-value가 산출된 모든 검정을 하나의 family로 묶어 Benjamini-Hochberg FDR 보정을 적용합니다."
    )
    if len(clean) < 5:
        results["warnings"].append(f"CLEAN 이벤트가 {len(clean)}개뿐이라 통계 검정 신뢰도가 낮을 수 있습니다.")
    has_sentiment = {"sentiment_label", "sentiment_score"}.issubset(clean.columns)
    if has_sentiment:
        sentiment_events = clean.dropna(subset=["sentiment_label", "sentiment_score", "abs_abnormal_return"]).copy()
        results["group_sizes"]["sentiment_label"] = sentiment_events.groupby("sentiment_label").size().to_dict()
    else:
        sentiment_events = pd.DataFrame()

    paired = clean.dropna(subset=["volatility_before", "volatility_after"])
    results["tests"]["h1_volatility_before_after"] = _safe_test(
        "Wilcoxon 검정: 게시 전후 변동성 차이",
        lambda: wilcoxon(paired["volatility_after"], paired["volatility_before"]),
    )

    topic_groups = [
        group["abs_abnormal_return"].dropna().values
        for _, group in clean.groupby("topic")
        if len(group["abs_abnormal_return"].dropna()) >= 5
    ]
    topic_group_sizes = {
        topic: int(len(group["abs_abnormal_return"].dropna()))
        for topic, group in clean.groupby("topic")
    }
    results["group_sizes"]["topic_valid_for_h2"] = {k: v for k, v in topic_group_sizes.items() if v >= 5}
    results["tests"]["h2_topic_difference"] = _safe_test(
        "Kruskal-Wallis 검정: 토픽별 절대 초과수익률 차이",
        lambda: kruskal(*topic_groups),
    )
    results["posthoc"] = {
        "h2_topic_pairwise": _topic_pairwise_dunn(clean),
    }

    # H2가 topic 효과인지 ticker(종목 자체의 변동성) 효과인지 구분하기 위해,
    # ticker를 고정한 채로 topic만 비교하는 보조 검정을 추가한다.
    within_ticker = _topic_within_ticker_tests(clean)
    for ticker, result in within_ticker.items():
        results["tests"][f"h2b_topic_within_ticker_{ticker}"] = result
    results["h2b_rationale"] = (
        "topic이 map_ticker()로 ticker와 사실상 1:1에 가깝게 매핑돼 있어, H2(topic 차이)가 "
        "실제로는 종목별 변동성 차이(예: TSLA가 QQQ/SPY보다 원래 변동성이 큼)를 반영할 수 있다. "
        "h2b_topic_within_ticker_* 검정은 ticker를 고정한 채로 topic만 비교해 이 교란을 확인한다."
    )

    musk = clean[clean["person"] == "Musk"]["abs_abnormal_return"].dropna()
    trump = clean[clean["person"] == "Trump"]["abs_abnormal_return"].dropna()
    results["tests"]["h3_musk_vs_trump"] = _safe_test(
        "Mann-Whitney 검정: 머스크 이벤트가 트럼프 이벤트보다 큰가",
        lambda: mannwhitneyu(musk, trump, alternative="greater"),
    )

    # H3도 H2와 같은 교란(person이 ticker와 거의 1:1)에 노출돼 있다. Musk와 Trump가
    # 둘 다 이벤트를 가진 유일한 ticker(QQQ)로 한정해서, ticker를 고정한 채로도
    # person 차이가 남는지 확인한다.
    qqq_clean = clean[clean["ticker"] == "QQQ"]
    qqq_musk = qqq_clean[qqq_clean["person"] == "Musk"]["abs_abnormal_return"].dropna()
    qqq_trump = qqq_clean[qqq_clean["person"] == "Trump"]["abs_abnormal_return"].dropna()
    results["tests"]["h3b_musk_vs_trump_within_qqq"] = _safe_test(
        "Mann-Whitney 검정: QQQ로 ticker를 고정한 채 머스크 vs 트럼프 (ticker 교란 통제)",
        lambda: mannwhitneyu(
            _require_n(qqq_musk, 5, "QQQ musk"),
            _require_n(qqq_trump, 5, "QQQ trump"),
            alternative="two-sided",
        ),
    )
    results["h3b_rationale"] = (
        "Musk는 SPY 이벤트가 없고 Trump는 TSLA 이벤트가 없어 person도 ticker와 거의 1:1로 "
        "매핑된다. h3_musk_vs_trump의 유의성이 실제 인물 효과인지 종목 구성 차이인지 확인하기 "
        "위해, 둘 다 이벤트가 있는 유일한 ticker(QQQ)로 한정해 재검정한다."
    )

    corr_df = clean.dropna(subset=["engagement", "abs_abnormal_return"])
    results["tests"]["h4_engagement_correlation"] = _safe_test(
        "Spearman 상관: 참여도와 절대 초과수익률",
        lambda: spearmanr(corr_df["engagement"], corr_df["abs_abnormal_return"]),
    )

    trump_events = clean[clean["person"] == "Trump"]
    candidate = trump_events[trump_events["trump_role"] == "candidate_or_citizen"]["abs_abnormal_return"].dropna()
    president = trump_events[trump_events["trump_role"] == "president"]["abs_abnormal_return"].dropna()
    results["tests"]["trump_role_difference"] = _safe_test(
        "Mann-Whitney 검정: 트럼프 후보/시민 시기와 대통령 시기 차이",
        lambda: mannwhitneyu(candidate, president),
    )

    if has_sentiment:
        positive = sentiment_events[
            sentiment_events["sentiment_label"].astype(str).str.lower().str.contains("positive|label_2", regex=True)
        ]["abs_abnormal_return"]
        negative = sentiment_events[
            sentiment_events["sentiment_label"].astype(str).str.lower().str.contains("negative|label_0", regex=True)
        ]["abs_abnormal_return"]
        results["tests"]["exploratory_sentiment_positive_vs_negative"] = _safe_test(
            "Mann-Whitney 검정: 긍정 게시물과 부정 게시물의 절대 초과수익률 차이",
            lambda: mannwhitneyu(
                _require_n(positive, 5, "positive"),
                _require_n(negative, 5, "negative"),
                alternative="two-sided",
            ),
        )
        results["tests"]["exploratory_sentiment_score_correlation"] = _safe_test(
            "Spearman 상관: 감성 점수와 절대 초과수익률",
            lambda: (
                lambda df: spearmanr(df["sentiment_score"], df["abs_abnormal_return"])
            )(_require_frame(sentiment_events, ["sentiment_score", "abs_abnormal_return"], 5, "sentiment correlation")),
        )
    else:
        results["tests"]["exploratory_sentiment_positive_vs_negative"] = {
            "test": "Mann-Whitney 검정: 긍정 게시물과 부정 게시물의 절대 초과수익률 차이",
            "statistic": None,
            "p_value": None,
            "status": "생략: --add-sentiment 옵션으로 감성분석을 먼저 실행해야 합니다.",
        }
        results["tests"]["exploratory_sentiment_score_correlation"] = {
            "test": "Spearman 상관: 감성 점수와 절대 초과수익률",
            "statistic": None,
            "p_value": None,
            "status": "생략: --add-sentiment 옵션으로 감성분석을 먼저 실행해야 합니다.",
        }

    results["effect_sizes"]["eta_topic"] = eta_squared(topic_groups)
    results["effect_sizes"]["eta_person"] = eta_squared(
        [group["abs_abnormal_return"].dropna().values for _, group in clean.groupby("person")]
    )
    if has_sentiment:
        results["effect_sizes"]["eta_sentiment_label"] = eta_squared(
            [group["abs_abnormal_return"].dropna().values for _, group in sentiment_events.groupby("sentiment_label")]
        )
    if {"novelty_score", "abs_abnormal_return"}.issubset(clean.columns):
        novelty_events = clean.dropna(subset=["novelty_score", "abs_abnormal_return"]).copy()
        results["group_sizes"]["novelty_available"] = {"n": int(len(novelty_events))}
        results["tests"]["exploratory_novelty_correlation"] = _safe_test(
            "Spearman 상관: novelty score와 절대 초과수익률",
            lambda: (
                lambda df: spearmanr(df["novelty_score"], df["abs_abnormal_return"])
            )(_require_frame(novelty_events, ["novelty_score", "abs_abnormal_return"], 5, "novelty correlation")),
        )
        high = novelty_events[novelty_events["novelty_score"] >= novelty_events["novelty_score"].median()]["abs_abnormal_return"]
        low = novelty_events[novelty_events["novelty_score"] < novelty_events["novelty_score"].median()]["abs_abnormal_return"]
        results["tests"]["exploratory_high_vs_low_novelty"] = _safe_test(
            "Mann-Whitney 검정: 높은 novelty와 낮은 novelty의 절대 초과수익률 차이",
            lambda: mannwhitneyu(
                _require_n(high, 5, "high novelty"),
                _require_n(low, 5, "low novelty"),
                alternative="two-sided",
            ),
        )
        # H2/H3와 같은 이유로 novelty-반응 상관도 ticker 교란 가능성이 있는지 확인.
        # ticker별 novelty 평균이 다르면(예: TSLA 게시물이 평균적으로 더 novel하면)
        # pooled correlation이 실제 within-ticker novelty 효과가 아니라 ticker 구성
        # 차이를 반영할 수 있다.
        for ticker, ticker_group in novelty_events.groupby("ticker"):
            if len(ticker_group) < 10:
                continue
            results["tests"][f"h_novelty_within_ticker_{ticker}"] = _safe_test(
                f"Spearman 상관: {ticker} 내부 novelty score와 절대 초과수익률 (ticker 고정)",
                lambda g=ticker_group: spearmanr(g["novelty_score"], g["abs_abnormal_return"]),
            )
    else:
        results["tests"]["exploratory_novelty_correlation"] = {
            "test": "Spearman 상관: novelty score와 절대 초과수익률",
            "statistic": None,
            "p_value": None,
            "status": "생략: --add-novelty 옵션으로 novelty score를 먼저 계산해야 합니다.",
        }
        results["tests"]["exploratory_high_vs_low_novelty"] = {
            "test": "Mann-Whitney 검정: 높은 novelty와 낮은 novelty의 절대 초과수익률 차이",
            "statistic": None,
            "p_value": None,
            "status": "생략: --add-novelty 옵션으로 novelty score를 먼저 계산해야 합니다.",
        }
    _add_fdr_correction(results)
    return results


def save_stats(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
