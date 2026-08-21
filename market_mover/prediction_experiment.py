""""내용이 반응 크기를 예측하는 데 도움이 되는가?"를 실제 out-of-sample
예측으로 검증하는 실험.

§5-3의 η²(효과크기) 분해는 표본 안(in-sample) 설명력이라 낙관적으로 보일 수
있다. 이 모듈은 5-fold 교차검증으로 진짜 예측 정확도(R², MAE)를 재는, 더
엄격한 검증이다. 사용 피처를 다르게 한 세 모델을 비교해서 "종목만으로 예측한
것"과 "종목+내용으로 예측한 것"이 얼마나 다른지 직접 보여준다.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "abs_abnormal_return"

_MODELS = [
    ("A_종목만", ["ticker"], []),
    ("B_종목+내용(topic+인물+참여도+novelty)", ["ticker", "topic", "person"], ["engagement", "novelty_score"]),
    ("C_내용만(종목 제외)", ["topic", "person"], ["engagement", "novelty_score"]),
]


def _prepare(events: pd.DataFrame) -> pd.DataFrame:
    clean = events[events["contamination_level"] == "CLEAN"].copy()
    clean[TARGET] = clean["abnormal_return"].abs()
    clean["engagement"] = pd.to_numeric(clean.get("engagement"), errors="coerce").fillna(0)
    novelty = pd.to_numeric(clean.get("novelty_score"), errors="coerce")
    clean["novelty_score"] = novelty.fillna(novelty.median())
    return clean.dropna(subset=[TARGET, "ticker", "topic", "person"])


def run_prediction_experiment(events: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    clean = _prepare(events)
    rows = []
    for label, cat_features, num_features in _MODELS:
        X = clean[cat_features + num_features]
        y = clean[TARGET].values
        pre = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
            remainder="passthrough" if num_features else "drop",
        )
        pipe = Pipeline([("pre", pre), ("model", LinearRegression())])
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
        mae = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
        rows.append(
            {
                "모델": label,
                "피처": ",".join(cat_features + num_features),
                "R2_평균": round(float(r2.mean()), 4),
                "R2_최소": round(float(r2.min()), 4),
                "R2_최대": round(float(r2.max()), 4),
                "MAE_평균": round(float(mae.mean()), 6),
                "n": len(clean),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from . import config

    events = pd.read_csv(config.PROCESSED_DIR / "events_scored.csv")
    result = run_prediction_experiment(events)
    print(result.to_string(index=False))
    out_path = config.TABLE_DIR / "prediction_experiment.csv"
    result.to_csv(out_path, index=False)
    print(f"저장: {out_path}")
