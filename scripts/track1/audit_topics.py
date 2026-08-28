import argparse

import pandas as pd

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from market_mover import config
from market_mover.load_posts import load_all_posts
from market_mover.preprocess import preprocess_posts


def parse_args():
    parser = argparse.ArgumentParser(description="Topic 분류 검증용 수동 라벨링 CSV를 만들거나 평가합니다.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true", help="manual_topic이 채워진 audit CSV를 평가합니다.")
    return parser.parse_args()


def make_sample(sample_size: int, seed: int) -> None:
    posts = preprocess_posts(load_all_posts())
    sample = posts.sample(min(sample_size, len(posts)), random_state=seed).copy()
    sample = sample[["post_id", "person", "posted_at", "text_clean", "topic"]].rename(columns={"topic": "rule_topic"})
    sample["manual_topic"] = ""
    out = config.MANUAL_DIR / "topic_audit_sample.csv"
    sample.to_csv(out, index=False)
    print({"저장_경로": str(out), "표본_수": len(sample)})


def evaluate() -> None:
    path = config.MANUAL_DIR / "topic_audit_sample.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}가 없습니다. 먼저 audit_topics.py로 표본을 만드세요.")
    df = pd.read_csv(path)
    labeled = df[df["manual_topic"].fillna("").astype(str).str.len() > 0].copy()
    if labeled.empty:
        raise ValueError("manual_topic이 채워진 행이 없습니다.")
    labeled["correct"] = labeled["rule_topic"] == labeled["manual_topic"]
    accuracy = labeled["correct"].mean()
    matrix = pd.crosstab(labeled["manual_topic"], labeled["rule_topic"], rownames=["수동 라벨"], colnames=["룰 라벨"])
    labels = sorted(set(labeled["manual_topic"]) | set(labeled["rule_topic"]))
    metric_rows = []
    for label in labels:
        tp = ((labeled["manual_topic"] == label) & (labeled["rule_topic"] == label)).sum()
        fp = ((labeled["manual_topic"] != label) & (labeled["rule_topic"] == label)).sum()
        fn = ((labeled["manual_topic"] == label) & (labeled["rule_topic"] != label)).sum()
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and (precision + recall) else None
        metric_rows.append({"topic": label, "precision": precision, "recall": recall, "f1": f1, "support": int((labeled["manual_topic"] == label).sum())})
    out_matrix = config.TABLE_DIR / "topic_audit_confusion_matrix.csv"
    out_summary = config.TABLE_DIR / "topic_audit_summary.csv"
    out_metrics = config.TABLE_DIR / "topic_audit_precision_recall.csv"
    matrix.to_csv(out_matrix)
    pd.DataFrame([{"labeled_n": len(labeled), "accuracy": accuracy}]).to_csv(out_summary, index=False)
    pd.DataFrame(metric_rows).to_csv(out_metrics, index=False)
    print({"라벨링_수": len(labeled), "정확도": round(float(accuracy), 4), "혼동행렬": str(out_matrix), "precision_recall": str(out_metrics)})


def main():
    args = parse_args()
    config.ensure_output_folders()
    if args.evaluate:
        evaluate()
    else:
        make_sample(args.sample_size, args.seed)


if __name__ == "__main__":
    main()
