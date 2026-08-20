import re

import pandas as pd


def _tokens(text: str) -> set:
    return set(re.findall(r"[A-Za-z0-9가-힣%$]+", str(text).lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def add_novelty_scores(posts: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    if posts.empty:
        return posts
    out = posts.sort_values(["person", "posted_at"]).copy()
    out["_tokens"] = out["text_clean"].map(_tokens)
    novelty = []
    max_similarity = []
    for idx, row in out.iterrows():
        start = row["posted_at"] - pd.Timedelta(days=lookback_days)
        history = out[
            (out["person"] == row["person"])
            & (out["posted_at"] < row["posted_at"])
            & (out["posted_at"] >= start)
        ]
        sims = [_jaccard(row["_tokens"], tokens) for tokens in history["_tokens"]]
        max_sim = max(sims) if sims else 0.0
        max_similarity.append(max_sim)
        novelty.append(1.0 - max_sim)
    out["novelty_score"] = novelty
    out["max_prior_similarity"] = max_similarity
    return out.drop(columns=["_tokens"]).sort_index()
