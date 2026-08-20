import pandas as pd


DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _score_from_label(label: str) -> int:
    label_l = str(label).lower()
    if "positive" in label_l or label_l == "label_2":
        return 1
    if "negative" in label_l or label_l == "label_0":
        return -1
    return 0


class TwitterRobertaSentiment:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "Twitter-RoBERTa 감성분석을 쓰려면 선택 패키지가 필요합니다. "
                ".venv/bin/pip install -r requirements-optional.txt 를 실행하세요."
            ) from exc
        self.model_name = model_name
        self.classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

    def predict(self, texts):
        outputs = self.classifier(list(texts), truncation=True, max_length=512)
        rows = []
        for output in outputs:
            label = output["label"]
            rows.append(
                {
                    "sentiment_label": label,
                    "sentiment_score": _score_from_label(label),
                    "sentiment_confidence": float(output["score"]),
                    "sentiment_model": self.model_name,
                }
            )
        return pd.DataFrame(rows)


def add_sentiment_columns(posts: pd.DataFrame, model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> pd.DataFrame:
    if posts.empty:
        return posts
    analyzer = TwitterRobertaSentiment(model_name)
    chunks = []
    for start in range(0, len(posts), batch_size):
        chunk = posts.iloc[start:start + batch_size].copy()
        preds = analyzer.predict(chunk["text_clean"].fillna("").astype(str).tolist())
        preds.index = chunk.index
        chunks.append(pd.concat([chunk, preds], axis=1))
    return pd.concat(chunks).sort_index()
