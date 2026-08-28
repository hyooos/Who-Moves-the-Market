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

    def predict(self, texts, batch_size: int = 64):
        outputs = self.classifier(
            list(texts),
            truncation=True,
            max_length=512,
            batch_size=batch_size,
        )
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


def add_sentiment_columns(
    posts: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    text_column: str = "text_clean",
) -> pd.DataFrame:
    """텍스트 감성을 계산해 기존 데이터프레임에 안전하게 덮어씁니다.

    사건 clustering 이후에는 ``cluster_text_clean``을 전달해 대표 게시물 한 건이
    아니라 묶인 사건 전체 문장을 기준으로 감성을 계산할 수 있습니다.
    """
    if posts.empty:
        return posts.copy()
    if text_column not in posts.columns:
        raise ValueError(f"감성분석 텍스트 컬럼이 없습니다: {text_column}")
    analyzer = TwitterRobertaSentiment(model_name)
    result = posts.copy()
    prediction_chunks = []
    for start in range(0, len(posts), batch_size):
        chunk = result.iloc[start:start + batch_size]
        preds = analyzer.predict(
            chunk[text_column].fillna("").astype(str).tolist(),
            batch_size=batch_size,
        )
        preds.index = chunk.index
        prediction_chunks.append(preds)
    predictions = pd.concat(prediction_chunks).sort_index()
    for column in predictions.columns:
        result[column] = predictions[column]
    return result.sort_index()
