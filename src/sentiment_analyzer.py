from transformers import pipeline

from .config import SENTIMENT_MODEL


class SentimentAnalyzer:

    def __init__(self):

        self.model = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL
        )

    def analyze(self, text):

        result = self.model(
            text[:2000],
            truncation=True
        )[0]

        label = result["label"]

        # Normalize labels
        label_mapping = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }

        label = label_mapping.get(label, label)

        return {
            "sentiment": label,
            "confidence": float(result["score"])
        }