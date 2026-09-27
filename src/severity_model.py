from transformers import pipeline


class SeverityModel:

    def __init__(self):

        self.classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        )

        self.labels = [
            "Low severity",
            "Medium severity",
            "High severity",
            "Critical severity"
        ]

    def predict(self, text):

        result = self.classifier(
            text,
            candidate_labels=self.labels,
            multi_label=False
        )

        predicted_label = result["labels"][0]
        confidence = result["scores"][0]

        severity = predicted_label.replace(
            " severity",
            ""
        )

        return {
            "severity": severity,
            "confidence": float(confidence),
            "scores": {
                label.replace(
                    " severity",
                    ""
                ): float(score)
                for label, score in zip(
                    result["labels"],
                    result["scores"]
                )
            }
        }