import joblib

from .config import (
    PRODUCT_TFIDF_PATH,
    PRODUCT_MODEL_PATH
)


class ProductClassifier:

    def __init__(self):
        self.vectorizer = joblib.load(PRODUCT_TFIDF_PATH)
        self.model = joblib.load(PRODUCT_MODEL_PATH)

    def predict(self, text):

        features = self.vectorizer.transform([text])

        prediction = self.model.predict(features)[0]

        probabilities = self.model.predict_proba(features)[0]

        classes = self.model.classes_

        ranked = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "product": prediction,
            "confidence": float(max(probabilities)),
            "alternatives": [
                {
                    "product": product,
                    "confidence": float(probability)
                }
                for product, probability in ranked[:3]
            ]
        }