import joblib

# Load trained components
vectorizer = joblib.load(
    "models/product_tfidf.pkl"
)

model = joblib.load(
    "models/product_logistic_regression.pkl"
)


def predict_product(text):
    # Convert complaint text into TF-IDF
    text_vector = vectorizer.transform([text])

    # Prediction
    prediction = model.predict(text_vector)[0]

    # Probability
    probabilities = model.predict_proba(text_vector)[0]

    # Get top 3 predictions
    top_indices = probabilities.argsort()[-3:][::-1]

    classes = model.classes_

    top_predictions = [
        (classes[i], probabilities[i])
        for i in top_indices
    ]

    return prediction, top_predictions


# --------------------------------------------------
# TEST
# --------------------------------------------------

complaint = input("\nEnter complaint:\n")

prediction, top_predictions = predict_product(complaint)

print("\n" + "=" * 60)
print("PREDICTION")
print("=" * 60)

print(f"\nPredicted Product: {prediction}")

print("\nTop 3 predictions:")

for product, probability in top_predictions:
    print(f"{probability * 100:6.2f}%  {product}")