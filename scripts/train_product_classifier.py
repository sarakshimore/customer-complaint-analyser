import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

FILE = "data/processed/nlp_dataset.csv"

print("Loading dataset...")

df = pd.read_csv(FILE)

print(f"Total records: {len(df):,}")

# --------------------------------------------------
# 2. CLEAN TEXT
# --------------------------------------------------

df["Consumer complaint narrative"] = (
    df["Consumer complaint narrative"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# Remove empty narratives
df = df[df["Consumer complaint narrative"] != ""]

# Remove duplicate complaint IDs
df = df.drop_duplicates(subset=["Complaint ID"])

print(f"Records after cleaning: {len(df):,}")

# --------------------------------------------------
# 3. CHECK TARGET
# --------------------------------------------------

print("\nProduct distribution:")

print(
    df["Product"]
    .value_counts()
    .to_string()
)

# --------------------------------------------------
# 4. FEATURES AND TARGET
# --------------------------------------------------

X = df["Consumer complaint narrative"]
y = df["Product"]

# --------------------------------------------------
# 5. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))

# --------------------------------------------------
# 6. TF-IDF
# --------------------------------------------------

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape :", X_test_tfidf.shape)

# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

# --------------------------------------------------
# 7.5 SAVE MODEL
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    vectorizer,
    "models/product_tfidf.pkl"
)

joblib.dump(
    model,
    "models/product_logistic_regression.pkl"
)

print("\nModels saved:")
print("models/product_tfidf.pkl")
print("models/product_logistic_regression.pkl")

# --------------------------------------------------
# 8. PREDICTION
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = model.predict(X_test_tfidf)

# --------------------------------------------------
# 9. EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)