from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "cfpb_last_3_months.csv"
NLP_DATA_PATH = BASE_DIR / "data" / "processed" / "nlp_dataset.csv"

# Models
MODELS_DIR = BASE_DIR / "models"

PRODUCT_TFIDF_PATH = MODELS_DIR / "product_tfidf.pkl"
PRODUCT_MODEL_PATH = MODELS_DIR / "product_logistic_regression.pkl"
PRODUCT_LABELS_PATH = MODELS_DIR / "product_labels.pkl"

EMBEDDINGS_DIR = MODELS_DIR / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "complaint_embeddings.npy"

# Hugging Face models
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text settings
MAX_SIMILARITY_RESULTS = 5
MAX_SUMMARY_LENGTH = 100
MIN_SUMMARY_LENGTH = 30