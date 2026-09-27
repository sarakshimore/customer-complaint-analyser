import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .config import SIMILARITY_MODEL


class SimilarityEngine:

    def __init__(
        self,
        dataframe,
        model=None,
        embeddings=None,
        dataset_hash=None
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        if "Consumer complaint narrative" not in self.df.columns:
            raise ValueError(
                "Dataset must contain the "
                "'Consumer complaint narrative' column."
            )

        # --------------------------------------------------------
        # Reuse an already loaded Sentence-BERT model.
        #
        # If a model is not provided, load one.
        # In ComplaintPipeline, the model is always provided,
        # so Sentence-BERT is loaded only once.
        # --------------------------------------------------------

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(
                SIMILARITY_MODEL
            )

        self.embeddings = None

        # --------------------------------------------------------
        # Create dataset fingerprint.
        # --------------------------------------------------------

        self.dataset_hash = (
            dataset_hash
            if dataset_hash is not None
            else self.create_dataset_hash()
        )

        # --------------------------------------------------------
        # Use supplied embeddings only when their size matches
        # the current dataset.
        # --------------------------------------------------------

        if embeddings is not None:

            embeddings = np.asarray(
                embeddings
            )

            if len(embeddings) != len(self.df):

                raise ValueError(
                    "Number of embeddings does not match "
                    "number of complaints in the dataset."
                )

            self.embeddings = embeddings

    # ============================================================
    # DATASET HASH
    # ============================================================

    def create_dataset_hash(self):

        texts = (
            self.df[
                "Consumer complaint narrative"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .tolist()
        )

        content = "\n".join(texts)

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    # ============================================================
    # BUILD EMBEDDINGS
    # ============================================================

    def build_embeddings(self):

        texts = (
            self.df[
                "Consumer complaint narrative"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .tolist()
        )

        if not texts:

            raise ValueError(
                "No complaint narratives found."
            )

        print(
            f"Generating embeddings for "
            f"{len(texts):,} complaints..."
        )

        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        self.embeddings = np.asarray(
            self.embeddings
        )

        print(
            f"Generated "
            f"{len(self.embeddings):,} embeddings."
        )

        return self.embeddings

    # ============================================================
    # ENSURE EMBEDDINGS
    # ============================================================

    def ensure_embeddings(self):

        if self.embeddings is None:

            self.build_embeddings()

        return self.embeddings

    # ============================================================
    # FIND SIMILAR COMPLAINTS
    # ============================================================

    def find_similar(
        self,
        text,
        max_results=5,
        similarity_threshold=0.40
    ):

        if not text or not str(text).strip():

            return []

        self.ensure_embeddings()

        query_embedding = self.model.encode(
            [str(text)],
            normalize_embeddings=True
        )

        similarities = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        ranked_indices = np.argsort(
            similarities
        )[::-1]

        results = []

        for index in ranked_indices:

            similarity = float(
                similarities[index]
            )

            if similarity < similarity_threshold:
                continue

            complaint = self.df.iloc[index]

            results.append({
                "Complaint ID": complaint.get(
                    "Complaint ID",
                    index
                ),
                "Product": complaint.get(
                    "Product",
                    "N/A"
                ),
                "Issue": complaint.get(
                    "Issue",
                    "N/A"
                ),
                "Company": complaint.get(
                    "Company",
                    "N/A"
                ),
                "State": complaint.get(
                    "State",
                    "N/A"
                ),
                "Similarity": similarity,
                "Narrative": complaint[
                    "Consumer complaint narrative"
                ]
            })

            if len(results) >= max_results:
                break

        return results

    # ============================================================
    # GETTERS
    # ============================================================

    def get_embeddings(self):

        return self.ensure_embeddings()

    def get_embedding_count(self):

        if self.embeddings is None:
            return 0

        return len(
            self.embeddings
        )

    def embeddings_available(self):

        return self.embeddings is not None

    def get_dataset_hash(self):

        return self.dataset_hash