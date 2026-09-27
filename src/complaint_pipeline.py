import pandas as pd
from sentence_transformers import SentenceTransformer

from .product_classifier import ProductClassifier
from .sentiment_analyzer import SentimentAnalyzer
from .severity_analyzer import calculate_severity
from .key_issue_extractor import KeyIssueExtractor
from .summarizer import ComplaintSummarizer
from .routing import recommend_department
from .sentiment_interpreter import SentimentInterpreter
from .similarity_engine import SimilarityEngine
from .complaint_clusterer import ComplaintClusterer
from .config import SIMILARITY_MODEL


class ComplaintPipeline:

    def __init__(self):

        self.product_classifier = ProductClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.key_issue_extractor = KeyIssueExtractor()
        self.summarizer = ComplaintSummarizer()
        self.sentiment_interpreter = SentimentInterpreter()

        # --------------------------------------------------------
        # Load Sentence-BERT ONCE.
        #
        # Because ComplaintPipeline is created with
        # @st.cache_resource in the Streamlit page, this model
        # remains loaded while the cached pipeline remains alive.
        # --------------------------------------------------------

        print("Loading Sentence-BERT model...")

        self.similarity_model = SentenceTransformer(
            SIMILARITY_MODEL
        )

        print("Sentence-BERT model loaded.")

        # Initialized when a dataset is analyzed
        self.similarity_engine = None
        self.clusterer = None

        # --------------------------------------------------------
        # Dataset-aware embedding cache.
        #
        # {
        #     dataset_hash: embeddings
        # }
        # --------------------------------------------------------

        self.embedding_cache = {}

    # ============================================================
    # SINGLE COMPLAINT ANALYSIS
    # ============================================================

    def analyze(self, text):

        if not text or not text.strip():

            raise ValueError(
                "Complaint text cannot be empty."
            )

        text = text.strip()

        # --------------------------------------------
        # Product
        # --------------------------------------------

        product_result = (
            self.product_classifier.predict(
                text
            )
        )

        product = product_result["product"]

        # --------------------------------------------
        # Sentiment
        # --------------------------------------------

        sentiment_result = (
            self.sentiment_analyzer.analyze(
                text
            )
        )

        # --------------------------------------------
        # Severity
        # --------------------------------------------

        severity_result = calculate_severity(
            text
        )

        # --------------------------------------------
        # Key issues
        # --------------------------------------------

        key_issues = (
            self.key_issue_extractor.extract(
                text
            )
        )

        # --------------------------------------------
        # Summary
        # --------------------------------------------

        summary = (
            self.summarizer.summarize(
                text
            )
        )

        # --------------------------------------------
        # Contextual interpretation
        # --------------------------------------------

        interpretation = (
            self.sentiment_interpreter.interpret(
                sentiment_result["sentiment"],
                severity_result["severity"],
                key_issues,
                text,
                severity_result[
                    "escalation_detected"
                ]
            )
        )

        # --------------------------------------------
        # Department
        # --------------------------------------------

        department = recommend_department(
            product
        )

        return {
            "product": product_result,
            "sentiment": sentiment_result,
            "severity": severity_result,
            "key_issues": key_issues,
            "summary": summary,
            "interpretation": interpretation,
            "department": department
        }

    # ============================================================
    # BATCH ANALYSIS
    # ============================================================

    def analyze_batch(
        self,
        df,
        text_column
    ):

        if df is None or df.empty:

            raise ValueError(
                "The dataset is empty."
            )

        if text_column not in df.columns:

            raise ValueError(
                f"Column '{text_column}' was not found "
                "in the dataset."
            )

        # ========================================================
        # PREPARE DATASET
        # ========================================================

        working_df = (
            df.copy()
            .reset_index(drop=True)
        )

        working_df[text_column] = (
            working_df[text_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # --------------------------------------------------------
        # REMOVE EMPTY / INVALID COMPLAINTS
        # --------------------------------------------------------

        working_df = working_df[
            (working_df[text_column] != "")
            &
            (
                working_df[text_column]
                .str.lower()
                != "nan"
            )
        ].reset_index(drop=True)

        if working_df.empty:

            raise ValueError(
                "No valid complaint narratives were found "
                "in the selected column."
            )

        # ========================================================
        # SEMANTIC EMBEDDINGS + CLUSTERING
        # ========================================================

        try:

            # ----------------------------------------------------
            # Create dataframe for semantic analysis
            # ----------------------------------------------------

            semantic_df = pd.DataFrame({
                "Consumer complaint narrative":
                    working_df[text_column]
            })

            # Preserve available metadata.
            for column in [
                "Complaint ID",
                "Product",
                "Issue",
                "Company",
                "State"
            ]:

                if column in working_df.columns:

                    semantic_df[column] = (
                        working_df[column]
                    )

            # ----------------------------------------------------
            # Create similarity engine using the ALREADY LOADED
            # Sentence-BERT model.
            # ----------------------------------------------------

            temp_similarity_engine = (
                SimilarityEngine(
                    semantic_df,
                    model=self.similarity_model
                )
            )

            dataset_hash = (
                temp_similarity_engine
                .get_dataset_hash()
            )

            # ====================================================
            # EMBEDDING CACHE
            # ====================================================

            if dataset_hash in self.embedding_cache:

                print(
                    "Reusing cached "
                    "Sentence-BERT embeddings..."
                )

                embeddings = (
                    self.embedding_cache[
                        dataset_hash
                    ]
                )

                # Create active engine using the cached
                # embeddings and the already loaded model.
                self.similarity_engine = (
                    SimilarityEngine(
                        semantic_df,
                        model=self.similarity_model,
                        embeddings=embeddings,
                        dataset_hash=dataset_hash
                    )
                )

            else:

                print(
                    "Generating new "
                    "Sentence-BERT embeddings..."
                )

                embeddings = (
                    temp_similarity_engine
                    .build_embeddings()
                )

                # Cache embeddings using the dataset hash.
                self.embedding_cache[
                    dataset_hash
                ] = embeddings

                # Keep this engine as the active engine.
                self.similarity_engine = (
                    temp_similarity_engine
                )

            # ====================================================
            # CLUSTERING
            # ====================================================

            self.clusterer = (
                ComplaintClusterer()
            )

            cluster_labels = (
                self.clusterer.fit_predict(
                    embeddings
                )
            )

        except Exception as e:

            # Semantic analysis should not stop
            # the other NLP components.

            self.similarity_engine = None
            self.clusterer = None

            cluster_labels = [
                None
                for _ in range(
                    len(working_df)
                )
            ]

            print(
                f"Semantic analysis error: {e}"
            )

        # ========================================================
        # INDIVIDUAL COMPLAINT ANALYSIS
        # ========================================================

        results = []

        for index, row in working_df.iterrows():

            text = row[text_column].strip()

            try:

                # =================================================
                # PRODUCT CLASSIFICATION
                # =================================================

                product_result = (
                    self.product_classifier.predict(
                        text
                    )
                )

                product = (
                    product_result["product"]
                )

                # =================================================
                # SENTIMENT
                # =================================================

                sentiment_result = (
                    self.sentiment_analyzer.analyze(
                        text
                    )
                )

                # =================================================
                # SEVERITY
                # =================================================

                severity_result = (
                    calculate_severity(
                        text
                    )
                )

                # =================================================
                # KEY ISSUES
                # =================================================

                key_issues = (
                    self.key_issue_extractor.extract(
                        text
                    )
                )

                # =================================================
                # DEPARTMENT
                # =================================================

                department = (
                    recommend_department(
                        product
                    )
                )

                # =================================================
                # PRESERVE ORIGINAL CSV
                # =================================================

                result = {}

                for column in working_df.columns:

                    if column == text_column:

                        result["Complaint"] = text

                    else:

                        result[column] = (
                            row[column]
                        )

                # =================================================
                # NLP RESULTS
                # =================================================

                result["Predicted Product"] = (
                    product
                )

                result["Product Confidence"] = (
                    product_result[
                        "confidence"
                    ]
                )

                result["Sentiment"] = (
                    sentiment_result[
                        "sentiment"
                    ]
                )

                result["Sentiment Confidence"] = (
                    sentiment_result[
                        "confidence"
                    ]
                )

                result["Severity"] = (
                    severity_result[
                        "severity"
                    ]
                )

                result["Severity Score"] = (
                    severity_result[
                        "score"
                    ]
                )

                result["Escalation Detected"] = (
                    severity_result[
                        "escalation_detected"
                    ]
                )

                result["Key Issues"] = (
                    ", ".join(
                        item["keyword"]
                        for item in key_issues
                    )
                )

                result["Department"] = (
                    department
                )

                # =================================================
                # SEMANTIC CLUSTER
                # =================================================

                result["Cluster ID"] = (
                    cluster_labels[index]
                )

                results.append(result)

            except Exception as e:

                # =================================================
                # PRESERVE ORIGINAL CSV
                # =================================================

                result = {}

                for column in working_df.columns:

                    if column == text_column:

                        result["Complaint"] = text

                    else:

                        result[column] = (
                            row[column]
                        )

                # =================================================
                # ERROR VALUES
                # =================================================

                result["Predicted Product"] = (
                    "Error"
                )

                result["Product Confidence"] = 0

                result["Sentiment"] = (
                    "Error"
                )

                result["Sentiment Confidence"] = 0

                result["Severity"] = (
                    "Error"
                )

                result["Severity Score"] = 0

                result["Escalation Detected"] = (
                    False
                )

                result["Key Issues"] = ""

                result["Department"] = (
                    "Error"
                )

                result["Cluster ID"] = (
                    cluster_labels[index]
                )

                print(
                    f"Complaint {index} error: {e}"
                )

                results.append(result)

        return pd.DataFrame(
            results
        )

    # ============================================================
    # SEMANTIC SIMILARITY
    # ============================================================

    def find_similar_complaints(
        self,
        text,
        max_results=5,
        similarity_threshold=0.40
    ):

        if self.similarity_engine is None:

            raise ValueError(
                "No dataset has been analyzed yet."
            )

        return (
            self.similarity_engine.find_similar(
                text,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
        )

    # ============================================================
    # CLUSTER INFORMATION
    # ============================================================

    def get_cluster_count(self):

        if self.clusterer is None:

            return 0

        return (
            self.clusterer.get_cluster_count()
        )

    def get_cluster_sizes(
        self,
        labels
    ):

        if self.clusterer is None:

            return {}

        if labels is None:

            raise ValueError(
                "Cluster labels must be provided."
            )

        return (
            self.clusterer.get_cluster_sizes(
                labels
            )
        )

    # ============================================================
    # SILHOUETTE SCORE
    # ============================================================

    def get_clustering_score(self):

        if self.clusterer is None:

            return None

        return (
            self.clusterer.get_silhouette_score()
        )
