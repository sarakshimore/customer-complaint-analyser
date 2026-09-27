import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ComplaintClusterer:

    def __init__(
        self,
        min_clusters=2,
        max_clusters=15,
        random_state=42
    ):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state

        self.model = None
        self.n_clusters = None
        self.silhouette_score = None

    def determine_cluster_count(self, embeddings):
        """
        Automatically determine a suitable number of clusters
        based on the dataset using silhouette score.
        """

        embeddings = np.asarray(embeddings)

        n_samples = len(embeddings)

        if n_samples < 2:
            raise ValueError(
                "At least 2 complaints are required for clustering."
            )

        # Maximum possible clusters must be smaller than
        # the number of samples.
        max_k = min(
            self.max_clusters,
            n_samples - 1
        )

        min_k = min(
            self.min_clusters,
            max_k
        )

        if min_k > max_k:
            return min_k

        best_k = min_k
        best_score = -1

        for k in range(min_k, max_k + 1):

            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )

            labels = model.fit_predict(embeddings)

            # Silhouette score requires at least
            # two different clusters.
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(
                embeddings,
                labels
            )

            if score > best_score:
                best_score = score
                best_k = k

        self.n_clusters = best_k
        self.silhouette_score = best_score

        return best_k

    def fit_predict(self, embeddings):
        """
        Automatically determine the number of clusters
        and assign every complaint to a cluster.
        """

        if embeddings is None:
            raise ValueError(
                "Embeddings cannot be None."
            )

        embeddings = np.asarray(embeddings)

        if len(embeddings) == 0:
            raise ValueError(
                "No embeddings were provided."
            )

        # Determine dataset-aware cluster count.
        n_clusters = self.determine_cluster_count(
            embeddings
        )

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        cluster_labels = self.model.fit_predict(
            embeddings
        )

        return cluster_labels

    def predict(self, embeddings):
        """
        Assign new complaint embeddings to the
        already fitted clusters.
        """

        if self.model is None:
            raise ValueError(
                "Clusterer has not been fitted yet."
            )

        embeddings = np.asarray(embeddings)

        return self.model.predict(
            embeddings
        )

    def get_cluster_count(self):
        """
        Return the number of clusters created.
        """

        if self.model is None:
            return 0

        return self.model.n_clusters

    def get_cluster_sizes(self, labels):
        """
        Return the number of complaints in each cluster.
        """

        labels = np.asarray(labels)

        unique, counts = np.unique(
            labels,
            return_counts=True
        )

        return {
            int(cluster): int(count)
            for cluster, count in zip(
                unique,
                counts
            )
        }

    def get_silhouette_score(self):
        """
        Return the silhouette score used to select
        the number of clusters.
        """

        if self.silhouette_score is None:
            return None

        return self.silhouette_score