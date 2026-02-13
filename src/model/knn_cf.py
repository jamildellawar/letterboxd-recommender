"""KNN Collaborative Filtering model using MovieLens user-user similarity."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.metrics.pairwise import cosine_similarity


class KNNCFModel:
    """User-user KNN collaborative filtering on MovieLens ratings.

    For a guest user, builds a sparse rating vector from their tmdb_id ratings,
    finds the K most similar MovieLens users by cosine similarity, and predicts
    scores for candidate movies as a weighted average of neighbor ratings.
    """

    def __init__(self, k: int = 50):
        self.k = k
        self.user_item_matrix: csr_matrix | None = None
        self.movie_id_to_idx: dict[int, int] = {}
        self.movie_idx_to_id: dict[int, int] = {}
        self.ml_to_tmdb: dict[int, int] = {}
        self.tmdb_to_ml: dict[int, int] = {}

    def fit(
        self,
        user_item_matrix: csr_matrix,
        movie_id_to_idx: dict[int, int],
        movie_idx_to_id: dict[int, int],
        ml_to_tmdb: dict[int, int],
        tmdb_to_ml: dict[int, int],
    ) -> "KNNCFModel":
        """Store the user-item matrix and ID mappings."""
        self.user_item_matrix = user_item_matrix
        self.movie_id_to_idx = movie_id_to_idx
        self.movie_idx_to_id = movie_idx_to_id
        self.ml_to_tmdb = ml_to_tmdb
        self.tmdb_to_ml = tmdb_to_ml
        return self

    def predict(
        self,
        guest_ratings: dict[int, float],
        candidate_tmdb_ids: list[int] | set[int] | None = None,
    ) -> dict[int, float]:
        """Predict scores for candidates based on KNN collaborative filtering.

        Args:
            guest_ratings: {tmdb_id: rating} from the guest user.
            candidate_tmdb_ids: Optional subset of tmdb_ids to score.
                If None, scores all movies in the matrix.

        Returns:
            {tmdb_id: predicted_score} for each scorable candidate.
        """
        if self.user_item_matrix is None:
            raise RuntimeError("Must call fit() before predict().")

        n_movies = self.user_item_matrix.shape[1]

        # Map guest's tmdb_ids to matrix column indices
        guest_indices = []
        guest_values = []
        for tmdb_id, rating in guest_ratings.items():
            ml_id = self.tmdb_to_ml.get(int(tmdb_id))
            if ml_id is not None and ml_id in self.movie_id_to_idx:
                guest_indices.append(self.movie_id_to_idx[ml_id])
                guest_values.append(rating)

        if not guest_indices:
            return {}

        # Build sparse guest vector
        guest_vector = csr_matrix(
            (guest_values, ([0] * len(guest_indices), guest_indices)),
            shape=(1, n_movies),
        )

        # Cosine similarity with all MovieLens users
        similarities = cosine_similarity(guest_vector, self.user_item_matrix).flatten()

        # Top-K neighbors
        top_k_indices = np.argsort(similarities)[::-1][: self.k]
        top_k_sims = similarities[top_k_indices]

        # Filter out zero/negative similarities
        valid_mask = top_k_sims > 0
        top_k_indices = top_k_indices[valid_mask]
        top_k_sims = top_k_sims[valid_mask]

        if len(top_k_indices) == 0:
            return {}

        # Weighted average of neighbor ratings for each candidate
        neighbor_matrix = self.user_item_matrix[top_k_indices]
        weighted_sum = (neighbor_matrix.T.multiply(top_k_sims)).T.sum(axis=0)
        weighted_sum = np.asarray(weighted_sum).flatten()

        # Count how many neighbors rated each movie (for confidence)
        neighbor_rated = (neighbor_matrix > 0).T.multiply(top_k_sims).T.sum(axis=0)
        neighbor_rated = np.asarray(neighbor_rated).flatten()

        # Avoid division by zero
        scores = np.zeros(n_movies)
        rated_mask = neighbor_rated > 0
        scores[rated_mask] = weighted_sum[rated_mask] / neighbor_rated[rated_mask]

        # Build output: map movie indices back to tmdb_ids
        results = {}
        target_tmdb_ids = set(candidate_tmdb_ids) if candidate_tmdb_ids else None

        for col_idx in range(n_movies):
            if scores[col_idx] <= 0:
                continue
            ml_id = self.movie_idx_to_id.get(col_idx)
            if ml_id is None:
                continue
            tmdb_id = self.ml_to_tmdb.get(ml_id)
            if tmdb_id is None:
                continue
            if target_tmdb_ids and tmdb_id not in target_tmdb_ids:
                continue
            # Skip movies the guest already rated
            if tmdb_id in guest_ratings:
                continue
            results[tmdb_id] = float(scores[col_idx])

        return results

    def save_artifacts(self, output_dir: Path) -> None:
        """Save model artifacts to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_npz(output_dir / "user_item_matrix.npz", self.user_item_matrix)

        mappings = {
            "k": self.k,
            "movie_id_to_idx": self.movie_id_to_idx,
            "movie_idx_to_id": self.movie_idx_to_id,
            "ml_to_tmdb": self.ml_to_tmdb,
            "tmdb_to_ml": self.tmdb_to_ml,
        }
        with open(output_dir / "knn_mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)

        # Save matrix size info
        shape = self.user_item_matrix.shape
        nnz = self.user_item_matrix.nnz
        size_mb = (output_dir / "user_item_matrix.npz").stat().st_size / (1024 * 1024)
        print(f"  Saved KNN artifacts: {shape[0]:,} users x {shape[1]:,} movies, {nnz:,} ratings ({size_mb:.1f} MB)")

    @classmethod
    def load_artifacts(cls, artifacts_dir: Path) -> "KNNCFModel":
        """Load model from saved artifacts."""
        artifacts_dir = Path(artifacts_dir)

        matrix = load_npz(artifacts_dir / "user_item_matrix.npz")

        with open(artifacts_dir / "knn_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)

        model = cls(k=mappings["k"])
        model.user_item_matrix = matrix
        model.movie_id_to_idx = mappings["movie_id_to_idx"]
        model.movie_idx_to_id = mappings["movie_idx_to_id"]
        model.ml_to_tmdb = mappings["ml_to_tmdb"]
        model.tmdb_to_ml = mappings["tmdb_to_ml"]

        return model
