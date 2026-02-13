"""LightFM hybrid collaborative filtering model."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from lightfm import LightFM
from lightfm.data import Dataset


class LightFMRecommender:
    """Hybrid CF + content model using LightFM.

    Trains on MovieLens ratings with item features (genres, keywords, directors)
    from TMDB metadata. For guest users, computes a user embedding as a weighted
    average of item embeddings and scores candidates via dot product.
    """

    def __init__(
        self,
        n_components: int = 64,
        loss: str = "warp",
        epochs: int = 30,
        positive_threshold: float = 3.5,
    ):
        self.n_components = n_components
        self.loss = loss
        self.epochs = epochs
        self.positive_threshold = positive_threshold
        self.model: LightFM | None = None
        self.dataset: Dataset | None = None
        self.item_id_map: dict[int, int] = {}  # tmdb_id -> internal index
        self.item_idx_to_tmdb: dict[int, int] = {}  # internal index -> tmdb_id
        self.item_features_matrix: csr_matrix | None = None

    def fit(
        self,
        ratings_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        id_mappings: dict,
    ) -> "LightFMRecommender":
        """Train LightFM on MovieLens ratings filtered to candidate pool.

        Args:
            ratings_df: MovieLens ratings DataFrame (userId, movieId, rating).
            candidates_df: Candidate movies with TMDB metadata.
            id_mappings: From build_id_mappings() with tmdb_to_ml, ml_to_tmdb.
        """
        tmdb_to_ml = id_mappings["tmdb_to_ml"]
        ml_to_tmdb = id_mappings["ml_to_tmdb"]
        valid_ml_ids = set(tmdb_to_ml.values())

        # Filter ratings to shared movies
        filtered_ratings = ratings_df[ratings_df["movieId"].isin(valid_ml_ids)].copy()
        # Only positive interactions
        positive = filtered_ratings[filtered_ratings["rating"] >= self.positive_threshold]
        print(f"  Positive interactions (>= {self.positive_threshold}): {len(positive):,}")

        # Build item features from TMDB metadata
        item_features_list = []
        candidate_lookup = candidates_df.set_index("tmdb_id")

        for tmdb_id in id_mappings["shared_tmdb_ids"]:
            if tmdb_id not in candidate_lookup.index:
                continue
            row = candidate_lookup.loc[tmdb_id]
            features = []

            # Genres
            genres = row.get("genres", [])
            if isinstance(genres, list):
                features.extend(f"genre:{g}" for g in genres)

            # Director
            directors = row.get("director", [])
            if isinstance(directors, list):
                features.extend(f"director:{d}" for d in directors[:2])

            # Top keywords
            keywords = row.get("keywords", [])
            if isinstance(keywords, list):
                features.extend(f"keyword:{k}" for k in keywords[:5])

            if features:
                item_features_list.append((tmdb_id, features))

        # Collect all unique feature tags
        all_features = set()
        for _, feats in item_features_list:
            all_features.update(feats)

        # Build LightFM Dataset
        user_ids = positive["userId"].unique()
        item_tmdb_ids = list(id_mappings["shared_tmdb_ids"])

        self.dataset = Dataset()
        self.dataset.fit(
            users=user_ids,
            items=item_tmdb_ids,
            item_features=all_features,
        )

        # Build interactions
        interactions_data = []
        for _, row in positive.iterrows():
            ml_id = int(row["movieId"])
            tmdb_id = ml_to_tmdb.get(ml_id)
            if tmdb_id is not None:
                interactions_data.append((row["userId"], tmdb_id))

        interactions, _ = self.dataset.build_interactions(interactions_data)
        print(f"  Interactions matrix: {interactions.shape}")

        # Build item features
        if item_features_list:
            self.item_features_matrix = self.dataset.build_item_features(
                item_features_list, normalize=True
            )
            print(f"  Item features: {self.item_features_matrix.shape}")
        else:
            self.item_features_matrix = None

        # Train
        self.model = LightFM(
            no_components=self.n_components,
            loss=self.loss,
            random_state=42,
        )
        print(f"  Training LightFM ({self.loss}, {self.n_components}d, {self.epochs} epochs)...")
        self.model.fit(
            interactions,
            item_features=self.item_features_matrix,
            epochs=self.epochs,
            num_threads=4,
            verbose=True,
        )

        # Store ID mappings
        _, _, item_id_map_raw = self.dataset.mapping()[:3]
        self.item_id_map = {int(k): int(v) for k, v in item_id_map_raw.items()}
        self.item_idx_to_tmdb = {v: k for k, v in self.item_id_map.items()}

        return self

    def predict(
        self,
        guest_ratings: dict[int, float],
        candidate_tmdb_ids: list[int] | set[int] | None = None,
    ) -> dict[int, float]:
        """Predict scores for candidates using LightFM embeddings.

        Computes a guest user embedding as a weighted average of item embeddings,
        then scores candidates via dot product.

        Args:
            guest_ratings: {tmdb_id: rating} from the guest user.
            candidate_tmdb_ids: Optional subset of tmdb_ids to score.

        Returns:
            {tmdb_id: predicted_score} for each scorable candidate.
        """
        if self.model is None:
            raise RuntimeError("Must call fit() before predict().")

        # Get item embeddings + biases from the trained model
        item_embeddings = self.model.get_item_representations(
            features=self.item_features_matrix
        )
        item_biases, item_factors = item_embeddings

        # Build user embedding from guest's rated movies
        weights = []
        embeddings = []
        for tmdb_id, rating in guest_ratings.items():
            idx = self.item_id_map.get(int(tmdb_id))
            if idx is not None:
                weight = rating - 2.5  # center around neutral
                weights.append(weight)
                embeddings.append(item_factors[idx])

        if not embeddings:
            return {}

        weights = np.array(weights)
        embeddings = np.array(embeddings)

        # Weighted average of item embeddings
        user_embedding = np.average(embeddings, axis=0, weights=np.abs(weights))
        # Apply sign: positive weights attract, negative repel
        sign_weights = np.sign(weights)
        user_embedding_signed = np.average(embeddings, axis=0, weights=np.abs(weights) * sign_weights)
        user_embedding = user_embedding_signed
        norm = np.linalg.norm(user_embedding)
        if norm > 0:
            user_embedding = user_embedding / norm

        # Score candidates via dot product + item bias
        results = {}
        target_tmdb_ids = set(candidate_tmdb_ids) if candidate_tmdb_ids else set(self.item_id_map.keys())

        for tmdb_id in target_tmdb_ids:
            if tmdb_id in guest_ratings:
                continue
            idx = self.item_id_map.get(int(tmdb_id))
            if idx is None:
                continue
            score = float(np.dot(user_embedding, item_factors[idx]) + item_biases[idx])
            results[tmdb_id] = score

        return results

    def save_artifacts(self, output_dir: Path) -> None:
        """Save model artifacts to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "lightfm_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        with open(output_dir / "lightfm_mappings.pkl", "wb") as f:
            pickle.dump({
                "item_id_map": self.item_id_map,
                "item_idx_to_tmdb": self.item_idx_to_tmdb,
                "n_components": self.n_components,
            }, f)

        if self.item_features_matrix is not None:
            from scipy.sparse import save_npz
            save_npz(output_dir / "lightfm_item_features.npz", self.item_features_matrix)

        size_mb = sum(
            f.stat().st_size for f in output_dir.glob("lightfm_*")
        ) / (1024 * 1024)
        print(f"  Saved LightFM artifacts ({size_mb:.1f} MB)")

    @classmethod
    def load_artifacts(cls, artifacts_dir: Path) -> "LightFMRecommender":
        """Load model from saved artifacts."""
        artifacts_dir = Path(artifacts_dir)

        with open(artifacts_dir / "lightfm_model.pkl", "rb") as f:
            model_obj = pickle.load(f)

        with open(artifacts_dir / "lightfm_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)

        item_features = None
        features_path = artifacts_dir / "lightfm_item_features.npz"
        if features_path.exists():
            from scipy.sparse import load_npz
            item_features = load_npz(features_path)

        recommender = cls(n_components=mappings["n_components"])
        recommender.model = model_obj
        recommender.item_id_map = mappings["item_id_map"]
        recommender.item_idx_to_tmdb = mappings["item_idx_to_tmdb"]
        recommender.item_features_matrix = item_features

        return recommender
