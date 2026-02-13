"""Train all 3 CF models + content-based baseline, evaluate, and compare."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR
from src.data.movielens import download_movielens, load_ratings, build_id_mappings
from src.features.vectorize import MovieVectorizer
from src.features.profile import build_taste_profile
from src.model.evaluate import evaluate_cf_model, leave_one_out_eval
from src.model.knn_cf import KNNCFModel
from src.data.movielens import build_user_item_matrix


class ContentBasedWrapper:
    """Wraps the existing content-based model with a predict() interface for evaluation."""

    def __init__(self, vectorizer: MovieVectorizer, rated_df: pd.DataFrame):
        self.vectorizer = vectorizer
        self.rated_df = rated_df
        self._candidate_vectors = None
        self._candidate_tmdb_ids = None

    def set_candidates(self, candidates_df: pd.DataFrame):
        self._candidate_vectors = self.vectorizer.transform(candidates_df)
        self._candidate_tmdb_ids = candidates_df["tmdb_id"].values

    def predict(
        self,
        guest_ratings: dict[int, float],
        candidate_tmdb_ids=None,
    ) -> dict[int, float]:
        from sklearn.metrics.pairwise import cosine_similarity

        # Build a mini rated_df from guest_ratings
        rated_subset = self.rated_df[self.rated_df["tmdb_id"].isin(guest_ratings.keys())].copy()
        if rated_subset.empty:
            return {}

        rated_subset["memberRating"] = rated_subset["tmdb_id"].map(guest_ratings)

        rated_vectors = self.vectorizer.transform(rated_subset)
        profile = build_taste_profile(rated_vectors, rated_subset["memberRating"])

        if self._candidate_vectors is None:
            return {}

        scores = cosine_similarity(
            self._candidate_vectors, profile.reshape(1, -1)
        ).flatten()

        results = {}
        target = set(candidate_tmdb_ids) if candidate_tmdb_ids else None
        for i, tmdb_id in enumerate(self._candidate_tmdb_ids):
            tmdb_id = int(tmdb_id)
            if tmdb_id in guest_ratings:
                continue
            if target and tmdb_id not in target:
                continue
            results[tmdb_id] = float(scores[i])

        return results


def main():
    print("=" * 60)
    print("  Model Comparison: Content-Based vs CF Models")
    print("=" * 60)

    # Load data
    print("\n--- Loading Data ---")
    rated_path = PROCESSED_DIR / "rated_movies.parquet"
    if not rated_path.exists():
        print("Error: rated_movies.parquet not found. Run weekly_update first.")
        return

    rated_df = pd.read_parquet(rated_path)
    print(f"Rated movies: {len(rated_df)}")

    candidates_path = PROCESSED_DIR / "candidates.parquet"
    if not candidates_path.exists():
        print("Error: candidates.parquet not found. Run weekly_update first.")
        return

    candidates_df = pd.read_parquet(candidates_path)
    print(f"Candidates: {len(candidates_df)}")

    # Download MovieLens if needed
    print("\n--- MovieLens Data ---")
    download_movielens()
    ratings_df = load_ratings()
    print(f"MovieLens ratings: {len(ratings_df):,}")

    # Build ID mappings
    print("\nBuilding ID mappings...")
    id_mappings = build_id_mappings(candidates_df)
    candidate_tmdb_ids = set(int(x) for x in candidates_df["tmdb_id"].values)
    shared_tmdb_ids = id_mappings["shared_tmdb_ids"]

    # Fit vectorizer
    print("\nFitting MovieVectorizer...")
    vectorizer = MovieVectorizer()
    vectorizer.fit(rated_df)

    results = {}

    # ── Content-Based Baseline ──
    print("\n--- Model 1: Content-Based ---")
    cb_model = ContentBasedWrapper(vectorizer, rated_df)
    cb_model.set_candidates(candidates_df)
    cb_metrics = evaluate_cf_model(cb_model, rated_df, candidate_tmdb_ids)
    results["Content-Based"] = cb_metrics
    print(f"  {cb_metrics}")

    # Also run leave-one-out for content-based
    loo = leave_one_out_eval(rated_df, vectorizer)
    print(f"  Leave-one-out: hit_rate={loo['hit_rate']}, MRR={loo['mean_reciprocal_rank']}")

    # ── KNN CF ──
    print("\n--- Model 2: KNN CF ---")
    valid_ml_ids = set(id_mappings["tmdb_to_ml"].values())
    matrix_data = build_user_item_matrix(ratings_df, valid_ml_ids)

    knn_model = KNNCFModel(k=50)
    knn_model.fit(
        matrix_data["matrix"],
        matrix_data["movie_id_to_idx"],
        matrix_data["movie_idx_to_id"],
        id_mappings["ml_to_tmdb"],
        id_mappings["tmdb_to_ml"],
    )

    knn_metrics = evaluate_cf_model(knn_model, rated_df, shared_tmdb_ids)
    results["KNN CF"] = knn_metrics
    print(f"  {knn_metrics}")

    knn_dir = MODELS_DIR / "knn_cf"
    knn_model.save_artifacts(knn_dir)

    # ── LightFM ──
    print("\n--- Model 3: LightFM ---")
    try:
        from src.model.lightfm_model import LightFMRecommender

        lfm_model = LightFMRecommender(n_components=64, epochs=30)
        lfm_model.fit(ratings_df, candidates_df, id_mappings)

        lfm_metrics = evaluate_cf_model(lfm_model, rated_df, shared_tmdb_ids)
        results["LightFM"] = lfm_metrics
        print(f"  {lfm_metrics}")

        lfm_dir = MODELS_DIR / "lightfm"
        lfm_model.save_artifacts(lfm_dir)
    except ImportError:
        print("  Skipping: lightfm not installed (pip install lightfm)")
        results["LightFM"] = {"hit_rate": "N/A", "mrr": "N/A", "ndcg": "N/A", "coverage": "N/A"}

    # ── Two-Tower ──
    print("\n--- Model 4: Two-Tower ---")
    try:
        from src.model.two_tower import TwoTowerModel

        tt_model = TwoTowerModel(embed_dim=64, epochs=10)
        tt_model.fit(ratings_df, candidates_df, id_mappings, vectorizer)

        tt_metrics = evaluate_cf_model(tt_model, rated_df, shared_tmdb_ids)
        results["Two-Tower"] = tt_metrics
        print(f"  {tt_metrics}")

        tt_dir = MODELS_DIR / "two_tower"
        tt_model.export_onnx(tt_dir)
    except ImportError as e:
        print(f"  Skipping: {e}")
        results["Two-Tower"] = {"hit_rate": "N/A", "mrr": "N/A", "ndcg": "N/A", "coverage": "N/A"}

    # ── Comparison Table ──
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"\n{'Model':<16} {'Hit Rate':>10} {'MRR':>10} {'NDCG':>10} {'Coverage':>10}")
    print("-" * 58)
    for name, metrics in results.items():
        hr = metrics.get("hit_rate", "N/A")
        mrr = metrics.get("mrr", "N/A")
        ndcg = metrics.get("ndcg", "N/A")
        cov = metrics.get("coverage", "N/A")
        hr_s = f"{hr:.4f}" if isinstance(hr, float) else hr
        mrr_s = f"{mrr:.4f}" if isinstance(mrr, float) else mrr
        ndcg_s = f"{ndcg:.4f}" if isinstance(ndcg, float) else ndcg
        cov_s = f"{cov:.4f}" if isinstance(cov, float) else cov
        print(f"{name:<16} {hr_s:>10} {mrr_s:>10} {ndcg_s:>10} {cov_s:>10}")

    # ── Top-30 Qualitative Comparison ──
    print("\n" + "=" * 60)
    print("  Top-30 Recommendations (Qualitative)")
    print("=" * 60)

    guest_ratings = dict(zip(
        rated_df["tmdb_id"].astype(int),
        rated_df["memberRating"].astype(float),
    ))

    title_lookup = dict(zip(
        candidates_df["tmdb_id"].astype(int),
        candidates_df["title"],
    ))

    models_to_show = {"KNN CF": knn_model}
    if "LightFM" in results and isinstance(results["LightFM"].get("hit_rate"), float):
        models_to_show["LightFM"] = lfm_model
    if "Two-Tower" in results and isinstance(results["Two-Tower"].get("hit_rate"), float):
        models_to_show["Two-Tower"] = tt_model

    for name, model in models_to_show.items():
        print(f"\n--- {name} Top 30 ---")
        scores = model.predict(guest_ratings, shared_tmdb_ids)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:30]
        for i, (tmdb_id, score) in enumerate(ranked, 1):
            title = title_lookup.get(tmdb_id, f"tmdb:{tmdb_id}")
            print(f"  {i:2d}. {title} (score={score:.4f})")

    print(f"\nAll model artifacts saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
