"""MovieLens 25M data pipeline for collaborative filtering models."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.config import MOVIELENS_DIR, RAW_DIR

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP_PATH = RAW_DIR / "ml-25m.zip"


def download_movielens(force: bool = False) -> Path:
    """Download and extract MovieLens 25M dataset.

    Returns path to extracted directory (data/raw/ml-25m/).
    """
    ratings_path = MOVIELENS_DIR / "ratings.csv"

    if ratings_path.exists() and not force:
        print(f"MovieLens already extracted at {MOVIELENS_DIR}")
        return MOVIELENS_DIR

    # Download ZIP
    if not ZIP_PATH.exists() or force:
        print(f"Downloading MovieLens 25M (~250MB)...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)
        size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
        print(f"  Downloaded {size_mb:.0f} MB")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)
    print(f"  Extracted to {MOVIELENS_DIR}")

    return MOVIELENS_DIR


def load_ratings() -> pd.DataFrame:
    """Load MovieLens ratings.csv (userId, movieId, rating, timestamp)."""
    path = MOVIELENS_DIR / "ratings.csv"
    if not path.exists():
        raise FileNotFoundError(f"MovieLens ratings not found at {path}. Run download_movielens() first.")
    return pd.read_csv(path)


def load_links() -> pd.DataFrame:
    """Load MovieLens links.csv (movieId, imdbId, tmdbId)."""
    path = MOVIELENS_DIR / "links.csv"
    if not path.exists():
        raise FileNotFoundError(f"MovieLens links not found at {path}. Run download_movielens() first.")
    df = pd.read_csv(path)
    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce")
    df = df.dropna(subset=["tmdbId"])
    df["tmdbId"] = df["tmdbId"].astype(int)
    return df


def build_id_mappings(candidates_df: pd.DataFrame) -> dict:
    """Intersect MovieLens movies with our candidate pool via tmdbId.

    Returns dict with:
        tmdb_to_ml: {tmdb_id -> movieLens_movieId}
        ml_to_tmdb: {movieLens_movieId -> tmdb_id}
        shared_tmdb_ids: set of tmdb_ids in both datasets
    """
    links = load_links()

    candidate_tmdb_ids = set(int(x) for x in candidates_df["tmdb_id"].values)
    ml_tmdb_ids = set(int(x) for x in links["tmdbId"].values)
    shared_tmdb_ids = candidate_tmdb_ids & ml_tmdb_ids

    # Build bidirectional mappings
    links_shared = links[links["tmdbId"].isin(shared_tmdb_ids)]
    tmdb_to_ml = dict(zip(links_shared["tmdbId"].astype(int), links_shared["movieId"].astype(int)))
    ml_to_tmdb = dict(zip(links_shared["movieId"].astype(int), links_shared["tmdbId"].astype(int)))

    print(f"  Candidate pool: {len(candidate_tmdb_ids)} movies")
    print(f"  MovieLens:      {len(ml_tmdb_ids)} movies")
    print(f"  Overlap:        {len(shared_tmdb_ids)} movies")

    return {
        "tmdb_to_ml": tmdb_to_ml,
        "ml_to_tmdb": ml_to_tmdb,
        "shared_tmdb_ids": shared_tmdb_ids,
    }


def build_user_item_matrix(
    ratings_df: pd.DataFrame,
    valid_ml_movie_ids: set[int],
) -> dict:
    """Build sparse user-item matrix filtered to candidate pool movies.

    Args:
        ratings_df: MovieLens ratings DataFrame.
        valid_ml_movie_ids: Set of MovieLens movieIds to include.

    Returns dict with:
        matrix: scipy CSR matrix (n_users x n_movies)
        user_id_to_idx: {userId -> row index}
        user_idx_to_id: {row index -> userId}
        movie_id_to_idx: {movieId -> col index}
        movie_idx_to_id: {col index -> movieId}
    """
    # Filter to valid movies
    filtered = ratings_df[ratings_df["movieId"].isin(valid_ml_movie_ids)].copy()
    print(f"  Ratings after filtering to candidate pool: {len(filtered):,}")

    # Build index mappings
    unique_users = sorted(filtered["userId"].unique())
    unique_movies = sorted(filtered["movieId"].unique())

    user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    user_idx_to_id = {i: uid for uid, i in user_id_to_idx.items()}
    movie_id_to_idx = {mid: j for j, mid in enumerate(unique_movies)}
    movie_idx_to_id = {j: mid for mid, j in movie_id_to_idx.items()}

    # Build sparse matrix
    row_indices = filtered["userId"].map(user_id_to_idx).values
    col_indices = filtered["movieId"].map(movie_id_to_idx).values
    values = filtered["rating"].values.astype(np.float32)

    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(unique_users), len(unique_movies)),
    )

    print(f"  Matrix shape: {matrix.shape[0]:,} users x {matrix.shape[1]:,} movies")
    print(f"  Sparsity: {1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")

    return {
        "matrix": matrix,
        "user_id_to_idx": user_id_to_idx,
        "user_idx_to_id": user_idx_to_id,
        "movie_id_to_idx": movie_id_to_idx,
        "movie_idx_to_id": movie_idx_to_id,
    }
