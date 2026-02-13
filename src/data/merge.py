"""Combine Letterboxd ratings with TMDB metadata into a single dataset."""

import pandas as pd

from src.config import PROCESSED_DIR
from src.data.letterboxd import load_or_fetch_history
from src.data.tmdb import enrich_all


def build_rated_movies() -> pd.DataFrame:
    """Build the rated movies dataset by merging Letterboxd history with TMDB metadata.

    Returns DataFrame saved to data/processed/rated_movies.parquet.
    """
    history = load_or_fetch_history()
    print(f"Loaded {len(history)} rated films from Letterboxd")

    enriched = enrich_all(history)
    df = pd.DataFrame(enriched)

    if df.empty:
        print("Warning: No enriched movies. Check TMDB API key.")
        return df

    # Ensure required columns
    required = ["tmdb_id", "title", "year", "memberRating", "genres", "director"]
    for col in required:
        if col not in df.columns:
            df[col] = None

    output_path = PROCESSED_DIR / "rated_movies.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rated movies to {output_path}")
    return df


def load_rated_movies() -> pd.DataFrame:
    """Load the processed rated movies dataset."""
    path = PROCESSED_DIR / "rated_movies.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_catalog.py first."
        )
    return pd.read_parquet(path)
