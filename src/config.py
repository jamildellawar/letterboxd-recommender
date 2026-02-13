"""Constants and paths for the letterboxd recommender."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
HISTORY_DIR = DATA_DIR / "history"
MOVIELENS_DIR = RAW_DIR / "ml-25m"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, HISTORY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Letterboxd ───────────────────────────────────────────────────────────────
LETTERBOXD_USERNAME = "jamildellawar"
LETTERBOXD_RSS_URL = f"https://letterboxd.com/{LETTERBOXD_USERNAME}/rss/"
LETTERBOXD_FILMS_URL = f"https://letterboxd.com/{LETTERBOXD_USERNAME}/films/page/"

# ── TMDB ─────────────────────────────────────────────────────────────────────
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_CACHE_PATH = RAW_DIR / "tmdb_cache.json"

# ── AWS / DynamoDB ───────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
LETTERBOXD_RECS_TABLE = os.getenv("LETTERBOXD_RECS_TABLE_NAME", "LetterboxdRecs")
LETTERBOXD_PROFILE_TABLE = os.getenv("LETTERBOXD_PROFILE_TABLE_NAME", "LetterboxdProfile")

# ── Model ────────────────────────────────────────────────────────────────────
TOP_N_RECOMMENDATIONS = 30
TOP_N_DIRECTORS = 10  # one-hot threshold for directors (lower = less sparse with small dataset)
CANDIDATE_POOL_SIZE = 10000
