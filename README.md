# Letterboxd Recommender

Movie recommendation system built from my Letterboxd watch history. Generates personalized recommendations using content-based filtering + collaborative filtering (KNN on MovieLens 25M), served via DynamoDB and a guest-facing Lambda.

## How It Works

Three recommendation signals are blended together:

| Signal | Weight | Source | What it captures |
|--------|--------|--------|-----------------|
| **Content-based** | 50% | TMDB metadata (genres, directors, cast, keywords, plot) | "Movies that look like what you like" |
| **Collaborative filtering** | 30% | KNN on 162K MovieLens users | "People with similar taste also liked..." |
| **Quality** | 20% | Letterboxd community ratings (fallback: TMDB Bayesian) | Filters out niche films with inflated ratings |

Movies with a Letterboxd community rating below 3.0/5 are excluded entirely.

## Architecture

```
Automated Pipeline (every 2 days via GitHub Actions)
  weekly_update.py
    ├── Fetch new watches from Letterboxd RSS
    ├── Enrich with TMDB metadata
    ├── Fetch Letterboxd community ratings for candidates
    ├── Train KNN CF model (MovieLens 25M)
    ├── Generate 30 main recs + genre picks + streaming picks
    ├── Write to DynamoDB (personal dashboard)
    └── Upload cache + model artifacts to S3 (for guest Lambda)

Personal Recommendations (jamildellawar.com)
  DynamoDB → /api/movie-recs → "For You", "Genre Picks", "Streaming" tabs

Guest "Try It" Feature (jamildellawar.com)
  User uploads Letterboxd CSV
    → POST /api/movie-recs/guest (creates async job)
    → Lambda loads S3 cache + CF model, generates recs
    → GET /api/movie-recs/guest?jobId=... (polls until complete)
    → Displays personalized recommendations for the guest
```

## Project Structure

```
src/
├── config.py                     # Paths, API keys, constants
├── data/
│   ├── letterboxd.py             # RSS feed, CSV export, scraping
│   ├── tmdb.py                   # TMDB API client + caching
│   ├── movielens.py              # MovieLens 25M download + sparse matrix
│   ├── merge.py                  # Combine Letterboxd + TMDB
│   └── history.py                # Recommendation history + feedback
├── features/
│   ├── vectorize.py              # MovieVectorizer (TF-IDF + multi-hot)
│   ├── profile.py                # Taste profile (positive - 0.5 * negative)
│   └── feedback.py               # Learn from past rec outcomes
├── model/
│   ├── recommend.py              # Main scoring + MMR diversity reranking
│   ├── evaluate.py               # Leave-one-out, correlation, CF K-fold
│   ├── knn_cf.py                 # KNN collaborative filtering model
│   ├── lightfm_model.py          # LightFM hybrid model (experimental)
│   └── two_tower.py              # Two-tower neural model (experimental)
├── deploy/
│   └── dynamo_writer.py          # Write recs to DynamoDB
└── scripts/
    ├── weekly_update.py          # Automated pipeline (runs every 2 days)
    ├── generate_recs.py          # Manual one-off rec generation
    ├── compare_models.py         # Train + evaluate all models
    ├── build_catalog.py          # One-time initial catalog build
    └── download_movielens.py     # Download MovieLens 25M dataset
```

## Setup

```bash
# Create environment (Python 3.11+)
conda create -n letterboxd python=3.12 -y
conda activate letterboxd

# Install core dependencies
pip install -e "."

# Optional: ML model training dependencies (torch, onnx, etc.)
pip install -e ".[ml]"
```

### Environment Variables

```bash
TMDB_API_KEY=...              # Required: TMDB API key
AWS_ACCESS_KEY_ID=...         # For DynamoDB + S3
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
GUEST_CACHE_BUCKET=jd-guest-recs-cache  # S3 bucket for guest Lambda cache
```

### First-Time Setup

1. Place your Letterboxd data export in `data/raw/letterboxd-<username>/`
2. Build the initial catalog:
   ```bash
   python -m src.scripts.build_catalog
   ```
3. Download MovieLens data (~250MB):
   ```bash
   python -m src.scripts.download_movielens
   ```

## Scripts

### `weekly_update.py` — Automated Pipeline

Runs every 2 days via GitHub Actions. This is the production pipeline.

```bash
GUEST_CACHE_BUCKET=jd-guest-recs-cache python -m src.scripts.weekly_update
```

What it does:
1. Fetches new watches from Letterboxd RSS, merges into catalog
2. Detects feedback from previously recommended movies you've since rated
3. Fetches ~1,600 candidate movies from TMDB (popular + top rated)
4. Fetches Letterboxd community ratings for all candidates
5. Trains KNN CF model on MovieLens 25M (162K users, filtered to candidate pool)
6. Generates 30 main recommendations, genre picks, and streaming picks
7. Writes everything to DynamoDB
8. Uploads TMDB cache, candidates, and model artifacts to S3

### `generate_recs.py` — Manual Generation

Quick one-off regeneration with evaluation metrics. Useful for testing changes.

```bash
python -m src.scripts.generate_recs
```

### `compare_models.py` — Model Comparison

Trains all available models and prints a comparison table.

```bash
python -m src.scripts.compare_models
```

Outputs hit rate, MRR, NDCG, and coverage for:
- Content-Based (baseline)
- KNN CF (production model)
- LightFM (requires `lightfm` — experimental)
- Two-Tower (requires `torch` — experimental)

### `build_catalog.py` — Initial Setup

One-time script to build the rated movies catalog from a Letterboxd CSV export.

```bash
python -m src.scripts.build_catalog
```

### `download_movielens.py` — Data Download

Downloads and extracts MovieLens 25M dataset. Idempotent.

```bash
python -m src.scripts.download_movielens
```

## Models

### KNN Collaborative Filtering (Production)

User-user KNN on MovieLens 25M ratings. For a guest user:
1. Maps their ratings to MovieLens movie IDs
2. Cosine similarity with 162K MovieLens users
3. Top-50 neighbors' weighted average ratings predict scores

Artifacts: ~14MB sparse matrix + ID mappings.

### LightFM Hybrid (Experimental)

Hybrid CF + content features using WARP loss. Requires `lightfm` (build issues on recent Python).

### Two-Tower Neural (Experimental)

Attention-based user tower + MLP item tower trained with BPR loss. Exports to ONNX for lightweight inference. Showed weaker results than KNN CF in testing — tended toward franchise-heavy recommendations.

## Guest Lambda

The guest recommendation Lambda (`personal-website/aws/cdk/lambda/guest-recs/`) is a self-contained pipeline that:
- Loads pre-cached TMDB data + candidates + CF model artifacts from S3
- Enriches guest ratings with TMDB metadata
- Generates recommendations using the same three-way blend
- Runs in a Docker container on AWS Lambda (ARM64, 2048MB, 5min timeout)

Deploy with:
```bash
cd ~/Desktop/personal-website/aws/cdk
cdk deploy
```

## Data

| File | Location | Description |
|------|----------|-------------|
| `rated_movies.parquet` | `data/processed/` | User's rated movies + TMDB metadata |
| `candidates.parquet` | `data/processed/` | Candidate pool (~1,600 movies) |
| `tmdb_cache.json` | `data/raw/` | Cached TMDB API responses |
| `ml-25m/` | `data/raw/` | MovieLens 25M dataset (~250MB) |
| `letterboxd-*/` | `data/raw/` | Letterboxd CSV export |
| `models/knn_cf/` | `models/` | KNN model artifacts |

All data files are gitignored.
