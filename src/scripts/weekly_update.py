"""Periodic pipeline: fetch new watches via RSS, merge into catalog, regenerate recs."""

import json
import os

import boto3
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, TMDB_CACHE_PATH
from src.data.letterboxd import fetch_rss, resolve_letterboxd_urls, fetch_letterboxd_ratings
from src.data.tmdb import enrich_all
from src.data.tmdb import fetch_popular_movies, fetch_watch_providers
from src.data.history import detect_feedback, save_run, save_feedback, feedback_metrics, load_all_history
from src.features.vectorize import MovieVectorizer
from src.features.profile import profile_summary
from src.features.feedback import compute_feedback_adjustment
from src.model.recommend import generate_recommendations, generate_genre_picks, get_top_candidate_ids, generate_streaming_picks
from src.deploy.dynamo_writer import write_recommendations, write_genre_picks, write_streaming_picks, write_profile


def main():
    output_path = PROCESSED_DIR / "rated_movies.parquet"

    # Load existing catalog (built from CSV export or previous runs)
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        previous_rated_ids = set(existing_df["tmdb_id"].values)
        existing_slugs = set(existing_df["filmSlug"].values) if "filmSlug" in existing_df.columns else set()
        print(f"Existing catalog: {len(existing_df)} rated movies")
    else:
        existing_df = pd.DataFrame()
        previous_rated_ids = set()
        existing_slugs = set()
        print("No existing catalog — will build from RSS.")

    # Fetch latest from RSS
    print("\nFetching latest Letterboxd ratings from RSS...")
    rss = fetch_rss()
    rss_df = pd.DataFrame(rss)
    rss_df = rss_df[rss_df["memberRating"] > 0].copy()
    if "filmSlug" not in rss_df.columns:
        rss_df["filmSlug"] = rss_df["filmTitle"].str.lower().str.replace(r"[^a-z0-9]+", "-", regex=True)
    print(f"Found {len(rss_df)} rated films in RSS feed")

    # Find new movies not already in the catalog
    new_movies = rss_df[~rss_df["filmSlug"].isin(existing_slugs)].copy()
    print(f"New films to add: {len(new_movies)}")

    # Check for re-ratings (same slug, different rating)
    if not existing_df.empty and "filmSlug" in existing_df.columns:
        overlap = rss_df[rss_df["filmSlug"].isin(existing_slugs)]
        updated = 0
        for _, rss_row in overlap.iterrows():
            slug = rss_row["filmSlug"]
            mask = existing_df["filmSlug"] == slug
            if mask.any():
                old_rating = existing_df.loc[mask, "memberRating"].iloc[0]
                if abs(old_rating - rss_row["memberRating"]) > 0.01:
                    existing_df.loc[mask, "memberRating"] = rss_row["memberRating"]
                    updated += 1
        if updated > 0:
            print(f"Updated {updated} re-rated films")

    # Enrich only the new movies with TMDB metadata
    if not new_movies.empty:
        print(f"\nEnriching {len(new_movies)} new movies with TMDB metadata...")
        new_enriched = enrich_all(new_movies)
        new_enriched_df = pd.DataFrame(new_enriched)

        if not new_enriched_df.empty:
            # Merge into existing catalog
            enriched_df = pd.concat([existing_df, new_enriched_df], ignore_index=True)
            # Deduplicate by tmdb_id (keep latest in case of re-enrichment)
            enriched_df = enriched_df.drop_duplicates(subset="tmdb_id", keep="last")
        else:
            enriched_df = existing_df
    else:
        print("\nNo new movies to enrich.")
        enriched_df = existing_df

    if enriched_df.empty:
        print("Error: No rated movies in catalog. Run build_catalog.py first.")
        return

    # Save updated catalog
    enriched_df.to_parquet(output_path, index=False)
    print(f"Updated catalog: {len(enriched_df)} movies")

    # Detect feedback from past recommendations
    print("\n--- Feedback Loop ---")
    history = load_all_history()
    if history.empty:
        print("No history from past recommendations.")
        feedback_adj = None
    else:
        fb = detect_feedback(enriched_df, previous_rated_ids)
        if fb.empty:
            print("No new feedback detected.")
            feedback_adj = None
        else:
            save_feedback(fb)
            hits = (fb["outcome"] == "hit").sum()
            misses = (fb["outcome"] == "miss").sum()
            neutral = (fb["outcome"] == "neutral").sum()
            print(f"Feedback: {hits} hits, {misses} misses, {neutral} neutral")

            vectorizer_tmp = MovieVectorizer()
            vectorizer_tmp.fit(enriched_df)
            rated_vectors = vectorizer_tmp.transform(enriched_df)
            feedback_adj = compute_feedback_adjustment(fb, rated_vectors, enriched_df)
            if feedback_adj is not None:
                print("Feedback adjustment computed and will be applied.")

        metrics = feedback_metrics()
        if metrics:
            print(f"Overall: watch_rate={metrics['watch_rate']:.1%}, "
                  f"avg_rating={metrics['avg_rating']:.2f}, "
                  f"hit_rate={metrics['hit_rate']:.1%}, "
                  f"miss_rate={metrics['miss_rate']:.1%}")

    # Generate recommendations
    print("\nFitting vectorizer...")
    vectorizer = MovieVectorizer()
    vectorizer.fit(enriched_df)

    print("Fetching candidate movies...")
    candidates = fetch_popular_movies(pages=50)
    candidates_df = pd.DataFrame(candidates)

    # Fetch Letterboxd community ratings BEFORE generating recs (used for quality scoring)
    print("\n--- Fetching Letterboxd Community Ratings ---")
    print("Resolving Letterboxd URLs for candidates...")
    url_map = resolve_letterboxd_urls(candidates_df)
    print("Fetching community ratings...")
    rating_map = fetch_letterboxd_ratings(url_map)
    candidates_df["letterboxd_rating"] = candidates_df["tmdb_id"].apply(
        lambda tid: rating_map.get(str(tid), None)
    )
    rated_count = candidates_df["letterboxd_rating"].notna().sum()
    print(f"  {rated_count}/{len(candidates_df)} candidates have Letterboxd ratings")

    # Train KNN CF model for blending
    cf_model = None
    try:
        from src.data.movielens import download_movielens, load_ratings, build_id_mappings, build_user_item_matrix
        from src.model.knn_cf import KNNCFModel

        print("\n--- Training KNN CF Model ---")
        download_movielens()
        ml_ratings = load_ratings()
        id_mappings = build_id_mappings(candidates_df)

        valid_ml_ids = set(id_mappings["tmdb_to_ml"].values())
        matrix_data = build_user_item_matrix(ml_ratings, valid_ml_ids)

        cf_model = KNNCFModel(k=50)
        cf_model.fit(
            matrix_data["matrix"],
            matrix_data["movie_id_to_idx"],
            matrix_data["movie_idx_to_id"],
            id_mappings["ml_to_tmdb"],
            id_mappings["tmdb_to_ml"],
        )
        print("  KNN CF model ready for blending")
    except Exception as e:
        print(f"  KNN CF training failed (non-fatal): {e}")
        cf_model = None

    print("\nGenerating recommendations...")
    recs = generate_recommendations(enriched_df, candidates_df, vectorizer, feedback_adjustment=feedback_adj, cf_model=cf_model)
    print(f"Generated {len(recs)} recommendations")

    # Generate genre picks
    print("Generating genre picks...")
    main_rec_ids = set(recs["tmdb_id"].values)
    genre_picks = generate_genre_picks(enriched_df, candidates_df, vectorizer, exclude_ids=main_rec_ids, feedback_adjustment=feedback_adj, cf_model=cf_model)
    print(f"Generated {len(genre_picks)} genre picks")

    # Generate streaming picks
    print("Generating streaming picks...")
    shortlist_ids = get_top_candidate_ids(enriched_df, candidates_df, vectorizer, 150, feedback_adj, cf_model=cf_model)
    streaming_map = fetch_watch_providers(shortlist_ids)
    streaming_picks = generate_streaming_picks(enriched_df, candidates_df, vectorizer, streaming_map, feedback_adjustment=feedback_adj, cf_model=cf_model)
    print(f"Generated {len(streaming_picks)} streaming picks")

    # Write to DynamoDB
    print("\nWriting to DynamoDB...")
    write_recommendations(recs)
    write_genre_picks(genre_picks)
    if not streaming_picks.empty:
        write_streaming_picks(streaming_picks)

    summary = profile_summary(enriched_df)
    write_profile(summary)

    # Save recommendation history
    print("\n--- Saving History ---")
    save_run(recs, genre_picks, streaming_picks if not streaming_picks.empty else None)

    # Upload cache to S3 for guest recommendations Lambda
    guest_bucket = os.environ.get("GUEST_CACHE_BUCKET")
    if guest_bucket:
        print("\n--- Uploading Guest Recs Cache to S3 ---")
        s3 = boto3.client("s3")

        # Upload TMDB cache
        if TMDB_CACHE_PATH.exists():
            s3.upload_file(str(TMDB_CACHE_PATH), guest_bucket, "tmdb_cache.json")
            cache_size = TMDB_CACHE_PATH.stat().st_size / (1024 * 1024)
            print(f"  Uploaded tmdb_cache.json ({cache_size:.1f} MB)")

        # Upload candidates as parquet (now includes letterboxd_rating)
        candidates_path = PROCESSED_DIR / "candidates.parquet"
        candidates_df.to_parquet(candidates_path, index=False)
        s3.upload_file(str(candidates_path), guest_bucket, "candidates.parquet")
        print(f"  Uploaded candidates.parquet ({len(candidates_df)} movies)")

        # Upload CF model artifacts to S3 (already trained earlier)
        if cf_model is not None:
            from src.config import MODELS_DIR
            knn_dir = MODELS_DIR / "knn_cf"
            cf_model.save_artifacts(knn_dir)

            for artifact_file in knn_dir.iterdir():
                s3_key = f"models/knn_cf/{artifact_file.name}"
                s3.upload_file(str(artifact_file), guest_bucket, s3_key)
                size_mb = artifact_file.stat().st_size / (1024 * 1024)
                print(f"  Uploaded {s3_key} ({size_mb:.1f} MB)")
    else:
        print("\nSkipping guest cache upload (GUEST_CACHE_BUCKET not set)")

    print("\nUpdate complete!")


if __name__ == "__main__":
    main()
