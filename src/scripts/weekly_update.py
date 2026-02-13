"""Weekly pipeline: check for new watches, refresh catalog, regenerate recs."""

import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR
from src.data.letterboxd import fetch_rss
from src.data.tmdb import enrich_all, fetch_popular_movies
from src.data.history import detect_feedback, save_run, save_feedback, feedback_metrics, load_all_history
from src.features.vectorize import MovieVectorizer
from src.features.profile import profile_summary
from src.features.feedback import compute_feedback_adjustment
from src.model.recommend import generate_recommendations, generate_genre_picks
from src.deploy.dynamo_writer import write_recommendations, write_genre_picks, write_profile


def main():
    cache_path = RAW_DIR / "letterboxd_history.parquet"
    output_path = PROCESSED_DIR / "rated_movies.parquet"

    # Capture previous rated IDs before rebuilding catalog
    previous_rated_ids: set = set()
    if output_path.exists():
        old_rated = pd.read_parquet(output_path)
        if "tmdb_id" in old_rated.columns:
            previous_rated_ids = set(old_rated["tmdb_id"].values)
        print(f"Previous catalog: {len(previous_rated_ids)} rated movies")

    # Fetch latest from RSS
    print("Fetching latest Letterboxd ratings from RSS...")
    rss = fetch_rss()
    new_df = pd.DataFrame(rss)
    new_df = new_df[new_df["memberRating"] > 0].copy()
    if "filmSlug" not in new_df.columns:
        new_df["filmSlug"] = new_df["filmTitle"].str.lower().str.replace(r"[^a-z0-9]+", "-", regex=True)
    print(f"Found {len(new_df)} rated films in RSS feed")

    # Compare with cached version
    if cache_path.exists():
        old_df = pd.read_parquet(cache_path)
        old_slugs = set(old_df["filmSlug"].values)
        new_slugs = set(new_df["filmSlug"].values)
        added = new_slugs - old_slugs
        print(f"New films since last run: {len(added)}")
    else:
        print("No cached history — building from scratch.")

    # Update cache
    new_df.to_parquet(cache_path, index=False)

    # Enrich with TMDB
    print("\nEnriching with TMDB metadata...")
    enriched = enrich_all(new_df)
    enriched_df = pd.DataFrame(enriched)
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

    print("Generating recommendations...")
    recs = generate_recommendations(enriched_df, candidates_df, vectorizer, feedback_adjustment=feedback_adj)
    print(f"Generated {len(recs)} recommendations")

    # Generate genre picks
    print("Generating genre picks...")
    main_rec_ids = set(recs["tmdb_id"].values)
    genre_picks = generate_genre_picks(enriched_df, candidates_df, vectorizer, exclude_ids=main_rec_ids, feedback_adjustment=feedback_adj)
    print(f"Generated {len(genre_picks)} genre picks")

    # Write to DynamoDB
    print("\nWriting to DynamoDB...")
    write_recommendations(recs)
    write_genre_picks(genre_picks)

    summary = profile_summary(enriched_df)
    write_profile(summary)

    # Save recommendation history
    print("\n--- Saving History ---")
    save_run(recs, genre_picks)

    print("\nWeekly update complete!")


if __name__ == "__main__":
    main()
