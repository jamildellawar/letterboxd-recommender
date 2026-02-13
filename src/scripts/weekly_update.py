"""Periodic pipeline: fetch new watches via RSS, merge into catalog, regenerate recs."""

import pandas as pd

from src.config import PROCESSED_DIR
from src.data.letterboxd import fetch_rss
from src.data.tmdb import enrich_all
from src.data.tmdb import fetch_popular_movies
from src.data.history import detect_feedback, save_run, save_feedback, feedback_metrics, load_all_history
from src.features.vectorize import MovieVectorizer
from src.features.profile import profile_summary
from src.features.feedback import compute_feedback_adjustment
from src.model.recommend import generate_recommendations, generate_genre_picks
from src.deploy.dynamo_writer import write_recommendations, write_genre_picks, write_profile


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

    print("\nUpdate complete!")


if __name__ == "__main__":
    main()
