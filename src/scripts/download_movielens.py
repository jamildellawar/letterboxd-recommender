"""One-time script to download and verify MovieLens 25M data."""

from src.data.movielens import download_movielens, load_ratings, load_links


def main():
    print("=== MovieLens 25M Download ===\n")

    download_movielens()

    print("\nVerifying data...")
    ratings = load_ratings()
    links = load_links()

    print(f"  Ratings: {len(ratings):,} rows")
    print(f"  Unique users: {ratings['userId'].nunique():,}")
    print(f"  Unique movies: {ratings['movieId'].nunique():,}")
    print(f"  Links with tmdbId: {len(links):,}")

    print("\nMovieLens data ready!")


if __name__ == "__main__":
    main()
