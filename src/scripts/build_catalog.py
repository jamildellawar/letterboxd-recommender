"""One-time script: enrich all rated movies with TMDB metadata."""

from src.data.merge import build_rated_movies


def main():
    df = build_rated_movies()
    print(f"\nCatalog built: {len(df)} movies")
    print(f"Columns: {list(df.columns)}")
    if not df.empty:
        print(f"Rating range: {df['memberRating'].min():.1f} - {df['memberRating'].max():.1f}")
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")


if __name__ == "__main__":
    main()
