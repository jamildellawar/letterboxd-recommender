"""Write recommendations and profile data to DynamoDB."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import boto3
import pandas as pd

from src.config import AWS_REGION, LETTERBOXD_RECS_TABLE, LETTERBOXD_PROFILE_TABLE


def get_dynamodb_resource():
    """Get a DynamoDB resource."""
    return boto3.resource("dynamodb", region_name=AWS_REGION)


def _convert_floats(obj):
    """Convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, dict):
        return {k: _convert_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_floats(i) for i in obj]
    return obj


def write_recommendations(recs_df: pd.DataFrame) -> int:
    """Write top recommendations to DynamoDB LetterboxdRecs table.

    Table schema:
        PK: pk ("RECS")
        SK: tmdb_id (string)

    Args:
        recs_df: DataFrame from generate_recommendations() with
            tmdb_id, title, year, similarity_score, genres, director,
            cast, poster_url, tmdb_url, explanation.

    Returns:
        Number of items written.
    """
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(LETTERBOXD_RECS_TABLE)
    now = datetime.now(timezone.utc).isoformat()

    # Clear existing recommendations first
    _clear_table(table, "RECS")

    count = 0
    with table.batch_writer() as batch:
        for _, row in recs_df.iterrows():
            item = {
                "pk": "RECS",
                "tmdb_id": str(row["tmdb_id"]),
                "title": row["title"],
                "year": int(row["year"]) if pd.notna(row.get("year")) else 0,
                "similarity_score": _convert_floats(float(row["similarity_score"])),
                "genres": row.get("genres", []),
                "director": row.get("director", []),
                "cast_preview": (row.get("cast") or [])[:3],
                "poster_url": row.get("poster_url", ""),
                "tmdb_url": row.get("tmdb_url", ""),
                "letterboxd_url": row.get("letterboxd_url", ""),
                "explanation": row.get("explanation", []),
                "generated_at": now,
            }
            batch.put_item(Item=item)
            count += 1

    print(f"Wrote {count} recommendations to {LETTERBOXD_RECS_TABLE}")
    return count


def write_genre_picks(picks_df: pd.DataFrame) -> int:
    """Write genre-based picks to DynamoDB LetterboxdRecs table.

    Uses pk="GENRE_PICKS" to distinguish from main recs.
    """
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(LETTERBOXD_RECS_TABLE)
    now = datetime.now(timezone.utc).isoformat()

    _clear_table(table, "GENRE_PICKS")

    count = 0
    with table.batch_writer() as batch:
        for _, row in picks_df.iterrows():
            item = {
                "pk": "GENRE_PICKS",
                "tmdb_id": str(row["tmdb_id"]),
                "title": row["title"],
                "year": int(row["year"]) if pd.notna(row.get("year")) else 0,
                "similarity_score": _convert_floats(float(row["similarity_score"])),
                "genres": row.get("genres", []),
                "director": row.get("director", []),
                "cast_preview": (row.get("cast") or [])[:3],
                "poster_url": row.get("poster_url", ""),
                "tmdb_url": row.get("tmdb_url", ""),
                "letterboxd_url": row.get("letterboxd_url", ""),
                "explanation": row.get("explanation", []),
                "picked_genre": row.get("picked_genre", ""),
                "generated_at": now,
            }
            batch.put_item(Item=item)
            count += 1

    print(f"Wrote {count} genre picks to {LETTERBOXD_RECS_TABLE}")
    return count


def write_streaming_picks(picks_df: pd.DataFrame) -> int:
    """Write streaming picks to DynamoDB LetterboxdRecs table.

    Uses pk="STREAMING" to distinguish from main recs and genre picks.
    """
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(LETTERBOXD_RECS_TABLE)
    now = datetime.now(timezone.utc).isoformat()

    _clear_table(table, "STREAMING")

    count = 0
    with table.batch_writer() as batch:
        for _, row in picks_df.iterrows():
            item = {
                "pk": "STREAMING",
                "tmdb_id": str(row["tmdb_id"]),
                "title": row["title"],
                "year": int(row["year"]) if pd.notna(row.get("year")) else 0,
                "similarity_score": _convert_floats(float(row["similarity_score"])),
                "genres": row.get("genres", []),
                "director": row.get("director", []),
                "cast_preview": (row.get("cast") or [])[:3],
                "poster_url": row.get("poster_url", ""),
                "tmdb_url": row.get("tmdb_url", ""),
                "letterboxd_url": row.get("letterboxd_url", ""),
                "explanation": row.get("explanation", []),
                "streaming_services": row.get("streaming_services", []),
                "generated_at": now,
            }
            batch.put_item(Item=item)
            count += 1

    print(f"Wrote {count} streaming picks to {LETTERBOXD_RECS_TABLE}")
    return count


def write_profile(profile_summary: dict) -> None:
    """Write user taste profile summary to DynamoDB LetterboxdProfile table.

    Table schema:
        PK: pk ("PROFILE")
        SK: "latest"
    """
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(LETTERBOXD_PROFILE_TABLE)
    now = datetime.now(timezone.utc).isoformat()

    item = {
        "pk": "PROFILE",
        "sk": "latest",
        "total_rated": profile_summary["total_rated"],
        "avg_rating": _convert_floats(profile_summary["avg_rating"]),
        "top_genres": profile_summary["top_genres"],
        "top_directors": profile_summary["top_directors"],
        "updated_at": now,
    }

    table.put_item(Item=item)
    print(f"Wrote profile summary to {LETTERBOXD_PROFILE_TABLE}")


def _clear_table(table, pk_value: str) -> None:
    """Delete all items with the given partition key."""
    response = table.query(
        KeyConditionExpression="pk = :pk",
        ExpressionAttributeValues={":pk": pk_value},
        ProjectionExpression="pk, tmdb_id",
    )

    with table.batch_writer() as batch:
        for item in response.get("Items", []):
            batch.delete_item(Key={"pk": item["pk"], "tmdb_id": item["tmdb_id"]})
