"""TMDB API client for movie metadata enrichment."""

from __future__ import annotations

import json
import time

import requests

from src.config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_CACHE_PATH, TMDB_IMAGE_BASE


def _load_cache() -> dict:
    """Load TMDB response cache from disk."""
    if TMDB_CACHE_PATH.exists():
        with open(TMDB_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    """Save TMDB response cache to disk."""
    with open(TMDB_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def _tmdb_get(endpoint: str, params: dict | None = None) -> dict:
    """Make a GET request to TMDB API."""
    url = f"{TMDB_BASE_URL}{endpoint}"
    default_params = {"api_key": TMDB_API_KEY}
    if params:
        default_params.update(params)
    resp = requests.get(url, params=default_params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# TMDB Watch Provider IDs (powered by JustWatch data)
STREAMING_PROVIDERS: dict[int, str] = {
    8: "Netflix",
    9: "Amazon Prime Video",
    119: "Amazon Prime Video",
    1899: "Max",
    384: "Max",
}


def fetch_watch_providers(tmdb_ids: list[int], country: str = "US") -> dict[int, list[str]]:
    """Look up which streaming services carry each movie.

    Calls GET /movie/{id}/watch/providers for each ID.
    Returns dict mapping tmdb_id -> list of provider names (e.g. ["Netflix", "Max"]).
    Only includes movies available via flatrate (subscription) streaming.
    """
    result: dict[int, list[str]] = {}
    total = len(tmdb_ids)

    for i, tmdb_id in enumerate(tmdb_ids):
        try:
            data = _tmdb_get(f"/movie/{tmdb_id}/watch/providers")
            country_data = data.get("results", {}).get(country, {})
            flatrate = country_data.get("flatrate", [])

            providers: list[str] = []
            seen_names: set[str] = set()
            for entry in flatrate:
                provider_id = entry.get("provider_id")
                if provider_id in STREAMING_PROVIDERS:
                    name = STREAMING_PROVIDERS[provider_id]
                    if name not in seen_names:
                        providers.append(name)
                        seen_names.add(name)

            if providers:
                result[tmdb_id] = providers
        except requests.HTTPError:
            pass

        if (i + 1) % 50 == 0:
            print(f"  Watch providers: checked {i + 1}/{total}...")

        time.sleep(0.05)

    print(f"Found {len(result)}/{total} movies on streaming services")
    return result


def search_movie(title: str, year: int | None = None) -> dict | None:
    """Search TMDB for a movie by title and optional year.

    Returns the best match result dict or None.
    """
    params = {"query": title}
    if year:
        params["year"] = str(year)

    data = _tmdb_get("/search/movie", params)
    results = data.get("results", [])

    if not results and year:
        # Retry without year filter
        data = _tmdb_get("/search/movie", {"query": title})
        results = data.get("results", [])

    return results[0] if results else None


def get_movie_details(tmdb_id: int) -> dict:
    """Fetch full movie details from TMDB."""
    return _tmdb_get(f"/movie/{tmdb_id}", {"append_to_response": "credits,keywords"})


def enrich_movie(title: str, year: int | None = None, cache: dict | None = None) -> dict | None:
    """Search and enrich a single movie with TMDB metadata.

    Returns enriched dict with genres, cast, director, keywords, etc.
    """
    cache_key = f"{title}|{year}"

    if cache is not None and cache_key in cache:
        return cache[cache_key]

    search_result = search_movie(title, year)
    if not search_result:
        print(f"  TMDB: No match for '{title}' ({year})")
        if cache is not None:
            cache[cache_key] = None
        return None

    tmdb_id = search_result["id"]
    details = get_movie_details(tmdb_id)

    # Extract director(s) from credits
    directors = []
    for crew in details.get("credits", {}).get("crew", []):
        if crew.get("job") == "Director":
            directors.append(crew["name"])

    # Extract top 5 billed cast
    cast_list = details.get("credits", {}).get("cast", [])[:5]
    cast = [c["name"] for c in cast_list]

    # Extract keywords
    keywords = [kw["name"] for kw in details.get("keywords", {}).get("keywords", [])]

    # Extract genres
    genres = [g["name"] for g in details.get("genres", [])]

    poster_path = details.get("poster_path", "")
    poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else ""

    enriched = {
        "tmdb_id": tmdb_id,
        "title": details.get("title", title),
        "year": int(details.get("release_date", "0000")[:4]) if details.get("release_date") else year,
        "genres": genres,
        "director": directors,
        "cast": cast,
        "keywords": keywords,
        "runtime": details.get("runtime", 0),
        "vote_average": details.get("vote_average", 0.0),
        "vote_count": details.get("vote_count", 0),
        "original_language": details.get("original_language", "en"),
        "overview": details.get("overview", ""),
        "poster_url": poster_url,
        "tmdb_url": f"https://www.themoviedb.org/movie/{tmdb_id}",
    }

    if cache is not None:
        cache[cache_key] = enriched

    return enriched


def enrich_all(films_df, title_col: str = "filmTitle", year_col: str = "filmYear") -> list[dict]:
    """Enrich a DataFrame of films with TMDB metadata.

    Args:
        films_df: DataFrame with at minimum a title column.
        title_col: Column name for film title.
        year_col: Column name for film year (optional in df).

    Returns:
        List of enriched movie dicts.
    """
    cache = _load_cache()
    enriched = []
    total = len(films_df)

    for i, (_, row) in enumerate(films_df.iterrows()):
        title = row[title_col]
        year = int(row[year_col]) if year_col in row.index and pd.notna(row.get(year_col)) else None

        if (i + 1) % 50 == 0:
            print(f"  Enriching {i + 1}/{total}...")
            _save_cache(cache)

        result = enrich_movie(title, year, cache=cache)
        if result:
            # Carry over Letterboxd-specific fields
            result["memberRating"] = row.get("memberRating", 0.0)
            result["filmSlug"] = row.get("filmSlug", "")
            enriched.append(result)

        time.sleep(0.05)  # ~20 req/sec, well under TMDB's 40/sec limit

    _save_cache(cache)
    print(f"Enriched {len(enriched)}/{total} films ({len(enriched)/total*100:.1f}% match rate)")
    return enriched


def fetch_popular_movies(pages: int = 50) -> list[dict]:
    """Fetch popular/top-rated movies from TMDB as recommendation candidates.

    Fetches from both popular and top_rated endpoints for diversity.

    Returns list of enriched movie dicts.
    """
    cache = _load_cache()
    candidates = {}

    for endpoint in ["/movie/popular", "/movie/top_rated"]:
        endpoint_name = endpoint.split("/")[-1]
        print(f"  Fetching from {endpoint_name}...")
        for page in range(1, pages + 1):
            try:
                data = _tmdb_get(endpoint, {"page": str(page)})
            except requests.HTTPError:
                break

            page_new = 0
            sample_title = None
            for movie in data.get("results", []):
                tmdb_id = movie["id"]
                if tmdb_id in candidates:
                    continue

                cache_key = f"{movie['title']}|{movie.get('release_date', '')[:4]}"
                if cache_key in cache and cache[cache_key] is not None:
                    cached_entry = cache[cache_key]
                    # Backfill vote_count from list API if missing from cache
                    if "vote_count" not in cached_entry:
                        cached_entry["vote_count"] = movie.get("vote_count", 0)
                        cache[cache_key] = cached_entry
                    candidates[tmdb_id] = cached_entry
                    page_new += 1
                    if sample_title is None:
                        sample_title = movie["title"]
                    continue

                try:
                    details = get_movie_details(tmdb_id)
                except requests.HTTPError:
                    continue

                directors = [
                    c["name"]
                    for c in details.get("credits", {}).get("crew", [])
                    if c.get("job") == "Director"
                ]
                cast_list = details.get("credits", {}).get("cast", [])[:5]
                cast = [c["name"] for c in cast_list]
                keywords = [kw["name"] for kw in details.get("keywords", {}).get("keywords", [])]
                genres = [g["name"] for g in details.get("genres", [])]
                poster_path = details.get("poster_path", "")

                enriched = {
                    "tmdb_id": tmdb_id,
                    "title": details.get("title", movie["title"]),
                    "year": int(details.get("release_date", "0000")[:4]) if details.get("release_date") else None,
                    "genres": genres,
                    "director": directors,
                    "cast": cast,
                    "keywords": keywords,
                    "runtime": details.get("runtime", 0),
                    "vote_average": details.get("vote_average", 0.0),
                    "vote_count": details.get("vote_count", 0),
                    "original_language": details.get("original_language", "en"),
                    "overview": details.get("overview", ""),
                    "poster_url": f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else "",
                    "tmdb_url": f"https://www.themoviedb.org/movie/{tmdb_id}",
                }
                candidates[tmdb_id] = enriched
                cache[cache_key] = enriched
                page_new += 1
                if sample_title is None:
                    sample_title = movie["title"]
                time.sleep(0.05)

            print(f"    {endpoint_name} p{page}: +{page_new} movies (total: {len(candidates)}) — e.g. {sample_title}")

            if (page % 10) == 0:
                _save_cache(cache)

    _save_cache(cache)
    print(f"Fetched {len(candidates)} candidate movies from TMDB")
    return list(candidates.values())


# Avoid circular import — only needed in enrich_all
import pandas as pd  # noqa: E402
