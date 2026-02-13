"""Parse Letterboxd RSS feed and scrape full watch history."""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import (
    LETTERBOXD_FILMS_URL,
    LETTERBOXD_RSS_URL,
    LETTERBOXD_USERNAME,
    RAW_DIR,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}


def fetch_rss() -> list[dict]:
    """Fetch recent watches from Letterboxd RSS feed.

    Returns list of dicts with filmTitle, filmYear, memberRating,
    watchedDate, rewatch, posterUrl, link.
    """
    resp = requests.get(LETTERBOXD_RSS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    ns = {"letterboxd": "https://letterboxd.com"}
    films = []

    for item in root.findall(".//item"):
        title_el = item.find("letterboxd:filmTitle", ns)
        year_el = item.find("letterboxd:filmYear", ns)
        rating_el = item.find("letterboxd:memberRating", ns)
        watched_el = item.find("letterboxd:watchedDate", ns)
        rewatch_el = item.find("letterboxd:rewatch", ns)
        link_el = item.find("link")

        if title_el is None or title_el.text is None:
            continue

        # Extract poster URL from description HTML
        desc_el = item.find("description")
        poster_url = ""
        if desc_el is not None and desc_el.text:
            img_match = re.search(r'<img\s+src="([^"]+)"', desc_el.text)
            if img_match:
                poster_url = img_match.group(1)

        films.append({
            "filmTitle": title_el.text,
            "filmYear": int(year_el.text) if year_el is not None and year_el.text else None,
            "memberRating": float(rating_el.text) if rating_el is not None and rating_el.text else 0.0,
            "watchedDate": watched_el.text if watched_el is not None else None,
            "rewatch": rewatch_el.text == "Yes" if rewatch_el is not None else False,
            "posterUrl": poster_url,
            "link": link_el.text if link_el is not None else "",
        })

    return films


def scrape_full_history() -> list[dict]:
    """Scrape complete Letterboxd watch history by paginating through films pages.

    Returns list of dicts with filmTitle, filmYear, filmSlug.
    """
    films = []
    page = 1

    while True:
        url = f"{LETTERBOXD_FILMS_URL}{page}/"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            break
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        posters = soup.select("li.poster-container")

        if not posters:
            break

        for poster in posters:
            film_div = poster.select_one("div.film-poster")
            if not film_div:
                continue

            slug = film_div.get("data-film-slug", "")
            name_el = poster.select_one("img.image")
            title = name_el.get("alt", "") if name_el else ""

            films.append({
                "filmTitle": title,
                "filmSlug": slug,
            })

        page += 1
        time.sleep(1)

    return films


def scrape_ratings() -> list[dict]:
    """Scrape rated films from Letterboxd with star ratings.

    Paginates through /films/ratings/ pages and extracts rating from
    the star overlay on each poster.
    """
    base_url = f"https://letterboxd.com/{LETTERBOXD_USERNAME}/films/ratings/page/"
    films = []
    page = 1

    while True:
        url = f"{base_url}{page}/"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code in (403, 404):
            break
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("li.poster-container")

        if not rows:
            break

        for row in rows:
            film_div = row.select_one("div.film-poster")
            if not film_div:
                continue

            slug = film_div.get("data-film-slug", "")
            name_el = row.select_one("img.image")
            title = name_el.get("alt", "") if name_el else ""

            # Rating is in a span with class rated-X where X is 1-10 (half stars)
            rating_span = row.select_one("span.rating")
            rating = 0.0
            if rating_span:
                classes = rating_span.get("class", [])
                for cls in classes:
                    match = re.match(r"rated-(\d+)", cls)
                    if match:
                        rating = int(match.group(1)) / 2.0
                        break

            if rating > 0:
                films.append({
                    "filmTitle": title,
                    "filmSlug": slug,
                    "memberRating": rating,
                })

        print(f"  Page {page}: {len(rows)} films found")
        page += 1
        time.sleep(1.5)

    return films


def load_csv_export() -> pd.DataFrame | None:
    """Try to load ratings from a Letterboxd CSV export.

    Looks for ratings.csv inside any letterboxd-* folder in data/raw/.
    Returns DataFrame or None if not found.
    """
    import glob as globmod
    pattern = str(RAW_DIR / "letterboxd-*/ratings.csv")
    matches = globmod.glob(pattern)
    if not matches:
        return None

    csv_path = matches[0]
    print(f"Loading Letterboxd CSV export from {csv_path}")
    df = pd.read_csv(csv_path)

    # Rename columns to match our schema: Date,Name,Year,Letterboxd URI,Rating
    df = df.rename(columns={
        "Name": "filmTitle",
        "Year": "filmYear",
        "Rating": "memberRating",
        "Letterboxd URI": "letterboxdUri",
        "Date": "ratedDate",
    })

    # Filter out unrated
    df = df[df["memberRating"] > 0].copy()

    # Generate slug from title
    df["filmSlug"] = df["filmTitle"].str.lower().str.replace(r"[^a-z0-9]+", "-", regex=True)

    print(f"Found {len(df)} rated films in CSV export")
    return df


def load_or_fetch_history() -> pd.DataFrame:
    """Load ratings from CSV export, cached parquet, or RSS feed (in that order).

    Returns DataFrame with filmTitle, filmYear, filmSlug, memberRating.
    """
    cache_path = RAW_DIR / "letterboxd_history.parquet"

    # Prefer CSV export (most complete)
    csv_df = load_csv_export()
    if csv_df is not None:
        csv_df.to_parquet(cache_path, index=False)
        print(f"Cached {len(csv_df)} rated films to {cache_path}")
        return csv_df

    # Fall back to cached parquet
    if cache_path.exists():
        print(f"Loading cached history from {cache_path}")
        return pd.read_parquet(cache_path)

    # Fall back to RSS
    print("Fetching rated films from Letterboxd RSS feed...")
    rss = fetch_rss()
    df = pd.DataFrame(rss)
    df = df[df["memberRating"] > 0].copy()

    if "filmSlug" not in df.columns:
        df["filmSlug"] = df["filmTitle"].str.lower().str.replace(r"[^a-z0-9]+", "-", regex=True)

    print(f"Found {len(df)} rated films in RSS feed")
    df.to_parquet(cache_path, index=False)
    print(f"Cached to {cache_path}")
    return df


def fetch_letterboxd_rating(url: str) -> float | None:
    """Fetch the Letterboxd community average rating from a film page.

    Parses the JSON-LD structured data for aggregateRating.ratingValue.
    Returns rating (0.0-5.0) or None if not found.
    """
    if "/search/" in url:
        return None  # search fallback URL, can't scrape

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
    except requests.RequestException:
        return None

    import json as jsonmod
    soup = BeautifulSoup(resp.text, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            text = (script.string or script.get_text()).strip()
            # Letterboxd wraps JSON-LD in CDATA comments — strip them
            text = re.sub(r"/\*\s*<!\[CDATA\[\s*\*/", "", text)
            text = re.sub(r"/\*\s*\]\]>\s*\*/", "", text)
            text = text.strip()
            data = jsonmod.loads(text)
            agg = data.get("aggregateRating")
            if agg and "ratingValue" in agg:
                return float(agg["ratingValue"])
        except (jsonmod.JSONDecodeError, TypeError, ValueError):
            continue

    return None


def fetch_letterboxd_ratings(url_map: dict[str, str]) -> dict[str, float]:
    """Fetch Letterboxd ratings for a batch of movies.

    Args:
        url_map: Dict of tmdb_id -> letterboxd_url.

    Returns:
        Dict of tmdb_id -> letterboxd_rating (0.0-5.0).
    """
    import json as jsonmod

    cache_path = RAW_DIR / "letterboxd_ratings.json"
    rating_cache: dict[str, float] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            rating_cache = jsonmod.load(f)

    fetched = 0
    for tmdb_id, url in url_map.items():
        if tmdb_id in rating_cache:
            continue

        rating = fetch_letterboxd_rating(url)
        if rating is not None:
            rating_cache[tmdb_id] = rating
            fetched += 1

            if fetched % 10 == 0:
                print(f"  Fetched {fetched} Letterboxd ratings...")

        time.sleep(1.0)  # polite rate limiting

    with open(cache_path, "w") as f:
        jsonmod.dump(rating_cache, f, indent=2)

    print(f"Fetched {fetched} new Letterboxd ratings ({len(rating_cache)} cached total)")
    return rating_cache


def _title_to_slug(title: str) -> str:
    """Convert a movie title to a Letterboxd-style slug."""
    slug = title.lower()
    # Remove apostrophes/quotes
    slug = re.sub(r"[''']", "", slug)
    # Replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    return slug


def resolve_letterboxd_url(title: str, year: int | None = None) -> str:
    """Resolve the correct Letterboxd film URL.

    Tries slug-year first (to avoid hitting an older film with the same title),
    then plain slug, then falls back to search URL.
    """
    slug = _title_to_slug(title)

    # Try slug-year first (avoids collisions like "the-color-purple" vs "the-color-purple-2023")
    if year:
        url_year = f"https://letterboxd.com/film/{slug}-{year}/"
        try:
            resp = requests.head(url_year, headers=HEADERS, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                return url_year
        except requests.RequestException:
            pass

    # Try plain slug
    url = f"https://letterboxd.com/film/{slug}/"
    try:
        resp = requests.head(url, headers=HEADERS, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            return url
    except requests.RequestException:
        pass

    # Fallback to search
    return f"https://letterboxd.com/search/{slug}/"


def resolve_letterboxd_urls(df) -> dict[str, str]:
    """Resolve Letterboxd URLs for a DataFrame of movies.

    Args:
        df: DataFrame with 'title' and 'year' columns.

    Returns:
        Dict mapping tmdb_id (str) -> letterboxd URL.
    """
    # Load cached URLs
    cache_path = RAW_DIR / "letterboxd_urls.json"
    url_cache: dict[str, str] = {}
    if cache_path.exists():
        import json
        with open(cache_path) as f:
            url_cache = json.load(f)

    total = len(df)
    resolved = 0
    for i, (_, row) in enumerate(df.iterrows()):
        tmdb_id = str(row.get("tmdb_id", ""))
        if tmdb_id in url_cache:
            continue

        title = row.get("title", "")
        year = int(row["year"]) if row.get("year") else None
        url = resolve_letterboxd_url(title, year)
        url_cache[tmdb_id] = url
        resolved += 1

        if (resolved % 10) == 0:
            print(f"  Resolved {resolved} URLs... (e.g. {title} -> {url})")

        time.sleep(0.5)  # polite rate limiting

    # Save cache
    import json
    with open(cache_path, "w") as f:
        json.dump(url_cache, f, indent=2)

    print(f"Resolved {resolved} new Letterboxd URLs ({total} total)")
    return url_cache
