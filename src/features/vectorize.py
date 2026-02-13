"""Build feature vectors for movies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from src.config import TOP_N_DIRECTORS, MODELS_DIR


def _safe_list(val) -> list:
    """Convert a value to a Python list (handles numpy arrays, None, etc.)."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    try:
        return list(val)
    except TypeError:
        return []


class MovieVectorizer:
    """Builds combined feature vectors from TMDB metadata.

    Feature weights are tuned to avoid genre domination (Drama appearing
    in most films) and emphasize more discriminative signals like keywords,
    director, and cast overlap.
    """

    # Feature weights — tuned to balance discriminative power
    GENRE_WEIGHT = 1.0       # down from 2.0; IDF handles genre frequency
    DIRECTOR_WEIGHT = 2.0    # up from 1.5; directors are very taste-specific
    CAST_WEIGHT = 1.5        # up from 1.0; cast overlap is meaningful
    KEYWORD_WEIGHT = 2.0     # up from 1.0; keywords capture themes/tone
    PLOT_WEIGHT = 0.3        # down from 0.5; plot text is noisy
    META_WEIGHT = 0.5        # down from 1.0; decade/runtime are weak signals

    def __init__(self):
        self.genre_mlb = MultiLabelBinarizer()
        self.director_mlb = MultiLabelBinarizer()
        self.keyword_tfidf = TfidfVectorizer(
            max_features=150,  # smaller vocab = denser, more meaningful with 67 films
            token_pattern=r"\S+",
        )
        self.cast_tfidf = TfidfVectorizer(
            max_features=80,   # top 80 actors seen in rated films
            token_pattern=r"\S+",
        )
        self.plot_tfidf = TfidfVectorizer(max_features=100, stop_words="english")
        self.top_directors: list[str] = []
        self.genre_idf: np.ndarray | None = None  # IDF weights per genre
        self._mean_vector: np.ndarray | None = None  # for mean-centering
        self._fitted = False
        self._cast_fitted = False
        self._keyword_fitted = False

    def fit(self, df: pd.DataFrame) -> "MovieVectorizer":
        """Fit all sub-vectorizers on the rated movies dataset."""
        genres_col = df["genres"].apply(_safe_list)

        # Genre multi-hot + IDF
        self.genre_mlb.fit(genres_col)
        genre_matrix = self.genre_mlb.transform(genres_col)
        # IDF = log(N / doc_freq) — downweights ubiquitous genres like Drama
        n_docs = len(df)
        doc_freq = genre_matrix.sum(axis=0).astype(float)
        doc_freq = np.maximum(doc_freq, 1.0)  # avoid div by zero
        self.genre_idf = np.log(n_docs / doc_freq)

        # Director: top-N by frequency, rest bucketed as "other"
        all_directors = [d for directors in df["director"].apply(_safe_list) for d in directors]
        director_counts = pd.Series(all_directors).value_counts()
        self.top_directors = director_counts.head(TOP_N_DIRECTORS).index.tolist()
        director_labels = df["director"].apply(lambda x: self._bucket_directors(_safe_list(x)))
        self.director_mlb.fit(director_labels)

        # Cast TF-IDF
        cast_text = self._to_cast_text(df)
        try:
            self.cast_tfidf.fit(cast_text)
            self._cast_fitted = True
        except ValueError:
            self._cast_fitted = False

        # Keyword TF-IDF
        keyword_text = self._to_keyword_text(df)
        try:
            self.keyword_tfidf.fit(keyword_text)
            self._keyword_fitted = True
        except ValueError:
            self._keyword_fitted = False

        # Plot overview TF-IDF
        plot_text = df["overview"].fillna("")
        self.plot_tfidf.fit(plot_text)

        # Compute and store mean vector for centering
        # This removes the "average movie" component (dominated by Drama)
        # so the model focuses on what makes each movie distinctive
        raw_vectors = self._raw_transform(df)
        self._mean_vector = raw_vectors.mean(axis=0)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform movies into mean-centered feature vectors."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform().")
        raw = self._raw_transform(df)
        if self._mean_vector is not None:
            raw = raw - self._mean_vector
        return raw

    def _raw_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform movies into feature vectors (before centering)."""

        genres_col = df["genres"].apply(_safe_list)

        # Genre multi-hot * IDF weighting (downweights Drama, boosts rare genres)
        genre_vec = self.genre_mlb.transform(genres_col).astype(float)
        if self.genre_idf is not None:
            genre_vec = genre_vec * self.genre_idf

        # Director multi-hot (bucketed)
        director_labels = df["director"].apply(lambda x: self._bucket_directors(_safe_list(x)))
        director_vec = self.director_mlb.transform(director_labels)

        # Cast TF-IDF
        if self._cast_fitted:
            cast_text = self._to_cast_text(df)
            cast_vec = self.cast_tfidf.transform(cast_text).toarray()
        else:
            cast_vec = np.zeros((len(df), 0))

        # Keyword TF-IDF
        if self._keyword_fitted:
            keyword_text = self._to_keyword_text(df)
            keyword_vec = self.keyword_tfidf.transform(keyword_text).toarray()
        else:
            keyword_vec = np.zeros((len(df), 0))

        # Plot TF-IDF
        plot_text = df["overview"].fillna("")
        plot_vec = self.plot_tfidf.transform(plot_text).toarray()

        # Metadata features
        meta_vec = self._metadata_features(df)

        # Concatenate with tuned weights
        return np.hstack([
            genre_vec * self.GENRE_WEIGHT,
            director_vec * self.DIRECTOR_WEIGHT,
            cast_vec * self.CAST_WEIGHT,
            keyword_vec * self.KEYWORD_WEIGHT,
            plot_vec * self.PLOT_WEIGHT,
            meta_vec * self.META_WEIGHT,
        ])

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    @staticmethod
    def _to_cast_text(df: pd.DataFrame) -> pd.Series:
        """Convert cast lists to text, replacing spaces with underscores."""
        return df["cast"].apply(
            lambda x: " ".join(n.replace(" ", "_") for n in _safe_list(x))
        )

    @staticmethod
    def _to_keyword_text(df: pd.DataFrame) -> pd.Series:
        """Convert keyword lists to text, replacing spaces with underscores."""
        return df["keywords"].apply(
            lambda x: " ".join(k.replace(" ", "_") for k in _safe_list(x))
        )

    def _bucket_directors(self, directors: list[str]) -> list[str]:
        """Map directors to top-N or 'other' bucket."""
        if not directors:
            return ["other"]
        result = []
        for d in directors:
            if d in self.top_directors:
                result.append(d)
            else:
                result.append("other")
        return result if result else ["other"]

    @staticmethod
    def _metadata_features(df: pd.DataFrame) -> np.ndarray:
        """Build normalized metadata features."""
        features = []
        for _, row in df.iterrows():
            year = row.get("year", 2000) or 2000
            decade = (year - 1900) / 130.0

            runtime = row.get("runtime", 100) or 100
            runtime_norm = min(runtime / 240.0, 1.0)

            lang = row.get("original_language", "en")
            is_english = 1.0 if lang == "en" else 0.0

            features.append([decade, runtime_norm, is_english])

        return np.array(features)
