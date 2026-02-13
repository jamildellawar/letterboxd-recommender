"""Feedback adjustment vector for the taste profile."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_feedback_adjustment(
    feedback_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    rated_df: pd.DataFrame,
) -> np.ndarray | None:
    """Compute a feedback adjustment vector from hit/miss signals.

    For hits (rated >= 4.0): reinforce those feature dimensions.
    For misses (rated <= 2.5): penalize those feature dimensions.
    Neutral outcomes are ignored.

    Args:
        feedback_df: DataFrame with tmdb_id, user_rating, outcome columns.
        feature_matrix: Feature vectors for all rated movies (from vectorizer.transform).
        rated_df: Full rated movies DataFrame (aligned with feature_matrix rows).

    Returns:
        L2-normalized adjustment vector, or None if no actionable feedback.
    """
    if feedback_df.empty:
        return None

    hits = feedback_df[feedback_df["outcome"] == "hit"]
    misses = feedback_df[feedback_df["outcome"] == "miss"]

    if hits.empty and misses.empty:
        return None

    n_features = feature_matrix.shape[1]
    adjustment = np.zeros(n_features)

    rated_ids = rated_df["tmdb_id"].values

    # Reinforce hit features
    for _, row in hits.iterrows():
        idx = np.where(rated_ids == row["tmdb_id"])[0]
        if len(idx) > 0:
            weight = row["user_rating"] - 3.0  # 4.0→1.0, 5.0→2.0
            adjustment += feature_matrix[idx[0]] * weight

    # Penalize miss features
    for _, row in misses.iterrows():
        idx = np.where(rated_ids == row["tmdb_id"])[0]
        if len(idx) > 0:
            weight = 3.0 - row["user_rating"]  # 2.5→0.5, 1.0→2.0
            adjustment -= feature_matrix[idx[0]] * weight

    norm = np.linalg.norm(adjustment)
    if norm > 0:
        adjustment = adjustment / norm
        return adjustment

    return None


def apply_feedback_to_profile(
    base_profile: np.ndarray,
    adjustment: np.ndarray | None,
    strength: float = 0.1,
) -> np.ndarray:
    """Blend feedback adjustment into the base taste profile.

    Args:
        base_profile: The original taste profile vector.
        adjustment: Feedback adjustment vector (or None for no-op).
        strength: Blend weight for feedback (default 10%).

    Returns:
        Updated profile, L2-normalized.
    """
    if adjustment is None:
        return base_profile

    blended = (1 - strength) * base_profile + strength * adjustment

    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm

    return blended
