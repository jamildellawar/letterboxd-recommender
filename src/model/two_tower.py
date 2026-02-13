"""Two-Tower neural recommendation model with ONNX export."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ItemTower(nn.Module):
    """Maps item feature vectors to a 64-dim embedding."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)


class UserTower(nn.Module):
    """Attention-based aggregation of rated item embeddings into a user embedding."""

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        # Attention: input is item_embed + rating (embed_dim + 1)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, item_embeds: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
        """Compute user embedding from rated item embeddings.

        Args:
            item_embeds: (batch, n_items, embed_dim) item embeddings.
            ratings: (batch, n_items, 1) ratings for each item.

        Returns:
            (batch, embed_dim) L2-normalized user embedding.
        """
        # Concat embeddings and ratings for attention input
        att_input = torch.cat([item_embeds, ratings], dim=-1)  # (batch, n_items, embed_dim+1)
        att_weights = self.attention(att_input)  # (batch, n_items, 1)
        att_weights = F.softmax(att_weights, dim=1)

        # Weighted sum of item embeddings
        user_embed = (att_weights * item_embeds).sum(dim=1)  # (batch, embed_dim)
        return F.normalize(user_embed, p=2, dim=-1)


class BPRDataset(Dataset):
    """BPR triplet dataset: (user_items, user_ratings, positive_item, negative_item)."""

    def __init__(self, user_histories: list, item_embeddings: np.ndarray, n_items: int, max_history: int = 50):
        self.user_histories = user_histories
        self.item_embeddings = torch.tensor(item_embeddings, dtype=torch.float32)
        self.n_items = n_items
        self.max_history = max_history

        # Build triplets
        self.triplets = []
        all_item_set = set(range(n_items))
        for user_items, user_ratings in user_histories:
            positive_set = set(user_items)
            negatives = list(all_item_set - positive_set)
            if not negatives or len(user_items) < 2:
                continue
            for pos_idx in user_items:
                neg_idx = negatives[np.random.randint(len(negatives))]
                self.triplets.append((user_items, user_ratings, pos_idx, neg_idx))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        user_items, user_ratings, pos_idx, neg_idx = self.triplets[idx]

        # Pad/truncate history
        n = min(len(user_items), self.max_history)
        items = user_items[:n]
        ratings = user_ratings[:n]

        item_embeds = self.item_embeddings[items]
        rating_tensor = torch.tensor(ratings, dtype=torch.float32).unsqueeze(-1)

        # Pad if needed
        if n < self.max_history:
            pad_size = self.max_history - n
            item_embeds = torch.cat([item_embeds, torch.zeros(pad_size, item_embeds.shape[1])])
            rating_tensor = torch.cat([rating_tensor, torch.zeros(pad_size, 1)])

        pos_embed = self.item_embeddings[pos_idx]
        neg_embed = self.item_embeddings[neg_idx]

        return item_embeds, rating_tensor, pos_embed, neg_embed


class TwoTowerModel:
    """Two-tower recommendation model with BPR training and ONNX export."""

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, lr: float = 1e-3, epochs: int = 10):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.item_tower: ItemTower | None = None
        self.user_tower: UserTower | None = None
        self.item_embeddings: np.ndarray | None = None  # precomputed (n_items, embed_dim)
        self.tmdb_to_idx: dict[int, int] = {}
        self.idx_to_tmdb: dict[int, int] = {}

    def fit(
        self,
        ratings_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        id_mappings: dict,
        vectorizer,
    ) -> "TwoTowerModel":
        """Train the two-tower model on MovieLens data.

        Args:
            ratings_df: MovieLens ratings DataFrame.
            candidates_df: Candidate movies with TMDB metadata.
            id_mappings: From build_id_mappings().
            vectorizer: Fitted MovieVectorizer for item features.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ml_to_tmdb = id_mappings["ml_to_tmdb"]
        tmdb_to_ml = id_mappings["tmdb_to_ml"]

        # Build item features using the existing vectorizer
        shared_tmdb_ids = sorted(id_mappings["shared_tmdb_ids"])
        shared_candidates = candidates_df[candidates_df["tmdb_id"].isin(shared_tmdb_ids)].copy()

        if shared_candidates.empty:
            raise ValueError("No shared candidates between MovieLens and candidate pool.")

        # Reindex for contiguous indices
        shared_candidates = shared_candidates.reset_index(drop=True)
        self.tmdb_to_idx = {int(tid): i for i, tid in enumerate(shared_candidates["tmdb_id"])}
        self.idx_to_tmdb = {i: int(tid) for tid, i in self.tmdb_to_idx.items()}

        # Get feature vectors from the existing vectorizer
        item_feature_vectors = vectorizer.transform(shared_candidates)
        input_dim = item_feature_vectors.shape[1]
        n_items = len(shared_candidates)

        # Initialize towers
        self.item_tower = ItemTower(input_dim, self.hidden_dim, self.embed_dim).to(device)
        self.user_tower = UserTower(self.embed_dim).to(device)

        # Compute initial item embeddings
        with torch.no_grad():
            features_tensor = torch.tensor(item_feature_vectors, dtype=torch.float32).to(device)
            item_embeds_initial = self.item_tower(features_tensor).cpu().numpy()

        # Build user histories from MovieLens data
        valid_ml_ids = set(tmdb_to_ml.values())
        filtered = ratings_df[ratings_df["movieId"].isin(valid_ml_ids)].copy()

        # Map to contiguous indices
        ml_to_idx = {}
        for ml_id, tmdb_id in ml_to_tmdb.items():
            if tmdb_id in self.tmdb_to_idx:
                ml_to_idx[ml_id] = self.tmdb_to_idx[tmdb_id]

        user_histories = []
        for user_id, group in filtered.groupby("userId"):
            items = []
            ratings = []
            for _, row in group.iterrows():
                idx = ml_to_idx.get(int(row["movieId"]))
                if idx is not None:
                    items.append(idx)
                    ratings.append(float(row["rating"]))
            if len(items) >= 3:
                user_histories.append((items, ratings))

        print(f"  Users with 3+ rated items: {len(user_histories):,}")
        print(f"  Items: {n_items}, Feature dim: {input_dim}")

        # Sample users for training efficiency (cap at 50K)
        if len(user_histories) > 50000:
            indices = np.random.choice(len(user_histories), 50000, replace=False)
            user_histories = [user_histories[i] for i in indices]
            print(f"  Sampled down to {len(user_histories):,} users for training")

        # Create dataset and train
        dataset = BPRDataset(user_histories, item_embeds_initial, n_items)
        loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(
            list(self.item_tower.parameters()) + list(self.user_tower.parameters()),
            lr=self.lr,
        )

        self.item_tower.train()
        self.user_tower.train()

        print(f"  Training ({self.epochs} epochs, {len(dataset):,} triplets)...")
        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0

            for item_embeds, rating_tensors, pos_embeds, neg_embeds in loader:
                item_embeds = item_embeds.to(device)
                rating_tensors = rating_tensors.to(device)
                pos_embeds = pos_embeds.to(device)
                neg_embeds = neg_embeds.to(device)

                # Get user embedding
                user_embed = self.user_tower(item_embeds, rating_tensors)

                # BPR loss
                pos_score = (user_embed * pos_embeds).sum(dim=1)
                neg_score = (user_embed * neg_embeds).sum(dim=1)
                loss = -F.logsigmoid(pos_score - neg_score).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"    Epoch {epoch + 1}/{self.epochs}: loss={avg_loss:.4f}")

        # Precompute final item embeddings
        self.item_tower.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(item_feature_vectors, dtype=torch.float32).to(device)
            self.item_embeddings = self.item_tower(features_tensor).cpu().numpy()

        print(f"  Item embeddings: {self.item_embeddings.shape}")
        return self

    def predict(
        self,
        guest_ratings: dict[int, float],
        candidate_tmdb_ids: list[int] | set[int] | None = None,
    ) -> dict[int, float]:
        """Predict scores using the user tower + precomputed item embeddings.

        Args:
            guest_ratings: {tmdb_id: rating} from the guest user.
            candidate_tmdb_ids: Optional subset of tmdb_ids to score.

        Returns:
            {tmdb_id: predicted_score} for each scorable candidate.
        """
        if self.item_embeddings is None or self.user_tower is None:
            raise RuntimeError("Must call fit() or load_artifacts() before predict().")

        # Gather item embeddings and ratings for guest's movies
        items = []
        ratings = []
        for tmdb_id, rating in guest_ratings.items():
            idx = self.tmdb_to_idx.get(int(tmdb_id))
            if idx is not None:
                items.append(idx)
                ratings.append(rating)

        if not items:
            return {}

        # Build tensors
        item_embeds = torch.tensor(
            self.item_embeddings[items], dtype=torch.float32
        ).unsqueeze(0)  # (1, n_items, embed_dim)
        rating_tensor = torch.tensor(
            ratings, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(-1)  # (1, n_items, 1)

        # Get user embedding
        self.user_tower.eval()
        with torch.no_grad():
            user_embed = self.user_tower(item_embeds, rating_tensor).numpy()  # (1, embed_dim)

        # Score candidates via dot product
        target_tmdb_ids = set(candidate_tmdb_ids) if candidate_tmdb_ids else set(self.tmdb_to_idx.keys())
        results = {}

        for tmdb_id in target_tmdb_ids:
            if tmdb_id in guest_ratings:
                continue
            idx = self.tmdb_to_idx.get(int(tmdb_id))
            if idx is None:
                continue
            score = float(np.dot(user_embed[0], self.item_embeddings[idx]))
            results[tmdb_id] = score

        return results

    def export_onnx(self, output_dir: Path) -> None:
        """Export UserTower to ONNX and save item embeddings."""
        import onnx

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export UserTower to ONNX
        self.user_tower.eval()
        dummy_items = torch.randn(1, 10, self.embed_dim)
        dummy_ratings = torch.randn(1, 10, 1)

        onnx_path = output_dir / "user_tower.onnx"
        torch.onnx.export(
            self.user_tower,
            (dummy_items, dummy_ratings),
            str(onnx_path),
            input_names=["item_embeds", "ratings"],
            output_names=["user_embed"],
            dynamic_axes={
                "item_embeds": {0: "batch", 1: "n_items"},
                "ratings": {0: "batch", 1: "n_items"},
                "user_embed": {0: "batch"},
            },
            opset_version=17,
        )

        # Verify
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

        # Save item embeddings
        np.savez_compressed(
            output_dir / "item_embeddings.npz",
            embeddings=self.item_embeddings,
        )

        # Save mappings
        with open(output_dir / "two_tower_mappings.pkl", "wb") as f:
            pickle.dump({
                "tmdb_to_idx": self.tmdb_to_idx,
                "idx_to_tmdb": self.idx_to_tmdb,
                "embed_dim": self.embed_dim,
            }, f)

        total_size = sum(f.stat().st_size for f in output_dir.glob("*")) / (1024 * 1024)
        print(f"  Exported Two-Tower ONNX artifacts ({total_size:.1f} MB)")

    @classmethod
    def load_artifacts(cls, artifacts_dir: Path, use_onnx: bool = True) -> "TwoTowerModel":
        """Load model from saved artifacts.

        Args:
            artifacts_dir: Directory with exported artifacts.
            use_onnx: If True, loads ONNX model for inference (no PyTorch needed).
        """
        artifacts_dir = Path(artifacts_dir)

        with open(artifacts_dir / "two_tower_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)

        data = np.load(artifacts_dir / "item_embeddings.npz")
        item_embeddings = data["embeddings"]

        model = cls(embed_dim=mappings["embed_dim"])
        model.tmdb_to_idx = mappings["tmdb_to_idx"]
        model.idx_to_tmdb = mappings["idx_to_tmdb"]
        model.item_embeddings = item_embeddings

        if use_onnx:
            import onnxruntime as ort
            model._onnx_session = ort.InferenceSession(
                str(artifacts_dir / "user_tower.onnx")
            )
        else:
            raise ValueError("Non-ONNX loading requires PyTorch model checkpoint (not implemented).")

        return model

    def predict_onnx(
        self,
        guest_ratings: dict[int, float],
        candidate_tmdb_ids: list[int] | set[int] | None = None,
    ) -> dict[int, float]:
        """Predict using ONNX runtime (for Lambda inference)."""
        if not hasattr(self, "_onnx_session") or self._onnx_session is None:
            raise RuntimeError("ONNX session not loaded. Use load_artifacts(use_onnx=True).")

        # Gather item embeddings and ratings
        items = []
        ratings = []
        for tmdb_id, rating in guest_ratings.items():
            idx = self.tmdb_to_idx.get(int(tmdb_id))
            if idx is not None:
                items.append(idx)
                ratings.append(rating)

        if not items:
            return {}

        item_embeds = self.item_embeddings[items][np.newaxis, ...]  # (1, n, embed_dim)
        rating_array = np.array(ratings, dtype=np.float32)[np.newaxis, :, np.newaxis]  # (1, n, 1)

        # Run ONNX inference
        result = self._onnx_session.run(
            ["user_embed"],
            {"item_embeds": item_embeds.astype(np.float32), "ratings": rating_array},
        )
        user_embed = result[0][0]  # (embed_dim,)

        # Score candidates
        target_tmdb_ids = set(candidate_tmdb_ids) if candidate_tmdb_ids else set(self.tmdb_to_idx.keys())
        results = {}

        for tmdb_id in target_tmdb_ids:
            if tmdb_id in guest_ratings:
                continue
            idx = self.tmdb_to_idx.get(int(tmdb_id))
            if idx is None:
                continue
            score = float(np.dot(user_embed, self.item_embeddings[idx]))
            results[tmdb_id] = score

        return results
