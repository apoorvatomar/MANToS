import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.db_prediction.schema_catalog import build_schema_catalog

DB_MODEL_PATH = os.getenv("DB_MODEL_PATH", "models/db_id_predictor")
DB_DEVICE = os.getenv("DB_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


class DBIDPredictor:
    """BGE-style bi-encoder used as a high-recall DB retriever."""

    def __init__(
        self,
        model_path: str | None = None,
        tables_path: str | None = None,
        device: str | None = None,
        embedding_batch_size: int = 64,
    ):
        self.model_path = model_path or DB_MODEL_PATH
        self.tables_path = tables_path
        self.device = device or DB_DEVICE

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Trained DB-ID model not found at '{self.model_path}'. "
                "Set DB_MODEL_PATH to your saved bi-encoder."
            )

        self.model = SentenceTransformer(self.model_path, device=self.device)
        self.catalog = build_schema_catalog(self.tables_path)
        self.db_ids = list(self.catalog.keys())
        self.schema_texts = [self.catalog[x] for x in self.db_ids]

        # Schema embeddings are static for a loaded model, so compute them once.
        self.schema_embeddings = self.model.encode(
            self.schema_texts,
            batch_size=embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def predict(self, question: str, top_k: int = 10) -> list[dict[str, Any]]:
        top_k = min(top_k, len(self.db_ids))
        q = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        scores = self.schema_embeddings @ q
        order = np.argsort(-scores)[:top_k]
        return [
            {
                "rank": rank,
                "db_id": self.db_ids[i],
                "bi_score": float(scores[i]),
                "schema": self.schema_texts[i],
            }
            for rank, i in enumerate(order, start=1)
        ]

    def predict_top1(self, question: str) -> dict[str, Any]:
        return self.predict(question, top_k=1)[0]
