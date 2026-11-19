"""Train the Spider DB-ID bi-encoder.

This module is compatible with the Colab training notebook and can also be run
from the command line. It uses Sentence-Transformers' trainer with duplicate-free
batches, mixed precision, validation retrieval metrics, and resumable checkpoints.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import (
    BatchSamplers,
    InputExample,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from datasets import Dataset


def database_schema_text(item: dict[str, Any]) -> str:
    """Build the same schema representation used by runtime inference."""
    tables = item["table_names_original"]
    columns = item["column_names_original"]
    column_types = item["column_types"]
    primary_keys = set(item["primary_keys"])
    foreign_keys = item["foreign_keys"]

    lines = []
    for table_idx, table_name in enumerate(tables):
        lines.append(f"TABLE: {table_name}")
        for col_idx, (table_id, col_name) in enumerate(columns):
            if table_id != table_idx:
                continue
            line = f"  COLUMN: {col_name} TYPE: {column_types[col_idx]}"
            if col_idx in primary_keys:
                line += " PRIMARY_KEY"
            lines.append(line)

    if foreign_keys:
        lines.append("FOREIGN_KEYS:")
        for source_col, target_col in foreign_keys:
            source_table_id, source_col_name = columns[source_col]
            target_table_id, target_col_name = columns[target_col]
            lines.append(
                f"  {tables[source_table_id]}.{source_col_name} -> "
                f"{tables[target_table_id]}.{target_col_name}"
            )
    return "\n".join(lines)


def load_catalog(tables_path: str) -> dict[str, str]:
    with open(tables_path, "r", encoding="utf-8") as f:
        tables = json.load(f)
    return {x["db_id"]: database_schema_text(x) for x in tables}


def load_pairs(data_path: str, catalog: dict[str, str]) -> list[dict[str, str]]:
    with open(data_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    pairs = []
    for item in rows:
        db_id = item["db_id"]
        schema = catalog.get(db_id)
        if schema:
            pairs.append(
                {
                    "question": item["question"],
                    "schema": schema,
                    "db_id": db_id,
                }
            )
    return pairs


def build_dataset(pairs: list[dict[str, str]]) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [x["question"] for x in pairs],
            "schema": [x["schema"] for x in pairs],
        }
    )


class RetrievalEvaluator:
    """Small evaluator used by the notebook/CLI after training."""

    def __init__(self, model, dev_pairs, catalog, batch_size=64):
        self.model = model
        self.dev_pairs = dev_pairs
        self.db_ids = list(catalog.keys())
        self.schemas = [catalog[x] for x in self.db_ids]
        self.batch_size = batch_size

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        q = self.model.encode(
            [x["question"] for x in self.dev_pairs],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        s = self.model.encode(
            self.schemas,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = q @ s.T
        gold = {x["db_id"]: i for i, x in enumerate(self.db_ids)}

        metrics = {}
        for k in (1, 3, 5, 10):
            hits = 0
            for i, row in enumerate(self.dev_pairs):
                order = scores[i].argsort()[::-1][:k]
                hits += int(gold[row["db_id"]] in order)
            metrics[f"top{k}"] = hits / len(self.dev_pairs)
        rr = []
        for i, row in enumerate(self.dev_pairs):
            order = scores[i].argsort()[::-1]
            rank = int((order == gold[row["db_id"]]).nonzero()[0][0]) + 1
            rr.append(1.0 / rank)
        metrics["mrr"] = sum(rr) / len(rr)
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", default=None)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--output", default="models/db_id_predictor")
    parser.add_argument("--checkpoint-dir", default="models/db_id_predictor_checkpoints")
    parser.add_argument("--base-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available; training will be much slower.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))

    catalog = load_catalog(args.tables)
    train_pairs = load_pairs(args.train, catalog)
    dev_pairs = load_pairs(args.dev, catalog) if args.dev else []

    if not train_pairs:
        raise RuntimeError("No training pairs were created.")

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_length

    train_ds = build_dataset(train_pairs)
    loss = losses.MultipleNegativesRankingLoss(model)

    output_dir = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.10,
        fp16=use_fp16,
        bf16=use_bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=2,
        tf32=bool(torch.cuda.is_available()),
        seed=42,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        loss=loss,
    )

    if args.resume:
        print("Resuming from:", args.resume)
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    model.save(str(output_dir))

    print(f"Saved final model to: {output_dir}")
    if dev_pairs:
        metrics = RetrievalEvaluator(model, dev_pairs, catalog).evaluate()
        print("Dev retrieval:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
