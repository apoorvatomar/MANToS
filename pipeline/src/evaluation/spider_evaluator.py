import json
import os
from typing import Any

from src.tools.execute_sql import execute_sql_raw


def load_spider_file(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def execution_match(
    db_id: str,
    predicted_sql: str,
    gold_sql: str,
) -> bool:
    pred = execute_sql_raw(db_id, predicted_sql)
    gold = execute_sql_raw(db_id, gold_sql)

    if pred.get("status") != "success" or gold.get("status") != "success":
        return False

    # SQL result comparison should be order-insensitive for the common
    # Spider execution-accuracy setting. Preserve duplicates via sorting.
    pred_rows = sorted(
        [tuple(map(str, row)) for row in pred.get("rows", [])]
    )
    gold_rows = sorted(
        [tuple(map(str, row)) for row in gold.get("rows", [])]
    )

    return pred_rows == gold_rows


def evaluate_predictions(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(predictions)
    execution_correct = sum(
        1 for x in predictions if x.get("execution_correct")
    )
    exact_correct = sum(
        1 for x in predictions if x.get("exact_match")
    )

    return {
        "total": total,
        "execution_accuracy": execution_correct / total if total else 0.0,
        "exact_match": exact_correct / total if total else 0.0,
    }


def evaluate_spider_dev(
    dev_path: str,
    max_examples: int | None = None,
) -> dict[str, Any]:
    from src.pipeline import run_nlq_to_sql

    data = load_spider_file(dev_path)
    if max_examples:
        data = data[:max_examples]

    predictions = []

    for i, item in enumerate(data, 1):
        result = run_nlq_to_sql(
            question=item["question"],
            db_id=item["db_id"],
        )

        predicted = result.get("final_sql", "") or ""
        gold = item["query"]

        predictions.append({
            "index": i,
            "db_id": item["db_id"],
            "question": item["question"],
            "gold_sql": gold,
            "predicted_sql": predicted,
            "exact_match": (
                predicted.strip().lower()
                == gold.strip().lower()
            ),
            "execution_correct": execution_match(
                item["db_id"],
                predicted,
                gold,
            ) if predicted else False,
            "status": result.get("status"),
            "retries": result.get("retry_count", 0),
        })

    metrics = evaluate_predictions(predictions)
    return {
        "metrics": metrics,
        "predictions": predictions,
    }
