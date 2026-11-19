"""Reference-vs-predicted SQL execution comparison for Spider evaluation."""
from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from typing import Any

from src.tools.database_schema import database_path


def _execute_full(db_id: str, sql: str) -> dict[str, Any]:
    """Execute SQL and retain all rows for comparison (not the UI's 50-row cap)."""
    try:
        with sqlite3.connect(database_path(db_id)) as conn:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description or []]
        return {"status": "success", "columns": columns, "rows": rows, "row_count": len(rows)}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "row_count": 0}


def _normal_value(value: Any) -> tuple:
    if value is None:
        return ("null",)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isnan(number):
            return ("number", "nan")
        if math.isinf(number):
            return ("number", "inf" if number > 0 else "-inf")
        # SQLite numeric results are compared with a small tolerance.
        return ("number", round(number, 6))
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, bool):
        return ("bool", value)
    return ("text", str(value))


def _normal_row(row: tuple) -> tuple:
    return tuple(_normal_value(value) for value in row)


def _has_order_by(sql: str) -> bool:
    # Sufficient for ordinary Spider SQL; ignores ORDER BY inside comments only
    # approximately, which is acceptable for this reference comparator.
    return bool(re.search(r"\border\s+by\b", sql or "", flags=re.IGNORECASE))


def compare_result_sets(gold_result: dict, predicted_result: dict, gold_sql: str) -> dict[str, Any]:
    """Compare result values robustly; preserve order only when gold SQL orders rows."""
    if gold_result.get("status") != "success" or predicted_result.get("status") != "success":
        return {
            "match": False,
            "comparison_type": "EXECUTION_ERROR",
            "reason": "One or both SQL statements failed to execute.",
        }

    gold_rows = [tuple(row) for row in gold_result.get("rows", [])]
    pred_rows = [tuple(row) for row in predicted_result.get("rows", [])]
    if len(gold_result.get("columns", [])) != len(predicted_result.get("columns", [])):
        return {"match": False, "comparison_type": "COLUMN_COUNT_MISMATCH", "reason": "Result column counts differ."}
    if len(gold_rows) != len(pred_rows):
        return {
            "match": False,
            "comparison_type": "ROW_COUNT_MISMATCH",
            "reason": f"Gold returned {len(gold_rows)} rows; predicted returned {len(pred_rows)} rows.",
        }

    normalized_gold = [_normal_row(row) for row in gold_rows]
    normalized_pred = [_normal_row(row) for row in pred_rows]
    if _has_order_by(gold_sql):
        match = normalized_gold == normalized_pred
        comparison_type = "ORDERED_RESULT_MATCH" if match else "ORDERED_RESULT_MISMATCH"
    else:
        # Counter preserves duplicate-row multiplicity while ignoring unspecified order.
        match = Counter(normalized_gold) == Counter(normalized_pred)
        comparison_type = "UNORDERED_RESULT_MATCH" if match else "UNORDERED_RESULT_MISMATCH"
    return {
        "match": match,
        "comparison_type": comparison_type,
        "reason": "Result sets match." if match else "Result values differ after normalization.",
    }


def compare_gold_and_predicted_sql(
    gold_db_id: str,
    gold_sql: str,
    predicted_db_id: str,
    predicted_sql: str,
) -> dict[str, Any]:
    """Run gold SQL on gold DB and predicted SQL on predicted DB, then compare."""
    gold_result = _execute_full(gold_db_id, gold_sql)
    predicted_result = _execute_full(predicted_db_id, predicted_sql)
    comparison = compare_result_sets(gold_result, predicted_result, gold_sql)

    # Keep the full row sets internally for comparison but bound the returned payload.
    gold_public = {**gold_result, "rows": gold_result.get("rows", [])[:50]}
    predicted_public = {**predicted_result, "rows": predicted_result.get("rows", [])[:50]}
    # Expose only the boolean database-fit signal to downstream agents/logs.
    # The reference database ID is used internally for evaluation but is not
    # included in the comparison payload, printed trace, or returned state.
    return {
        "database_match": predicted_db_id == gold_db_id,
        "predicted_db_id": predicted_db_id,
        "gold_execution": gold_public,
        "predicted_execution": predicted_public,
        "comparison": comparison,
    }
