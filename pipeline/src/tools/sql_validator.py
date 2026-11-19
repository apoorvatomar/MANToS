import re
from typing import Any

from src.tools.database_schema import read_database_schema
from src.tools.execute_sql import execute_sql_raw


def strip_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


def validate_sql(db_id: str, sql: str) -> dict[str, Any]:
    """
    Deterministic checks:
      1. database exists / schema can be read
      2. SQLite can parse/execute the query
      3. execution errors are surfaced to the Critique Agent
    """
    sql = strip_sql(sql)

    try:
        schema = read_database_schema(db_id)
    except Exception as exc:
        return {
            "status": "INVALID",
            "syntax_valid": False,
            "schema_valid": False,
            "error": str(exc),
        }

    execution = execute_sql_raw(db_id, sql)

    # SQLite execution is a strong combined syntax/schema check.
    valid = execution["status"] == "success"

    return {
        "status": "VALID" if valid else "INVALID",
        "syntax_valid": valid,
        "schema_valid": valid,
        "error": execution.get("error"),
        "available_tables": [t["table"] for t in schema["tables"]],
    }
