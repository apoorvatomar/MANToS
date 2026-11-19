import json
import sqlite3

from langchain.tools import tool

from src.tools.database_schema import database_path


def execute_sql_raw(db_id: str, sql: str) -> dict:
    """Execute SQL and return a JSON-serializable result."""
    try:
        with sqlite3.connect(database_path(db_id)) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description or []]

        return {
            "status": "success",
            "columns": columns,
            "rows": rows[:50],
            "row_count": len(rows),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


@tool
def execute_sql(db_id: str, sql: str) -> str:
    """Execute a candidate SQL query against the selected Spider SQLite DB."""
    return json.dumps(execute_sql_raw(db_id, sql), default=str)
