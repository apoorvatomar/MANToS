import json
import sqlite3

from langchain.tools import tool

from src.tools.database_schema import database_path


@tool
def get_sample_data(db_id: str, table_name: str) -> str:
    """Return up to five rows from a Spider table."""
    try:
        with sqlite3.connect(database_path(db_id)) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 5')
            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description or []]

        return json.dumps(
            {"table": table_name, "columns": columns, "rows": rows},
            default=str,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})
