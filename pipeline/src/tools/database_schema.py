import json
import os
import sqlite3

from langchain.tools import tool


SPIDER_DB_PATH = os.getenv("SPIDER_DB_PATH", "spider_data/database")


def database_path(db_id: str) -> str:
    return os.path.join(SPIDER_DB_PATH, db_id, f"{db_id}.sqlite")


def read_database_schema(db_id: str) -> dict:
    """Deterministic schema reader used by agents and evaluation."""
    db_path = database_path(db_id)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Spider database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        tables = cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        schema = {"db_id": db_id, "tables": []}

        for (table,) in tables:
            columns = cursor.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()

            foreign_keys = cursor.execute(
                f'PRAGMA foreign_key_list("{table}")'
            ).fetchall()

            schema["tables"].append(
                {
                    "table": table,
                    "columns": [
                        {
                            "name": col[1],
                            "type": col[2],
                            "primary_key": bool(col[5]),
                        }
                        for col in columns
                    ],
                    "foreign_keys": [
                        {
                            "column": fk[3],
                            "references_table": fk[2],
                            "references_column": fk[4],
                        }
                        for fk in foreign_keys
                    ],
                }
            )

    return schema


@tool
def get_database_schema(db_id: str) -> str:
    """Return the complete Spider database schema as JSON."""
    try:
        return json.dumps(read_database_schema(db_id), indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def list_spider_databases() -> str:
    """List available Spider database IDs."""
    if not os.path.isdir(SPIDER_DB_PATH):
        return json.dumps({"error": f"Directory not found: {SPIDER_DB_PATH}"})

    dbs = sorted(
        name
        for name in os.listdir(SPIDER_DB_PATH)
        if os.path.isdir(os.path.join(SPIDER_DB_PATH, name))
    )
    return json.dumps(dbs)
