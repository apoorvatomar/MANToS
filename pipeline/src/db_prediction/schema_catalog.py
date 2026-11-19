import json
import os
from functools import lru_cache
from typing import Any

SPIDER_TABLES_PATH = os.getenv("SPIDER_TABLES_PATH", "spider_data/tables.json")

@lru_cache(maxsize=1)
def load_tables_metadata(path: str | None = None) -> tuple[dict[str, Any], ...]:
    path = path or SPIDER_TABLES_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Spider tables.json not found: {path}. Set SPIDER_TABLES_PATH.")
    with open(path, "r", encoding="utf-8") as f:
        return tuple(json.load(f))

def database_schema_text(item: dict[str, Any]) -> str:
    """V2 schema representation. DB ID is deliberately excluded."""
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
            lines.append(f"  {tables[source_table_id]}.{source_col_name} -> {tables[target_table_id]}.{target_col_name}")
    return "\n".join(lines)

@lru_cache(maxsize=1)
def build_schema_catalog(path: str | None = None) -> dict[str, str]:
    return {item["db_id"]: database_schema_text(item) for item in load_tables_metadata(path)}
