from typing import Any, Optional, TypedDict

class Attempt(TypedDict, total=False):
    sql: str
    critique: dict[str, Any]
    execution: dict[str, Any]
    validation: dict[str, Any]

class NL2SQLState(TypedDict, total=False):
    question: str
    db_id: Optional[str]
    db_candidates: list[dict[str, Any]]
    db_id_prediction: Optional[str]
    db_prediction_score: Optional[float]
    intent: Optional[str]
    feasibility: Optional[str]
    complexity: Optional[str]
    plan: list[str]
    orchestrator_analysis: dict[str, Any]
    relevant_tables: list[str]
    relevant_columns: list[str]
    join_path: list[str]
    schema_context: dict[str, Any]
    sql_candidate: str
    validation: dict[str, Any]
    execution: dict[str, Any]
    critique: dict[str, Any]
    retry_count: int
    max_retries: int
    attempts: list[Attempt]
    final_sql: Optional[str]
    status: str
