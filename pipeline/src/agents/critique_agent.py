import json

from langchain.agents import create_agent

from src.tools.execute_sql import execute_sql
from src.tools.database_schema import get_database_schema
from src.utils.llm_loader import get_llm
from src.utils.json_extractor import extract_json_from_string


def build_critique_agent():
    return create_agent(
        model=get_llm(),
        tools=[get_database_schema, execute_sql],
        system_prompt="""
You are the CRITIQUE AGENT in a Spider Text-to-SQL system. Be a strict verifier, not
an encourager. A query can execute successfully and still be wrong.

Evaluate these dimensions independently:
1. Syntax: valid SQLite syntax.
2. Schema: all referenced tables and columns exist in the SELECTED database.
3. Semantic: the SQL answers the complete natural-language question, including the
   correct entity, aggregation, filters, ranking, grouping, joins, and result cardinality.
4. Execution: deterministic validation says VALID and execution status is success.
5. Database fit: the selected database's purpose/entity rules and schema support the
   specific question, not merely a similarly named table/column. Prefer specific rule
   matches over generic entity overlap. A generic stadium/capacity match alone is weak
   evidence when another rule explicitly describes stadium-capacity aggregates.

Critical rules:
- Successful execution is NOT proof of semantic correctness.
- Do not claim that results match Spider gold unless a gold result was explicitly supplied.
- The database ID passed to you is the one selected by the Schema Agent. Do not invent
  alternate database IDs or output a candidate list.
- Use the selected database's purpose, matched rules, and schema context as evidence.
- If the selected database appears only weakly related to the question, fail semantic
  and explain why. Do not pass solely because the SQL references existing columns.
- If deterministic validation is not VALID, execution did not succeed, or required
  tables/columns are missing, the overall status must be FAIL regardless of the LLM's
  impression.
- If FAIL, correction should contain corrected SQL only when it can be inferred safely;
  otherwise use an empty string and explain what is uncertain.

Return ONLY valid JSON with exactly these keys:
{
  "status": "PASS" or "FAIL",
  "syntax": "PASS" or "FAIL",
  "schema": "PASS" or "FAIL",
  "semantic": "PASS" or "FAIL",
  "execution": "PASS" or "FAIL",
  "database_fit": "PASS" or "FAIL",
  "reason": "specific explanation",
  "correction": "corrected SQL if safely inferable, otherwise empty string"
}
Do not include markdown fences or candidate reasoning.
""",
    )


def run_critique_agent(
    question: str,
    db_id: str,
    schema_context: dict,
    sql: str,
    validation: dict,
    execution: dict,
    result_comparison: dict | None = None,
) -> dict:
    agent = build_critique_agent()

    prompt = f"""
QUESTION:
{question}

SINGLE SELECTED DATABASE ID:
{db_id}

SELECTED DATABASE EVIDENCE (not a list of alternatives):
{json.dumps({key: schema_context.get(key) for key in (
    "db_id", "db_confidence", "selection_reason", "database_purpose",
    "matched_entity_rules", "relevant_tables", "relevant_columns", "join_path", "schema_context"
)}, ensure_ascii=False, default=str)}

SQL CANDIDATE:
{sql}

DETERMINISTIC VALIDATION:
{json.dumps(validation, ensure_ascii=False, default=str)}

SQL EXECUTION RESULT:
{json.dumps(execution, ensure_ascii=False, default=str)}

REFERENCE EXECUTION COMPARISON (may be null when no Spider reference is supplied):
{json.dumps(result_comparison, ensure_ascii=False, default=str)}

The comparison may include a boolean field named database_match. It indicates only
whether the selected database matches the evaluation reference. The reference database
identifier is intentionally not provided. Use database_match internally to route a
failure: when false, recommend database reselection; when true, preserve the selected
DB and focus on SQL semantics. Never request, infer, reveal, or print the reference
DB identifier. Do not equate successful execution with a correct answer. If reference
results differ, fail. Return only one critique JSON object; do not output candidate lists.
"""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    content = result["messages"][-1].content
    parsed = extract_json_from_string(content)

    if parsed is None:
        # Deterministic execution evidence remains authoritative even if the LLM
        # critique response is malformed.
        comparison = (result_comparison or {}).get("comparison", {})
        if comparison.get("match") is True and str(validation.get("status", "")).upper() == "VALID" and str(execution.get("status", "")).lower() == "success":
            return {
                "status": "PASS", "syntax": "PASS", "schema": "PASS",
                "semantic": "PASS", "execution": "PASS", "database_fit": "PASS",
                "reason": "Predicted and reference execution result sets match; LLM critique output was malformed.",
                "correction": "", "failure_type": "", "retry_action": "NONE",
                "feedback_to_schema_agent": {},
            }
        is_db_mismatch = bool(result_comparison and result_comparison.get("database_match") is False)
        mismatch = comparison.get("match") is False
        return {
            "status": "FAIL", "syntax": "UNKNOWN", "schema": "UNKNOWN",
            "semantic": "FAIL" if mismatch else "UNKNOWN", "execution": "FAIL" if str(execution.get("status", "")).lower() != "success" else "UNKNOWN",
            "database_fit": "FAIL" if mismatch and is_db_mismatch else "UNKNOWN",
            "reason": f"Critique Agent returned non-JSON output: {content}",
            "correction": "",
            "failure_type": "DATABASE_SELECTION_MISMATCH" if mismatch and is_db_mismatch else ("SQL_RESULT_MISMATCH" if mismatch else "CRITIQUE_PARSE_ERROR"),
            "retry_action": "RESELECT_DATABASE" if mismatch and is_db_mismatch else ("REPAIR_SQL" if mismatch else "NONE"),
            "feedback_to_schema_agent": {"action": "RECONSIDER_DATABASE_SELECTION", "reason": comparison.get("reason", "Reference result mismatch."), "exclude_current_db": True} if mismatch and is_db_mismatch else {},
        }

    # Deterministic gates override any optimistic LLM verdict.
    validation_ok = str(validation.get("status", "")).upper() == "VALID"
    execution_ok = str(execution.get("status", "")).lower() == "success"
    if not validation_ok:
        parsed["syntax"] = "FAIL" if validation.get("syntax_valid") is False else parsed.get("syntax", "FAIL")
        parsed["schema"] = "FAIL" if validation.get("schema_valid") is False else parsed.get("schema", "FAIL")
        parsed["execution"] = "FAIL"
    if not execution_ok:
        parsed["execution"] = "FAIL"
    if not validation_ok or not execution_ok:
        parsed["status"] = "FAIL"

    if result_comparison is not None:
        comparison = result_comparison.get("comparison", {})
        if comparison.get("match") is False:
            parsed["status"] = "FAIL"
            parsed["semantic"] = "FAIL"
            if result_comparison.get("database_match") is False:
                parsed["database_fit"] = "FAIL"
                parsed["failure_type"] = "DATABASE_SELECTION_MISMATCH"
                parsed["retry_action"] = "RESELECT_DATABASE"
                parsed["feedback_to_schema_agent"] = {
                    "action": "RECONSIDER_DATABASE_SELECTION",
                    "reason": comparison.get("reason", "Reference and predicted results differ."),
                    "exclude_current_db": True,
                }
            else:
                parsed["failure_type"] = "SQL_RESULT_MISMATCH"
                parsed["retry_action"] = "REPAIR_SQL"
                parsed["feedback_to_schema_agent"] = {
                    "action": "KEEP_DATABASE_REVIEW_SQL",
                    "reason": "The selected database matches the reference database, so the SQL semantics should be repaired before changing databases.",
                    "exclude_current_db": False,
                }
            parsed["reason"] = (
                str(parsed.get("reason", "")) + " Reference comparison: " +
                str(comparison.get("reason", "result mismatch"))
            ).strip()
        elif comparison.get("match") is True and validation_ok and execution_ok:
            # In reference-assisted evaluation, a normalized execution match is the
            # decisive correctness signal requested by the user.
            parsed["status"] = "PASS"
            parsed["semantic"] = "PASS"
            parsed["execution"] = "PASS"
            parsed["failure_type"] = ""
            parsed["retry_action"] = "NONE"
            parsed["feedback_to_schema_agent"] = {}
            parsed["reason"] = (
                str(parsed.get("reason", "")).strip()
                + " Predicted and reference execution result sets match."
            ).strip()

    # Normalize required keys so pipeline callers always receive a predictable shape.
    defaults = {
        "status": "FAIL", "syntax": "UNKNOWN", "schema": "UNKNOWN",
        "semantic": "UNKNOWN", "execution": "UNKNOWN", "database_fit": "UNKNOWN",
        "reason": "Critique did not provide a reason.", "correction": "",
        "failure_type": "", "retry_action": "NONE", "feedback_to_schema_agent": {},
    }
    for key, value in defaults.items():
        parsed.setdefault(key, value)
    parsed["status"] = str(parsed["status"]).upper()
    for key in ("syntax", "schema", "semantic", "execution", "database_fit"):
        parsed[key] = str(parsed[key]).upper()
    # Do not allow the agent to leak candidate lists into critique output.
    allowed = set(defaults)
    return {key: parsed.get(key, value) for key, value in defaults.items()}
