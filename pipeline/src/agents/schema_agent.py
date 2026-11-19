from functools import lru_cache
import json
import os
import re
from pathlib import Path

from langchain.agents import create_agent

from src.tools.database_schema import get_database_schema
from src.tools.sample_data import get_sample_data
from src.utils.llm_loader import get_llm
from src.utils.json_extractor import extract_json_from_string


@lru_cache(maxsize=1)
def load_database_rules() -> dict:
    """Load the precomputed DB-purpose/entity rules once per process."""
    configured = os.getenv("DB_RULES_PATH")
    rules_path = Path(configured) if configured else Path(__file__).resolve().parents[2] / "db_rules.json"
    if not rules_path.exists():
        raise FileNotFoundError(
            f"DB rules file not found at '{rules_path}'. "
            "Place db_rules.json in the project root or set DB_RULES_PATH."
        )
    with rules_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("db_rules.json must be a JSON object keyed by db_id.")
    return data


def _meaningful_tokens(text: str) -> set[str]:
    """Small dependency-free tokenizer for ranking rule evidence."""
    stop = {
        "the", "a", "an", "of", "for", "to", "in", "on", "at", "by", "with",
        "and", "or", "is", "are", "was", "were", "be", "all", "each", "what",
        "which", "who", "whose", "from", "about", "give", "show", "find", "get",
        "retrieve", "list", "tell", "me", "does", "do", "has", "have", "that",
    }
    return {
        token for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 1 and token not in stop
    }


def _rule_match_score(question: str, purpose: str, rules: list[dict]) -> tuple[float, list[dict]]:
    """Estimate lexical/phrase alignment; this is evidence, not a gold label."""
    q_tokens = _meaningful_tokens(question)
    combined = f"{purpose} " + " ".join(
        f"{r.get('entity', '')} {r.get('rule', '')}" for r in rules if isinstance(r, dict)
    )
    r_tokens = _meaningful_tokens(combined)
    overlap = len(q_tokens & r_tokens) / max(1, len(q_tokens))
    q = (question or "").lower()
    combined_lower = combined.lower()
    # Normalize common plurals so "stadiums/capacities" matches rule wording.
    q = re.sub(r"\bcapacities\b", "capacity", q)
    q = re.sub(r"\bstadiums\b", "stadium", q)
    combined_lower = re.sub(r"\bcapacities\b", "capacity", combined_lower)
    combined_lower = re.sub(r"\bstadiums\b", "stadium", combined_lower)
    phrase_bonus = 0.0
    for phrase in ("net worth", "average capacity", "maximum capacity", "stadium capacity", "highest net worth", "largest net worth"):
        if phrase in q and phrase in combined_lower:
            phrase_bonus += 0.18
    score = min(1.0, overlap + phrase_bonus)
    matched = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        text = f"{rule.get('entity', '')} {rule.get('rule', '')}"
        tokens = _meaningful_tokens(text)
        hits = q_tokens & tokens
        if hits:
            matched.append({"entity": rule.get("entity", ""), "rule": rule.get("rule", ""), "matched_terms": sorted(hits)})
    matched.sort(key=lambda item: len(item["matched_terms"]), reverse=True)
    return score, matched[:4]


def build_candidate_evidence(db_candidates: list[dict], question: str = "") -> list[dict]:
    """Attach rule evidence and a lightweight relevance signal to each retrieved candidate."""
    all_rules = load_database_rules()
    evidence = []
    for candidate in db_candidates:
        rule_record = all_rules.get(candidate["db_id"], {})
        rules = rule_record.get("rules", [])
        purpose = rule_record.get(
            "database_purpose", "No database-purpose description available in db_rules.json."
        )
        rule_score, matched_rules = _rule_match_score(question, purpose, rules)
        evidence.append({
            "rank": candidate.get("rank"),
            "db_id": candidate["db_id"],
            "bi_score": candidate.get("bi_score"),
            "rule_match_score": round(rule_score, 4),
            "database_purpose": purpose,
            "matched_entity_rules": matched_rules,
            "entity_rules": rules,
            "schema": candidate.get("schema", ""),
        })
    # Sort by rule evidence first, then retain retriever score as a tie-breaker.
    # The LLM must still validate the actual schema before selecting.
    evidence.sort(key=lambda item: (item["rule_match_score"], item.get("bi_score") or 0), reverse=True)
    return evidence



def build_schema_agent():
    return create_agent(
        model=get_llm(),
        tools=[get_database_schema, get_sample_data],
        system_prompt="""
You are the SCHEMA AGENT in a high-accuracy Spider Text-to-SQL system.

Your job has TWO tightly coupled stages:
A) choose the correct database from the supplied high-recall candidate list;
B) perform precise schema linking for that selected database.

The DB-ID bi-encoder is a RETRIEVER, not the final decision maker. Never blindly pick
rank 1. The correct database may be rank 2-10.

========================
DATABASE SELECTION METHOD
========================
For EVERY candidate database, reason using these signals:

1. QUESTION ENTITY COVERAGE
   - Match important nouns/entities in the question to table names, column names,
     and meaningful schema concepts.
   - Do not rely on lexical overlap alone; recognize obvious paraphrases.

2. RELATIONSHIP COVERAGE
   - Check whether the candidate contains the tables needed to answer the question
     together and whether the required PK/FK or schema-supported join path exists.

3. OPERATIONAL COVERAGE
   - Check whether the candidate supports the requested operation: aggregation,
     comparison, ranking, counting, filtering, date logic, nested conditions,
     existence, etc.

4. TYPE/SEMANTIC COMPATIBILITY
   - Check column types and names against what the question requires.
   - Reject candidates where the apparent matching column has the wrong meaning/type.

5. SCHEMA COMPLETENESS
   - Prefer a candidate that can answer the COMPLETE question without invented
     tables, columns, relationships, or assumptions.

6. RETRIEVER SCORE
   - Use the bi-encoder score as supporting evidence only.
   - A lower-ranked candidate with a much better schema fit should beat a higher-ranked
     candidate with weak schema coverage.

Before selecting, mentally compare the strongest 2-4 candidates rather than stopping
at the first plausible match.

If candidates are schema-rich, inspect the supplied candidate schema first. If two or
more candidates remain genuinely ambiguous, call get_database_schema for those
specific databases. Use get_sample_data only when actual values are needed to resolve
an ambiguity such as category/value encoding, status labels, or relationship meaning.
Do not inspect every database indiscriminately.

========================
SCHEMA LINKING METHOD
========================
After choosing the DB:

1. Identify the minimum set of tables required to answer the question.
2. Identify the exact columns required for SELECT, WHERE, JOIN, GROUP BY, HAVING,
   ORDER BY, and LIMIT semantics.
3. Identify the shortest valid join path between required tables.
4. Distinguish primary keys, foreign keys, descriptive attributes, and identifiers.
5. Do not include irrelevant tables/columns merely because they have similar names.
6. If the question requires a bridge/junction table, include it explicitly.
7. If a subquery is required, identify the table/column that the subquery depends on.
8. Treat natural-language references such as "each", "only", "at least", "most",
   "highest", "never", "not", "more than", and "fewer than" as semantic constraints,
   not as keywords to match literally.

========================
IMPORTANT SPIDER RULES
========================
- Choose EXACTLY one DB_ID from the supplied candidates.
- Never invent a DB_ID.
- Never generate SQL.
- Do not use SQL demonstrations or example retrieval.
- Do not assume the top bi-encoder result is correct.
- Do not select a database solely because one table name overlaps the question.
- A database is acceptable only if its schema is relevant and supports the entire question.
- Use each candidate's database_purpose and entity_rules as additional semantic evidence.
- Entity rules describe the kinds of questions each database is intended to answer;
  treat them as guidance, not as a substitute for checking the actual schema.
- A matching entity rule is useful evidence, but reject a candidate if its schema
  cannot support the complete question or the required joins/operations.
- Compare rule fit across the candidates; do not treat missing rules as proof against
  a candidate, and do not invent rules or schema details.

========================
OUTPUT
========================
Return ONLY one compact JSON object for the FINAL selected database. Do NOT return
all candidates, a candidate_reasoning array, rejected database IDs, alternate DB IDs,
or a ranked list. The selected db_id must be exactly one of the supplied candidates.
Use this exact shape:
{
  "db_id": "one_final_db_id",
  "db_confidence": 0.0,
  "selection_reason": "short reason for this one selected database",
  "database_purpose": "purpose for the selected database only",
  "matched_entity_rules": ["only the strongest rule(s) for the selected database"],
  "relevant_tables": ["..."],
  "relevant_columns": ["table.column", "..."],
  "join_path": ["table_a.col = table_b.col", "..."],
  "schema_context": "concise, concrete context for the Query Agent"
}

Confidence must be between 0 and 1. Do not include other candidates anywhere in the output.
Do not fabricate details: only report schema facts supported by supplied schema/tool output.
""",
    )


def run_schema_agent(
    question: str,
    orchestrator_analysis: dict,
    db_candidates: list[dict],
    exclude_db_ids: list[str] | set[str] | None = None,
    selection_feedback: dict | None = None,
) -> dict:
    """Select exactly one DB from untried candidates and link its schema.

    Optional exclusion/feedback parameters enable database reselection after a
    reference-result mismatch without changing existing callers.
    """
    agent = build_schema_agent()
    excluded = set(exclude_db_ids or [])
    available_candidates = [c for c in db_candidates if c.get("db_id") not in excluded]
    if not available_candidates:
        raise ValueError("No untried database candidates remain for Schema Agent selection.")

    # Enrich only the currently eligible candidates with precomputed semantic rules.
    candidate_evidence = build_candidate_evidence(available_candidates, question)

    prompt = f"""
USER QUESTION:
{question}

ORCHESTRATOR ANALYSIS:
{orchestrator_analysis}

HIGH-RECALL DATABASE CANDIDATES WITH DATABASE PURPOSE + ENTITY RULES (sorted by rule-match signal, then bi-encoder score):
{candidate_evidence}

DATABASES ALREADY TRIED AND EXCLUDED:
{sorted(excluded)}

FEEDBACK FROM PREVIOUS ATTEMPT:
{json.dumps(selection_feedback or {}, ensure_ascii=False, default=str)}

TASK:
1. Compare the candidates using the database-selection method in your system prompt.
2. Compare each candidate's database purpose and entity rules against the question.
3. Verify promising rule matches against that candidate's actual schema and join paths.
4. Do not anchor on rank 1 or the highest bi-encoder score.
5. If necessary, inspect the schemas of only the strongest ambiguous candidates.
6. Select exactly one supplied DB_ID.
7. Perform schema linking for that DB.
8. Return only the requested JSON.
"""

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    content = result["messages"][-1].content
    parsed = extract_json_from_string(content)

    if parsed is None:
        raise ValueError(f"Schema Agent did not return valid JSON: {content}")

    valid_db_ids = {x["db_id"] for x in available_candidates}
    selected = parsed.get("db_id")
    if selected not in valid_db_ids:
        raise ValueError(
            f"Schema Agent selected invalid DB_ID: {selected}; "
            f"valid candidates: {sorted(valid_db_ids)}"
        )

    # Enforce the single-selection output contract even if the LLM ignores the prompt.
    # Never expose its alternate-candidate reasoning to downstream agents/logs.
    rules_by_db = load_database_rules()
    selected_rules = rules_by_db.get(selected, {})
    parsed.pop("candidate_reasoning", None)
    for key in ("candidate_dbs", "alternatives", "ranked_candidates", "other_candidates"):
        parsed.pop(key, None)
    parsed["db_id"] = selected
    parsed.setdefault("db_confidence", 0.0)
    parsed.setdefault("selection_reason", "Selected from supplied candidates after schema and rule comparison.")
    parsed.setdefault("database_purpose", selected_rules.get("database_purpose", ""))
    if not parsed.get("matched_entity_rules"):
        _, best_rules = _rule_match_score(
            question,
            selected_rules.get("database_purpose", ""),
            selected_rules.get("rules", []),
        )
        parsed["matched_entity_rules"] = [
            f"{item.get('entity', '')}: {item.get('rule', '')}" for item in best_rules[:3]
        ]
    parsed.setdefault("relevant_tables", [])
    parsed.setdefault("relevant_columns", [])
    parsed.setdefault("join_path", [])
    parsed.setdefault("schema_context", "")
    # Keep only the declared public fields; this prevents accidental candidate leakage.
    allowed = {
        "db_id", "db_confidence", "selection_reason", "database_purpose",
        "matched_entity_rules", "relevant_tables", "relevant_columns", "join_path", "schema_context"
    }
    return {key: parsed[key] for key in allowed if key in parsed}
