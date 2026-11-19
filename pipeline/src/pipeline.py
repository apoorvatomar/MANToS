from rich import print

from src.agents.orchestrator import run_orchestrator
from src.agents.schema_agent import run_schema_agent
from src.agents.query_agent import run_query_agent
from src.agents.critique_agent import run_critique_agent
from src.db_prediction.model import DBIDPredictor
from src.tools.sql_validator import validate_sql
from src.tools.execute_sql import execute_sql_raw
from src.tools.execution_comparison import compare_gold_and_predicted_sql


def run_nlq_to_sql(
    question: str,
    max_retries: int = 3,
    top_k: int = 10,
    gold_db_id: str | None = None,
    gold_sql: str | None = None,
) -> dict:
    """NLQ-to-SQL with bounded SQL repair and optional Spider reference comparison.

    When both gold_db_id and gold_sql are provided, every valid generated query is
    compared against the reference execution. If results differ and the selected DB
    differs from the reference DB, the Schema Agent is called again with the failed
    DB excluded. If the selected DB already equals the reference DB, feedback is sent
    to the Query Agent to repair SQL rather than incorrectly blaming database choice.

    Existing callers remain compatible: gold_db_id and gold_sql are optional.
    max_retries counts repair/reselection attempts after the first generation.
    """
    has_reference = bool(gold_db_id and gold_sql)
    state = {
        "question": question,
        "db_id": None,
        "db_candidates": [],
        "db_id_prediction": None,
        "db_prediction_score": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "attempts": [],
        "status": "RUNNING",
        # Reference DB ID is intentionally kept out of the returned/public state.
        "reference_comparison_enabled": has_reference,
        "attempted_db_ids": [],
    }

    print("\n[bold cyan]1. ORCHESTRATOR[/bold cyan]")
    orchestrator = run_orchestrator(question)
    state["orchestrator_analysis"] = orchestrator
    state.update({
        "intent": orchestrator.get("intent"),
        "feasibility": orchestrator.get("feasibility"),
        "complexity": orchestrator.get("complexity"),
        "plan": orchestrator.get("plan", []),
    })
    print(orchestrator)
    if orchestrator.get("feasibility") == "NOT_FEASIBLE":
        state["status"] = "NOT_FEASIBLE"
        return state

    print("\n[bold cyan]2. BI-ENCODER — TOP DATABASE RETRIEVAL[/bold cyan]")
    predictor = DBIDPredictor()
    candidates = predictor.predict(question, top_k=top_k)
    state["db_candidates"] = candidates
    if not candidates:
        state["status"] = "NO_DB_CANDIDATES"
        return state
    state["db_id_prediction"] = candidates[0]["db_id"]
    state["db_prediction_score"] = candidates[0]["bi_score"]
    for c in candidates:
        print(f"{c['rank']:2}. {c['db_id']:30} score={c['bi_score']:.6f}")

    attempted_db_ids: set[str] = set()
    schema_feedback = None
    schema = run_schema_agent(
        question=question,
        orchestrator_analysis=orchestrator,
        db_candidates=candidates,
    )
    previous_sql = None
    previous_critique = None
    last_sql = None

    # One initial attempt plus up to max_retries repairs/reselections.
    for attempt_no in range(max_retries + 1):
        state["retry_count"] = attempt_no
        db_id = schema["db_id"]
        state["db_id"] = db_id
        state["schema_context"] = schema
        state["relevant_tables"] = schema.get("relevant_tables", [])
        state["relevant_columns"] = schema.get("relevant_columns", [])
        state["join_path"] = schema.get("join_path", [])
        attempted_db_ids.add(db_id)
        state["attempted_db_ids"] = sorted(attempted_db_ids)

        print(f"\n[bold cyan]3. SCHEMA AGENT — FINAL DB SELECTION (ATTEMPT {attempt_no + 1})[/bold cyan]")
        print(schema)
        print(f"\n[bold cyan]4. QUERY AGENT — ATTEMPT {attempt_no + 1}[/bold cyan]")
        sql = run_query_agent(
            question=question,
            db_id=db_id,
            orchestrator_analysis=orchestrator,
            schema_context=schema,
            previous_sql=previous_sql,
            critique=previous_critique,
        )
        last_sql = sql
        state["sql_candidate"] = sql
        print(sql)

        print("\n[bold cyan]5. VALIDATION + EXECUTION[/bold cyan]")
        validation = validate_sql(db_id, sql)
        execution = execute_sql_raw(db_id, sql) if validation["status"] == "VALID" else {
            "status": "error",
            "error": validation.get("error", "SQL validation failed"),
        }
        state["validation"] = validation
        state["execution"] = execution
        print("Validation:", validation)
        print("Execution:", execution)

        result_comparison = None
        if has_reference:
            result_comparison = compare_gold_and_predicted_sql(
                gold_db_id=gold_db_id,
                gold_sql=gold_sql,
                predicted_db_id=db_id,
                predicted_sql=sql,
            )
            state["result_comparison"] = result_comparison
            print({
                "database_match": result_comparison.get("database_match"),
                "predicted_db_id": result_comparison.get("predicted_db_id"),
                "reference_execution": result_comparison.get("gold_execution"),
                "predicted_execution": result_comparison.get("predicted_execution"),
                "comparison": result_comparison.get("comparison"),
            })

        print("\n[bold cyan]6. CRITIQUE AGENT[/bold cyan]")
        critique = run_critique_agent(
            question=question,
            db_id=db_id,
            schema_context=schema,
            sql=sql,
            validation=validation,
            execution=execution,
            result_comparison=result_comparison,
        )
        print(critique)
        state["critique"] = critique
        state["attempts"].append({
            "db_id": db_id,
            "sql": sql,
            "validation": validation,
            "execution": execution,
            "result_comparison": result_comparison,
            "critique": critique,
        })

        # A supplied reference comparison is a deterministic pass/fail gate.
        comparison_match = (
            result_comparison is None
            or result_comparison.get("comparison", {}).get("match") is True
        )
        if str(critique.get("status", "")).upper() == "PASS" and comparison_match:
            state["final_sql"] = sql
            state["status"] = "PASS"
            return state

        previous_sql = sql
        previous_critique = critique
        if attempt_no >= max_retries:
            break

        retry_action = critique.get("retry_action", "NONE")
        comparison_mismatch = bool(
            result_comparison is not None
            and result_comparison.get("comparison", {}).get("match") is False
        )
        wrong_database = has_reference and db_id != gold_db_id

        # If reference results disagree and selected DB differs from gold DB, treat
        # this as a database-selection failure and force a fresh, untried DB choice.
        if comparison_mismatch and wrong_database:
            remaining = [c for c in candidates if c.get("db_id") not in attempted_db_ids]
            if not remaining:
                print("\n[yellow]No untried DB candidates remain; stopping reselection.[/yellow]")
                break
            feedback = critique.get("feedback_to_schema_agent") or {
                "action": "RECONSIDER_DATABASE_SELECTION",
                "reason": result_comparison.get("comparison", {}).get("reason", "Reference result mismatch."),
                "exclude_current_db": True,
            }
            feedback = {
                **feedback,
                "previous_db_id": db_id,
                "previous_sql": sql,
                "reference_comparison": result_comparison.get("comparison", {}),
                "instruction": "Select a different untried candidate whose schema and entity rules best support the NLQ. Return one db_id only.",
            }
            print("\n[yellow]FAIL → Schema Agent database reselection[/yellow]")
            schema = run_schema_agent(
                question=question,
                orchestrator_analysis=orchestrator,
                db_candidates=candidates,
                exclude_db_ids=attempted_db_ids,
                selection_feedback=feedback,
            )
            # New DB means regenerate SQL from a clean attempt against its schema.
            previous_sql = None
            previous_critique = {
                "status": "FAIL",
                "reason": "Previous database produced results different from the reference. Rebuild SQL for the newly selected database.",
                "failure_type": "DATABASE_SELECTION_MISMATCH",
            }
            continue

        # If the selected DB equals the reference DB, mismatch is treated as a SQL
        # semantics issue: repair SQL on the same DB rather than excluding a correct DB.
        print(f"\n[yellow]FAIL → Query Agent repair ({retry_action})[/yellow]")

    state["final_sql"] = last_sql
    state["status"] = "FAILED_MAX_RETRIES"
    return state
