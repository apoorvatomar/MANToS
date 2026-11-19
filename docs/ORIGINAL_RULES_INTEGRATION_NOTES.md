# NL2SQL pipeline with DB-ID entity-rule evidence

## What changed
- The existing bi-encoder still retrieves the top-k candidate databases.
- The Schema Agent now receives `database_purpose` and `rules` for each retrieved candidate from `db_rules.json`.
- It is instructed to compare rule fit and verify matches against actual schema, columns, relationships, and operations.
- Query generation, validation, execution, and critique stages are unchanged.
- No additional Groq calls or model training are required by this integration; the rules are loaded locally and included in the existing Schema Agent request.

## Setup
1. Unzip the project.
2. Keep `db_rules.json` in the project root (same level as `src/`).
3. Ensure your existing model and Spider schema files are configured as before.
4. Run the pipeline as usual.

Optional: set `DB_RULES_PATH` to the absolute path of your rules JSON if it is stored elsewhere.

## Important
The rules file contains 166 database entries. Only rules for the predictor's retrieved top-k candidates are sent to the Schema Agent. Rules are supporting semantic evidence, not ground truth; the Schema Agent must still validate against the actual candidate schema. Evaluate on a held-out set to establish whether selection accuracy improves.


## Reference execution comparison and database reselection

`run_nlq_to_sql` now accepts two optional keyword arguments:

```python
run_nlq_to_sql(
    question=question,
    gold_db_id=gold_db_id,
    gold_sql=gold_sql,
    max_retries=3,
    top_k=10,
)
```

When both reference values are provided, the pipeline executes gold SQL on the gold
SQLite database and predicted SQL on the currently selected database. It compares the
full result sets (not only the first 50 rows), preserves duplicate rows, ignores row
order unless the gold SQL has `ORDER BY`, and normalizes numeric values to six decimal
places. If the results differ and the selected DB differs from the reference DB, the
Schema Agent is called again with the failed DB excluded. If the DB already matches the
reference, feedback is sent to the Query Agent to repair SQL instead of blaming DB
selection. `max_retries` bounds all retries/reselections. Existing callers that do not
pass gold values continue to work without reference comparisons.

The gold values must be supplied by the Spider evaluation loop; the pipeline does not
try to infer or retrieve a gold answer automatically.


## OpenAI model configuration

The LLM loader uses `langchain_openai.ChatOpenAI` with model `gpt-4o-mini`. Install
`langchain-openai` in the environment and set `OPENAI_API_KEY` (for example, in a
`.env` file or your Colab secrets). The Groq client and `GROQ_API_KEY` are no longer
used by the LLM loader. Other pipeline behavior is unchanged.


## Reference DB ID privacy in runtime output

The reference DB ID is still accepted internally by the evaluation entry point and used to execute the reference SQL and decide whether to repair SQL or reselect a database. It is intentionally omitted from the returned pipeline state, execution-comparison payload, critique-agent input, and printed comparison trace. The Critique Agent receives only a boolean `database_match` signal plus reference/predicted execution evidence.
