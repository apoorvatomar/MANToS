# Troubleshooting and FAQ

**`Missing inputs: DB model directory`** — The GitHub ZIP contains code, not trained weights. Supply the SentenceTransformer model directory for `DB_BACKEND="biencoder"`, or run Notebook 01 and unzip its `model/` + `metadata.json` into `ROBERTA_MODEL_PATH` for `DB_BACKEND="roberta"`.

**`GITHUB_REPO_URL` is empty or clone fails** — Publish this ZIP's contents as a public GitHub repository, paste its real HTTPS `.git` URL in the notebook, and confirm `GITHUB_BRANCH` (usually `main`). Before publication, set `PROJECT_SOURCE="drive_zip"` and use the original agent ZIP, but this fallback does not include the new BGE bi-encoder adapter.

**`Missing Spider tables.json / dev.json / database`** — Download the Spider dataset separately. Set `SPIDER_ROOT` to the folder directly containing `tables.json`, `dev.json`, and `database/`. SQLite files should appear as `database/<db_id>/<db_id>.sqlite`.

**`OPENAI_API_KEY is not set`** — In Colab's left sidebar select Secrets (key icon), add `OPENAI_API_KEY`, and enable notebook access. The supplied agent code uses OpenAI `gpt-4o-mini`, **not Groq**. Do not paste keys in GitHub.

**GPU not available** — In Colab: Runtime → Change runtime type → T4 GPU. Inference may run on CPU more slowly. Training BGE bi-encoder strongly benefits from a GPU.

**Out of memory while training** — Reduce `PER_DEVICE_BATCH_SIZE` to 4, lower `MAX_LENGTH` to 128, or increase gradient accumulation in Notebook 01.

**`ModuleNotFoundError: src`** — Run the source-load cell before importing pipeline modules; `ACTUAL_PROJECT_DIR` must contain `src/` and `db_rules.json`.

**`BGE bi-encoder artifacts must contain model/ and metadata.json`** — Do not point `ROBERTA_MODEL_PATH` at the `.zip` itself. Extract it so both are directly inside the directory. `MERGE_MAP` should be empty for direct DB-ID mapping.

**Why doesn't BGE bi-encoder use the 91% result?** — The saved 100-question run was produced with the existing SentenceTransformer bi-encoder pipeline and reference-guided correction. The BGE bi-encoder notebook trains a different classifier; the optional adapter is new.

**Why does the notebook score differ from 91%?** — The default release uses `USE_GOLD_FEEDBACK=False` for unguided inference, and only 5 evaluation questions to limit cost. The historical 91% was measured on 100 records with reference feedback and a specific external checkpoint. It is not a reproducibility guarantee.

**`MAX_EVAL_EXAMPLES=None` runs slowly / API costs** — Use 5 for demo, 100 only when you intend to pay for the API calls, and `None` only for the full dev set. `MAX_RETRIES` increases potential LLM calls per question.

**Why are there two rule sources?** — Notebook 01 uses schema-derived terms for BGE bi-encoder augmentation, stored in its model metadata. `pipeline/db_rules.json` is precomputed purpose/entity evidence for the Schema Agent. They are different artifacts.

**Can I run the entire project offline?** — Not as supplied. The inference agents call OpenAI and need API access. BGE bi-encoder training needs the base model weights downloaded once, and the Spider dataset is external.
