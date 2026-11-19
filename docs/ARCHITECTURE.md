# MANToS architecture

See the full walkthrough and Mermaid flowchart in the repository README. The runtime implementation is in `pipeline/src/pipeline.py`. It invokes the Orchestrator, BGE bi-encoder DB-ID predictor, Schema Agent, Query Agent, SQL validation/execution tools, and Critique Agent. The retry controller performs bounded repair/reselection attempts.

The DB-ID predictor is a Sentence-Transformer bi-encoder (`BAAI/bge-base-en-v1.5`), trained by `notebooks/01_DB_ID_BiEncoder_Training.ipynb`. The inference notebook loads that model and invokes the complete pipeline. The DB-ID model is not itself an LLM agent.
