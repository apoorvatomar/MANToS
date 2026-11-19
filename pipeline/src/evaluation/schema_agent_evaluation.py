"""Evaluate Bi-Encoder Top-10 retrieval followed by Schema Agent selection."""
from src.db_prediction.model import DBIDPredictor
from src.agents.schema_agent import run_schema_agent
from src.agents.orchestrator import run_orchestrator

def evaluate_schema_agent(dev_data, max_questions=None, top_k=10):
    predictor = DBIDPredictor()
    data = dev_data[:max_questions] if max_questions else dev_data
    retrieved = correct = 0
    rows = []
    for i, item in enumerate(data, start=1):
        question, gold_db = item["question"], item["db_id"]
        candidates = predictor.predict(question, top_k=top_k)
        ids = [x["db_id"] for x in candidates]
        in_topk = gold_db in ids
        retrieved += int(in_topk)
        predicted = None
        error = None
        if in_topk:
            try:
                decision = run_schema_agent(question, run_orchestrator(question), candidates)
                predicted = decision["db_id"]
            except Exception as exc:
                error = str(exc)
        hit = int(predicted == gold_db)
        correct += hit
        rows.append({"question": question, "gold_db": gold_db, "predicted_db": predicted, "gold_in_top10": in_topk, "correct": hit, "error": error})
        print(f"{i}/{len(data)} | gold={gold_db} | predicted={predicted} | top10={in_topk} | correct={bool(hit)}")
    n = len(data)
    print("\n" + "=" * 70)
    print("BI-ENCODER TOP-10 -> SCHEMA AGENT")
    print("=" * 70)
    print(f"Bi-Encoder Top-{top_k} Recall : {retrieved/n:.4f}")
    print(f"Schema Agent DB Accuracy      : {correct/n:.4f}")
    return {"top_k_recall": retrieved/n, "schema_agent_accuracy": correct/n, "rows": rows}
