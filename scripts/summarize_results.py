"""Verify the stored 100-question evaluation without calling external APIs."""
import json
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "results" / "spider_nlq_evaluation_100.json"
data = json.loads(p.read_text(encoding="utf-8"))
n = len(data)
def percent(fn): return f"{100*sum(bool(fn(x)) for x in data)/n:.2f}%"
print(f"Examples: {n}")
print(f"End-to-end correct: {sum(x.get('end_to_end_execution_correct') is True for x in data)}/{n}")
print("End-to-end accuracy:", percent(lambda x: x.get('end_to_end_execution_correct')))
print("Top-1 retrieval accuracy:", percent(lambda x: x.get('top1_correct')))
print("Top-3 recall:", percent(lambda x: x.get('top3_recall')))
print("Top-5 recall:", percent(lambda x: x.get('top5_recall')))
print("Top-10 recall:", percent(lambda x: x.get('top10_recall')))
print("Selected DB accuracy:", percent(lambda x: x.get('selected_db')==x.get('gold_db_id')))
print("Mean retries:", round(sum(x.get('retries',0) for x in data)/n, 2))
print("Caveat: historical evaluation used gold reference feedback in the correction loop.")
