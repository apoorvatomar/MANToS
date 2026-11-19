from src.pipeline import run_nlq_to_sql

if __name__ == "__main__":
    question = input("Enter your natural-language question: ").strip()
    result = run_nlq_to_sql(question=question, top_k=10, max_retries=3)
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print("Bi-Encoder Top-1 candidate:", result.get("db_id_prediction"))
    print("Bi-Encoder Top-1 score:", result.get("db_prediction_score"))
    print("Schema Agent selected DB:", result.get("db_id"))
    print("Schema Agent confidence:", result.get("schema_context", {}).get("db_confidence"))
    print("Status:", result.get("status"))
    print("Retries:", result.get("retry_count"))
    print("\nFinal SQL:")
    print(result.get("final_sql"))
