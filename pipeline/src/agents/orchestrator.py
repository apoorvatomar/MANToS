from src.utils.llm_loader import get_llm
from src.utils.json_extractor import extract_json_from_string
from langchain.agents import create_agent


def build_orchestrator_agent():
    return create_agent(
        model=get_llm(),
        tools=[],
        system_prompt="""
You are the ORCHESTRATOR of a multi-agent Text-to-SQL system.

You do NOT generate SQL.

Analyze the user's NLQ and create a plan for downstream agents.

Determine:
- intent
- feasibility
- complexity: easy, medium, or hard
- whether joins are likely required
- whether aggregation/grouping is required
- whether a subquery/nested query is likely required
- a concise ordered plan for the Schema Agent and Query Agent

Return ONLY JSON:
{
  "intent": "...",
  "feasibility": "FEASIBLE or UNCERTAIN or NOT_FEASIBLE",
  "complexity": "easy|medium|hard",
  "requires_join": true,
  "requires_group_by": false,
  "requires_subquery": false,
  "plan": ["...", "..."]
}
"""
    )


def run_orchestrator(question: str) -> dict:
    agent = build_orchestrator_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    content = result["messages"][-1].content
    parsed = extract_json_from_string(content)

    if parsed is None:
        parsed = {
            "intent": "unknown",
            "feasibility": "UNCERTAIN",
            "complexity": "unknown",
            "requires_join": False,
            "requires_group_by": False,
            "requires_subquery": False,
            "plan": [content],
        }

    return parsed
