"""Run `python scripts/verify_release.py` to validate source bundle structure."""
import ast
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "pipeline" / "src").rglob("*.py")):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            # Remove notebook IPython shell/magic lines for Python syntax checks.
            src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%")))
            ast.parse(src, filename=f"{path.name}:cell{i}")
data = json.loads((ROOT / "results" / "spider_nlq_evaluation_100.json").read_text())
assert len(data) == 100 and sum(r.get("end_to_end_execution_correct") is True for r in data) == 91
print("OK: Python modules and notebook cells parse; 100 evaluation records with 91 correct.")
print("Note: Full model/API integration cannot be tested without external artifacts and API key.")
