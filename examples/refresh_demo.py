"""Re-run the demo audits and save them for instant loading in the app."""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leakverity.pipeline import Options, audit  # noqa: E402

EX = ROOT / "examples"
RUNS = (("leaky", "readmission_model.ipynb", {}),
        ("leaky_confirmed", "readmission_model.ipynb", {"known_leaks": ["followup_ed_visits_30d"]}),   # auditor confirms the proposal
        ("clean", "readmission_model_clean.ipynb", {}))
for name, nb, extra in RUNS:
    r = audit(nb, (EX / nb).read_bytes(), "hospital_readmissions.csv", (EX / "hospital_readmissions.csv").read_bytes(),
              Options(n_seeds=6, **extra), on_progress=lambda e, log: print(e["message"], flush=True))
    (EX / f"demo_result_{name}.json").write_text(json.dumps(r, indent=1))
    print(name, r["verdict"]["verdict"])
