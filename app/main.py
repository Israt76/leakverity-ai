"""LeakVerity web server: upload → audit (background job) → results and certificate."""
from __future__ import annotations

import json
import os
import pathlib
import threading
import time
import secrets

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from leakverity.pipeline import Options, audit
from leakverity.report import certificate_html, gauge_svg

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
STATIC = pathlib.Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 25 * 2**20
# Arbitrary uploads execute user code. Public deployments should keep this off unless
# LEAKVERITY_SANDBOX=docker is configured (see docs/security.md); demos always work.
ALLOW_UPLOADS = os.getenv("LEAKVERITY_ALLOW_UPLOADS", "0") == "1"      # off unless explicitly enabled
JOB_TTL_SECONDS = int(os.getenv("LEAKVERITY_JOB_TTL", str(2 * 3600)))    # uploaded code and results are kept 2 h at most
MAX_JOBS = int(os.getenv("LEAKVERITY_MAX_JOBS", "50"))

app = FastAPI(title="LeakVerity", version="1.0.0")
JOBS: dict[str, dict] = {}
RUN_LOCK = threading.Semaphore(int(os.getenv("LEAKVERITY_CONCURRENCY", "1")))

DEMOS = {
    "leaky": {"code": "readmission_model.ipynb", "data": "hospital_readmissions.csv", "cached": "demo_result_leaky.json"},
    "clean": {"code": "readmission_model_clean.ipynb", "data": "hospital_readmissions.csv", "cached": "demo_result_clean.json"},
    "leaky_confirmed": {"code": "readmission_model.ipynb", "data": "hospital_readmissions.csv",
                        "cached": "demo_result_leaky_confirmed.json"},
}


def _purge():
    """Drop expired jobs (and the uploaded code they hold) and cap how many are kept in memory."""
    now = time.time()
    for k in [k for k, j in JOBS.items() if not k.startswith("demo-") and now - j["created"] > JOB_TTL_SECONDS]:
        JOBS.pop(k, None)
    live = sorted((k for k in JOBS if not k.startswith("demo-")), key=lambda k: JOBS[k]["created"])
    for k in live[:-MAX_JOBS] if len(live) > MAX_JOBS else []:
        JOBS.pop(k, None)


def _start(code_name, code_bytes, data_name, data_bytes, opts: Options) -> str:
    _purge()
    # the job id is an unguessable 128-bit capability: only whoever started the audit can read it
    jid = secrets.token_urlsafe(16)
    JOBS[jid] = {"id": jid, "status": "queued", "log": [], "result": None, "error": None, "created": time.time(),
                 "inputs": {"code": code_name, "data": data_name},
                 "_files": (code_name, code_bytes, data_name, data_bytes), "_opts": opts}

    def on_progress(entry, log):
        job = JOBS.get(jid)
        if job is not None:
            job["log"] = list(log)

    def work():
        with RUN_LOCK:
            if jid not in JOBS:
                return
            JOBS[jid]["status"] = "running"
            try:
                result = audit(code_name, code_bytes, data_name, data_bytes, opts, on_progress)
                if jid in JOBS:
                    JOBS[jid]["result"] = result
                    JOBS[jid]["status"] = "done"
            except Exception as e:  # noqa: BLE001
                if jid in JOBS:
                    JOBS[jid]["status"] = "error"
                    JOBS[jid]["error"] = f"{type(e).__name__}: {e}"
    threading.Thread(target=work, daemon=True).start()
    return jid


def _opts(form: dict) -> Options:
    o = Options()
    for k in ("target", "group_col", "date_col", "subgroup_col", "claim_text", "prediction_point"):
        v = (form.get(k) or "").strip()
        setattr(o, k, v or None)
    try:
        o.n_seeds = max(2, min(10, int(form.get("n_seeds") or 3)))
    except ValueError:
        o.n_seeds = 3
    for k in ("known_leaks", "not_leaks"):
        v = form.get(k) or ""
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        setattr(o, k, v or None)
    return o


@app.post("/api/audit")
async def start_audit(code: UploadFile = File(...), data: UploadFile = File(...), options: str = Form("{}")):
    if not ALLOW_UPLOADS:
        raise HTTPException(403, "Uploads are turned off on this public demo because they run your code. "
                                 "Run LeakVerity locally (see the README) to audit your own notebook, or try the demos here.")
    cb, db = await code.read(), await data.read()
    if len(cb) > MAX_UPLOAD or len(db) > MAX_UPLOAD:
        raise HTTPException(413, "Files must be under 25 MB.")
    if not code.filename.lower().endswith((".ipynb", ".py")):
        raise HTTPException(400, "Upload a Jupyter notebook (.ipynb) or Python script (.py).")
    if not data.filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(400, "Upload the dataset as a CSV file.")
    return {"job": _start(code.filename, cb, data.filename, db, _opts(json.loads(options or "{}")))}


@app.post("/api/demo/{name}/run")
def run_demo(name: str, options: str = Form("{}")):
    d = DEMOS.get(name) or {}
    if not d:
        raise HTTPException(404, "Unknown demo.")
    return {"job": _start(d["code"], (EXAMPLES / d["code"]).read_bytes(), d["data"], (EXAMPLES / d["data"]).read_bytes(),
                          _opts(json.loads(options or "{}")))}


@app.get("/api/demo/{name}")
def cached_demo(name: str):
    d = DEMOS.get(name) or {}
    p = EXAMPLES / d.get("cached", "_none_")
    if not p.exists():
        raise HTTPException(404, "No saved audit for this demo yet. Run it live instead.")
    res = json.loads(p.read_text())
    jid = "demo-" + name
    JOBS[jid] = {"id": jid, "status": "done", "log": res.get("log", []), "result": res, "error": None, "created": time.time(),
                 "inputs": {"code": d["code"], "data": d["data"]}}
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str, full: int = 0):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Audit not found. It may have expired when the server restarted.")
    out = {k: v for k, v in j.items() if k != "result" and not k.startswith("_")}
    if full and j["result"] is not None:
        out["result"] = dict(j["result"])
        out["result"]["gauge_svg"] = gauge_svg(j["result"])
    return JSONResponse(out)


def _result(jid):
    j = JOBS.get(jid)
    if not j or j.get("result") is None:
        raise HTTPException(404, "Audit not finished or not found.")
    return j["result"]


@app.get("/api/jobs/{jid}/certificate", response_class=HTMLResponse)
def certificate(jid: str, download: int = 0):
    r = _result(jid)
    headers = {"Content-Disposition": f'attachment; filename="leakverity-certificate-{r["id"]}.html"'} if download else {}
    return HTMLResponse(certificate_html(r), headers=headers)


@app.get("/api/jobs/{jid}/result.json")
def result_json(jid: str):
    r = _result(jid)
    return JSONResponse(r, headers={"Content-Disposition": f'attachment; filename="leakverity-{r["id"]}.json"'})


@app.get("/api/jobs/{jid}/corrected.py")
def corrected(jid: str):
    r = _result(jid)
    rep = r.get("repair") or {}
    if not rep.get("ok"):
        raise HTTPException(404, rep.get("reason", "No corrected code was generated."))
    return PlainTextResponse(rep["code"], headers={"Content-Disposition": f'attachment; filename="corrected_{r["inputs"]["code"].rsplit(".", 1)[0]}.py"'})


@app.get("/api/examples/{fname}")
def example_file(fname: str):
    allowed = {d["code"] for d in DEMOS.values()} | {d["data"] for d in DEMOS.values()}
    if fname not in allowed:
        raise HTTPException(404, "Not found.")
    return FileResponse(EXAMPLES / fname, filename=fname)


@app.post("/api/jobs/{jid}/decide")
def decide(jid: str, decisions: str = Form("{}")):
    """Re-run an audit with the user's decisions on proposed leaks ({"column": "leak" | "legitimate"})."""
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Audit not found.")
    d = json.loads(decisions or "{}")
    leaks = [c for c, v in d.items() if v == "leak"]
    legit = [c for c, v in d.items() if v == "legitimate"]
    if jid.startswith("demo-"):
        if jid == "demo-leaky" and leaks and not legit:
            return cached_demo("leaky_confirmed")
        raise HTTPException(400, "For the saved demo, run it live to try other decisions.")
    if "_files" not in j:
        raise HTTPException(410, "The uploaded files for this audit have expired. Start a new audit.")
    import copy
    opts = copy.copy(j["_opts"])
    opts.known_leaks = sorted(set((opts.known_leaks or []) + leaks))
    opts.not_leaks = sorted(set((opts.not_leaks or []) + legit))
    return {"job": _start(*j["_files"], opts)}


@app.get("/api/config")
def config():
    return {"uploads": ALLOW_UPLOADS, "sandbox": os.getenv("LEAKVERITY_SANDBOX", "process")}


@app.get("/api/health")
def health():
    return {"ok": True}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
