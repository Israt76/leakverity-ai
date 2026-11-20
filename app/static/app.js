"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f3 = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
// effects are reported as the change in the metric when a leak is fixed: a drop shows as −
const signed = (x, d = 3) => (x === null || x === undefined ? "—" : Math.abs(x) < 0.5 * 10 ** -d ? (0).toFixed(d) : (x > 0 ? "−" : "+") + Math.abs(x).toFixed(d));

const STAGES = [
  ["intake", "Reading the notebook and data"],
  ["reproduce", "Re-running the original experiment in the sandbox"],
  ["protocol", "Reconstructing the evaluation protocol"],
  ["investigate", "Investigating possible leaks"],
  ["fidelity", "Checking the replay reproduces the reported number"],
  ["replay", "Re-running the experiment with every combination of fixes"],
  ["repair", "Generating and executing the corrected code"],
  ["survival", "Running claim survival tests"],
  ["done", "Issuing the verdict"],
];
const FIX_LABEL = {
  preprocessing: "Preprocessing fitted before the split", resampling: "Resampling before the split",
  duplicates: "Duplicate rows across train and test", group_overlap: "Same person in train and test",
  future_features: "Feature unavailable at prediction time", test_selection: "Model chosen using the test set",
};

const STATUS = { inconclusive: "inconclusive", material: "material", real: "real, size unclear", small: "small but real", negligible: "negligible", uncertain: "uncertain" };
let JOB = null;
let R = null;

// ---------------------------------------------------------------- navigation
function show(tab) {
  $$(".tab").forEach((t) => (t.hidden = t.id !== "tab-" + tab));
  $$("#stages button").forEach((b) => b.toggleAttribute("aria-current", b.dataset.tab === tab) || b.removeAttribute("aria-current"));
  $$("#stages button").forEach((b) => { if (b.dataset.tab === tab) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
  $("#main").focus({ preventScroll: true });
  window.scrollTo({ top: 0 });
  if (tab !== "progress" && JOB) history.replaceState(null, "", `#${JOB}/${tab}`);
}
$$("#stages button").forEach((b) => b.addEventListener("click", () => !b.disabled && show(b.dataset.tab)));

function enableTabs(on) {
  $$("#stages button").forEach((b) => { if (b.dataset.tab !== "intake") b.disabled = !on; });
}

// ---------------------------------------------------------------- start audits
async function post(url, body) {
  const r = await fetch(url, { method: "POST", body });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `Request failed (${r.status})`);
  return j;
}

function formOptions(form) {
  const fd = new FormData(form);
  const o = {};
  for (const k of ["claim_text", "prediction_point", "group_col", "date_col", "subgroup_col", "api_key", "n_seeds", "known_leaks", "not_leaks"]) o[k] = fd.get(k) || "";
  return o;
}

$("#uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const err = $("#formError");
  err.textContent = "";
  const code = form.code.files[0], data = form.data.files[0];
  if (!code || !data) { err.textContent = "Choose both the notebook and the dataset."; return; }
  const body = new FormData();
  body.append("code", code);
  body.append("data", data);
  body.append("options", JSON.stringify(formOptions(form)));
  try {
    const { job } = await post("api/audit", body);
    track(job, code.name);
  } catch (ex) { err.textContent = ex.message; }
});
$$("#uploadForm input").forEach((i) => i.addEventListener("input", () => ($("#formError").textContent = "")));

$$("[data-demo]").forEach((b) => b.addEventListener("click", async () => {
  try {
    const r = await fetch(`api/demo/${b.dataset.demo}`);
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail);
    await load(j.job);
  } catch (ex) { alertBox(ex.message); }
}));
$$("[data-run]").forEach((b) => b.addEventListener("click", async () => {
  const body = new FormData();
  body.append("options", JSON.stringify({ n_seeds: 3 }));
  try {
    const { job } = await post(`api/demo/${b.dataset.run}/run`, body);
    track(job, "readmission_model.ipynb");
  } catch (ex) { alertBox(ex.message); }
}));

function alertBox(msg) {
  const e = $("#formError");
  e.textContent = msg;
  e.scrollIntoView({ block: "center" });
}

// ---------------------------------------------------------------- progress
function track(job, name) {
  JOB = job;
  enableTabs(false);
  $("#progName").textContent = name || "";
  show("progress");
  const tick = async () => {
    const r = await fetch(`api/jobs/${job}`);
    const j = await r.json();
    renderProgress(j);
    if (j.status === "done") return load(job);
    if (j.status === "error") { $("#progNote").textContent = "The audit stopped: " + j.error; return; }
    setTimeout(tick, 1000);
  };
  tick();
}

function renderProgress(j) {
  const log = j.log || [];
  const seen = new Map(log.map((l) => [l.stage, l]));
  const last = log[log.length - 1];
  let idx = STAGES.findIndex(([k]) => k === (last && last.stage));
  $("#progressList").innerHTML = STAGES.map(([k, label], i) => {
    const l = seen.get(k);
    const cls = i === idx && j.status !== "done" ? "active" : l ? "done" : "todo";
    if (!l && i > idx) return `<li class="todo" style="opacity:.45"><span class="dot" style="background:var(--rule)"></span><span>${esc(label)}</span><span></span></li>`;
    return `<li class="${cls}"><span class="dot"></span><span><b>${esc(label)}</b><br><span class="small">${esc(l ? l.message : "")}</span></span><span class="t">${l ? l.t.toFixed(0) + " s" : ""}</span></li>`;
  }).join("");
  const frac = idx < 0 ? 0 : (idx + (last && last.frac ? last.frac : 0.3)) / STAGES.length;
  $("#progBar").style.width = Math.min(100, frac * 100) + "%";
}

// ---------------------------------------------------------------- load results
async function load(job) {
  const r = await fetch(`api/jobs/${job}?full=1`);
  const j = await r.json();
  if (!r.ok || !j.result) { alertBox(j.detail || "This audit is not available."); return; }
  JOB = job;
  R = j.result;
  renderAll();
  enableTabs(true);
  const want = (location.hash.split("/")[1]) || "investigate";
  show(R.replay ? want : "investigate");
}

function claimHead() {
  const v = R.verdict || {};
  const cls = "v-" + (v.verdict || "Insufficient").split(" ")[0];
  const sc = v.scenarios;
  const scen = sc ? `<table class="t" style="max-width:560px;margin:-10px 0 24px"><tr><th>If the proposed leak (${esc((R.replay.proposed || []).join(", "))}) is…</th><th>Verdict</th><th>Verified ${esc((R.protocol || {}).metric_label)}</th></tr>
    <tr><td>confirmed: removed</td><td>${esc(sc.confirmed.verdict)}</td><td class="n">${f3(R.replay.corrected.mean)}</td></tr>
    <tr><td>legitimate: kept</td><td>${esc(sc.legitimate.verdict)}</td><td class="n">${f3(((R.replay.legit_scenario || {}).corrected || {}).mean)}</td></tr></table>` : "";
  const c = R.claim || {};
  const p = R.protocol || {};
  const text = c.text || (p.reported !== undefined ? `Reported ${p.metric_label}: ${f3(p.reported)}` : "No claim identified");
  return `<div class="claimhead"><div><q>${esc(text)}</q>
    <div class="meta">${esc(R.inputs.code)} with ${esc(R.inputs.data)} · ${R.ai && R.ai.enabled ? "AI reader: " + esc(R.ai.model) : "Rule-based reader (no AI key)"} · audit ${esc(R.id)}</div></div>
    <div class="stamp ${cls}">${esc(v.verdict || "")}${v.provisional ? " (provisional)" : ""}${v.conditional ? " (conditional)" : ""}</div></div>
    ${v.conditional ? `<div class="caveats"><b>Conditional verdict.</b> It assumes ${esc((v.conditional_on || []).join(", "))} ${(v.conditional_on || []).length === 1 ? "is" : "are"} legitimate. Resolve the review items on the Investigation tab.</div>` : ""}
    ${scen}
    ${v.provisional ? `<div class="caveats"><b>Provisional verdict.</b> ${esc(v.provisional_reason)}</div>` : ""}
    ${(v.caveats || []).length ? `<div class="caveats"><b>${v.verdict === "Supported" ? "Holds with caveats" : (sc && !sc.agree ? "Survival test findings (if the proposed leak is confirmed)" : "Survival test findings")}</b><ul>${v.caveats.map((c) => `<li>${esc(c.text)}${c.contradicts_claim ? " <span class=\"chip concern\">contradicts the claim's scope</span>" : ""}</li>`).join("")}</ul></div>` : ""}`;
}

function renderAll() {
  const v = R.verdict || {};
  const rv = $("#railVerdict");
  rv.hidden = false;
  rv.innerHTML = `<div>Verdict</div><div class="stamp v-${esc((v.verdict || "Insufficient").split(" ")[0])}">${esc(v.verdict)}</div>
    ${R.replay && R.replay.corrected ? `Reported ${f3(R.replay.reported)}<br>Verified ${f3(R.replay.corrected.mean)}${(R.verdict.scenarios && !R.verdict.scenarios.agree) ? " if confirmed" : ""}` : ""}`;
  $("#topRight").innerHTML = `<span>${esc(R.inputs.code)}</span><button class="link" id="newAudit">New audit</button>`;
  $("#newAudit").onclick = () => { history.replaceState(null, "", "./"); enableTabs(false); $("#railVerdict").hidden = true; $("#topRight").innerHTML = ""; show("intake"); };
  renderInvestigate();
  renderRepair();
  renderImpact();
  renderSurvival();
  renderCertificate();
}

// ---------------------------------------------------------------- code excerpt
function excerpt(lines, around = 2) {
  const code = (R.source && R.source.code || "").split("\n");
  const map = (R.source && R.source.line_map) || [];
  const set = new Set(lines.filter(Boolean));
  if (!set.size) return "";
  const lo = Math.max(1, Math.min(...set) - around), hi = Math.min(code.length, Math.max(...set) + around);
  let out = "";
  for (let i = lo; i <= hi; i++) {
    const m = map[i - 1];
    const label = m ? (R.source.kind === "notebook" ? `${m[0] + 1}:${m[1]}` : `${m[1]}`) : i;
    out += `<div class="${set.has(i) ? "hl" : ""}"><span class="ln">${label}</span>${esc(code[i - 1])}</div>`;
  }
  return `<div class="code" aria-label="Code excerpt">${out}</div>`;
}

// ---------------------------------------------------------------- 2. investigation
function renderInvestigate() {
  const el = $("#tab-investigate");
  const p = R.protocol || {};
  if (!p.supported) {
    el.innerHTML = claimHead() + `<h2>This experiment could not be replayed</h2><p class="note">${esc(p.reason || (R.verdict || {}).reason)}</p>
      <p class="small">LeakVerity abstains rather than guess. Supported today: pandas + scikit-learn notebooks with a hold-out split (train_test_split, ShuffleSplit or GroupShuffleSplit).</p>`;
    return;
  }
  const leakySteps = new Set((p.pre_split_steps || []).map((s) => `Fit ${s.class} (all rows)`));
  const flow = (p.flow || []).map((s, i) => (i ? `<li class="arrow" aria-hidden="true">→</li>` : "") +
    `<li class="${leakySteps.has(s) || /keep best/.test(s) ? "leaky" : ""}">${esc(s)}</li>`).join("");
  const finds = R.findings || [];
  const items = finds.map((f) => {
    const imp = f.impact || {};
    const eff = f.needs_review ? `<span class="chip uncertain">note</span><div class="k" style="margin-top:6px">Does not affect the verdict</div>`
      : f.proposed && imp.mean !== undefined ? `<div class="v">${signed(imp.mean)}</div><div class="k">if confirmed, fair share</div><span class="chip small">proposed: evaluated both ways</span>`
      : imp.mean !== undefined ? `<div class="v">${signed(imp.mean)}</div><div class="k">${esc(p.metric_label)} once fixed, fair share</div>
        <span class="chip ${esc(imp.status)}">${esc(STATUS[imp.status] || imp.status)}</span>` : `<div class="k">Not measured</div>`;
    return `<article class="finding"><div>
        <h3>${esc(f.title).replace(/`([^`]+)`/g, "<code>$1</code>")}</h3>
        <div class="src">${esc(f.rule)} · ${esc(f.source)}${f.confidence ? " · confidence " + esc(f.confidence) : ""}</div>
        <ul>${(f.evidence || []).map((e) => `<li>${esc(e).replace(/`([^`]+)`/g, "<code>$1</code>")}</li>`).join("")}</ul>
        <p class="expl">${esc(f.explanation)}</p>
        ${excerpt(f.lines || [])}
        ${f.proposed ? `<div class="toolbar" style="margin-top:12px"><button class="primary" data-decide="${esc(f.column)}" data-as="leak">Confirm: not available at prediction time</button>
          <button data-decide="${esc(f.column)}" data-as="legitimate">Mark as legitimate</button></div>` : ""}
      </div><div class="effect">${eff}</div></article>`;
  }).join("");
  const audit = R.feature_audit || {};
  el.innerHTML = claimHead() + `<h2>What the experiment actually did</h2>
    <p class="small">Recorded while the notebook ran. Highlighted steps can leak test information.</p>
    <ol class="flow">${flow}</ol>
    <h2>${finds.length ? `${finds.filter((f) => !f.needs_review).length} leak${finds.filter((f) => !f.needs_review).length === 1 ? "" : "s"} found` + (finds.some((f) => f.needs_review) ? `, ${finds.filter((f) => f.needs_review).length} feature${finds.filter((f) => f.needs_review).length === 1 ? "" : "s"} to review` : "") : "No leakage found"}</h2>
    ${finds.length ? items : `<p class="none-found">The split, preprocessing, model selection and features all passed. The replay still re-measures the result and runs the survival tests.</p>`}
    ${(R.options && ((R.options.known_leaks || []).length || (R.options.not_leaks || []).length)) ? `<p class="note">Your decisions in this audit: ${(R.options.known_leaks || []).map((c) => `<code>${esc(c)}</code> confirmed as a leak`).concat((R.options.not_leaks || []).map((c) => `<code>${esc(c)}</code> marked legitimate`)).join("; ")}.</p>` : ""}
    <p class="small">Features were checked against the prediction point by ${audit.source === "ai" ? "the AI reader" : "the rule-based reader"}${audit.prediction_point ? ` (prediction point: ${esc(audit.prediction_point)})` : ""}. Every flag is a hypothesis until the replay measures it.</p>`;
}

document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-decide]");
  if (!b) return;
  const body = new FormData();
  body.append("decisions", JSON.stringify({ [b.dataset.decide]: b.dataset.as }));
  b.disabled = true;
  try {
    const r = await fetch(`api/jobs/${encodeURIComponent(JOB)}/decide`, { method: "POST", body });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail);
    if (j.job.startsWith("demo-")) await load(j.job); else track(j.job, R.inputs.code);
  } catch (ex) { b.disabled = false; alert(ex.message); }
});

// ---------------------------------------------------------------- 3. repair
function diffView(diff) {
  return `<div class="code">${diff.split("\n").map((l) => {
    const cls = l.startsWith("+++") || l.startsWith("---") ? "hunk" : l.startsWith("@@") ? "hunk" : l.startsWith("+") ? "add" : l.startsWith("-") ? "del" : "";
    return `<div class="${cls}">${esc(l) || " "}</div>`;
  }).join("")}</div>`;
}

function renderRepair() {
  const el = $("#tab-repair");
  const rep = R.repair || {};
  if (!rep.ok) {
    el.innerHTML = claimHead() + `<h2>Repair</h2><p class="note">${esc(rep.reason || "No repair was generated.")}</p>`;
    return;
  }
  const v = rep.verification || {};
  const label = (R.protocol || {}).metric_label;
  el.innerHTML = claimHead() + `
    <div class="verify ${v.verified ? "" : "bad"}"><b>${v.verified ? "Repair verified by execution" : "Repair not verified"}</b>
      <span class="nums">The corrected code was executed and its own trace checked: ${(v.checks || []).filter((c) => c.ok).length} of ${(v.checks || []).length} checks passed. It printed ${esc(label)} ${f3(v.value, 6)}; the replay engine predicted ${f3(v.expected, 6)}.</span></div>
    <ul class="checks">${(v.checks || []).map((c) => `<li class="${c.ok ? "ok" : "bad"}"><b>${c.ok ? "Pass" : "Fail"}</b> ${esc(c.check)}<span class="small"> · ${esc(c.detail)}</span></li>`).join("")}</ul>
    <p class="note">LeakVerity keeps your data preparation (the first ${rep.kept_lines} lines) and rewrites only the evaluation protocol. Every change is marked <code># FIX</code>.</p>
    <div class="toolbar"><a class="btn primary" href="api/jobs/${encodeURIComponent(JOB)}/corrected.py">Download corrected code</a></div>
    <h2>Changes to your notebook</h2>${diffView(rep.diff || "")}`;
}

// ---------------------------------------------------------------- 4. impact
function ciRow(a, scale) {
  const w = 220, h = 26, pad = 6;
  const X = (v) => pad + (v - scale[0]) / (scale[1] - scale[0]) * (w - 2 * pad);
  const lo = a.lo ?? a.mean, hi = a.hi ?? a.mean;
  return `<svg class="ci" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">
    <line x1="${X(0)}" x2="${X(0)}" y1="2" y2="${h - 2}" stroke="#9AA6AD" stroke-dasharray="2 2"/>
    <line x1="${X(lo)}" x2="${X(hi)}" y1="${h / 2}" y2="${h / 2}" stroke="${a.status === "material" ? "#B0185E" : "#5A6670"}" stroke-width="2"/>
    <circle cx="${X(a.mean)}" cy="${h / 2}" r="4.5" fill="${a.status === "material" ? "#B0185E" : "#5A6670"}"/></svg>`;
}

function renderImpact() {
  const el = $("#tab-impact");
  const rp = R.replay;
  if (!rp || !rp.fidelity_ok) {
    el.innerHTML = claimHead() + `<h2>Impact</h2><p class="note">${esc((R.verdict || {}).reason || "No replay available.")}</p>`;
    return;
  }
  const label = R.protocol.metric_label;
  const att = Object.entries(rp.attribution || {}).sort((a, b) => b[1].mean - a[1].mean);
  const vals = att.flatMap(([, a]) => [a.lo ?? a.mean, a.hi ?? a.mean, 0]);
  const scale = [Math.min(...vals, -0.01) - 0.01, Math.max(...vals, 0.02) + 0.01];
  const rows = att.map(([k, a]) => `<tr><td>${esc(a.label)}${a.changes_test_population ? "<span class=\"small\"> (protocol change)</span>" : ""}<br><span class="chip ${esc(a.status)}">${esc(STATUS[a.status] || a.status)}</span></td>
      <td class="n">${signed(a.alone)}</td><td class="n">${signed(a.last)}</td>
      <td class="n">${signed(a.mean)}${ciRow(a, scale)}<span class="small">95% CI ${f3(a.lo)} to ${f3(a.hi)}</span></td></tr>`).join("");
  const seeds = rp.seeds.map((s, i) => `<tr><td class="n">${s}${i === 0 ? " (yours)" : ""}</td><td class="n">${f3(rp.original.values[i])}</td><td class="n">${f3(rp.corrected.values[i])}</td><td class="n">${f3(rp.original.values[i] - rp.corrected.values[i])}</td></tr>`).join("");
  el.innerHTML = claimHead() + `<div class="impact-grid">
      <div class="gauge">${R.gauge_svg || ""}</div>
      <div><h2>Where the ${f3(rp.inflation.from_reported)} of inflation came from</h2>
        <p class="note">Every fix was applied in every combination and the experiment re-run on ${rp.seeds.length} random splits (${rp.runs} runs, ${Math.round(rp.seconds)} s).
        <i>Alone</i> is the drop when only that fix is applied; <i>once others fixed</i> is the drop it still causes after every other leak is repaired; the <i>fair share</i> is the Shapley average over all orders, so shares add up to the total.</p>
        ${att.length ? `<table class="t"><tr><th>Leak</th><th>Alone</th><th>Once others fixed</th><th>Fair share</th></tr>${rows}</table>
          ${att.some(([, a]) => a.changes_test_population) ? `<p class="small">"Protocol change" fixes alter which rows are in the test set, so their share is a protocol-dependent change in performance, not a proven causal effect of the leak.</p>` : ""}` : `<p class="none-found">No leaks to attribute.</p>`}
      </div></div>
      <div class="seeds"><h2>Every split, before and after</h2>
        <p class="small">Fidelity check: replaying your protocol as written gave ${f3(rp.fidelity_value, 4)}; your notebook reported ${f3(rp.reported, 4)}.</p>
        <table class="t" style="max-width:640px"><tr><th>Split seed</th><th>${esc(label)} as written</th><th>${esc(label)} corrected</th><th>Inflation</th></tr>${seeds}</table></div>`;
}

// ---------------------------------------------------------------- 5. survival
function stripPlot(t, limit) {
  const w = 620, h = 90, pad = 30;
  const vals = t.values || [];
  const lo = Math.min(...vals) - 0.02, hi = Math.max(...vals) + 0.02;
  const X = (v) => pad + (v - lo) / (hi - lo) * (w - 2 * pad);
  const dots = vals.map((v, i) => `<circle cx="${X(v).toFixed(1)}" cy="${40 + ((i * 7) % 21) - 10}" r="5" fill="#13203A" opacity=".55"/>`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${vals.length} re-split results from ${f3(t.min)} to ${f3(t.max)}">
    <line x1="${pad}" x2="${w - pad}" y1="68" y2="68" stroke="#D3D9DC"/>
    <rect x="${X(t.lo)}" y="22" width="${Math.max(2, X(t.hi) - X(t.lo))}" height="36" fill="#0B6E64" opacity=".12"/>
    <line x1="${X(t.mean)}" x2="${X(t.mean)}" y1="18" y2="62" stroke="#0B6E64" stroke-width="2"/>
    ${dots}
    <text x="${pad}" y="84" font-size="11" fill="#5A6670">${f3(lo + 0.02)}</text><text x="${w - pad}" y="84" font-size="11" text-anchor="end" fill="#5A6670">${f3(hi - 0.02)}</text>
    <text x="${X(t.mean)}" y="13" font-size="11" text-anchor="middle" fill="#0B6E64">mean ${f3(t.mean)}</text></svg>`;
}

function barsPlot(groups, overall) {
  const w = 620, rowH = 34, pad = 130;
  const g = groups.filter((x) => x.mean !== undefined);
  const all = g.flatMap((x) => [x.lo, x.hi]).concat([overall]);
  const lo = Math.max(0.4, Math.min(...all) - 0.05), hi = Math.min(1, Math.max(...all) + 0.05);
  const X = (v) => pad + (v - lo) / (hi - lo) * (w - pad - 20);
  const h = rowH * groups.length + 30;
  const rows = groups.map((x, i) => {
    const y = 18 + i * rowH;
    if (x.mean === undefined) return `<text x="0" y="${y + 5}" font-size="13" fill="#13203A">${esc(x.level)}</text><text x="${pad}" y="${y + 5}" font-size="12" fill="#5A6670">inconclusive (too few cases)</text>`;
    const c = x.status === "concern" ? "#B0185E" : "#0B6E64";
    return `<text x="0" y="${y + 5}" font-size="13" fill="#13203A">${esc(x.level)} <tspan fill="#5A6670" font-size="11">${x.n}/split${x.unique ? ", " + x.unique + " unique" : ""}</tspan></text>
      <line x1="${X(x.lo)}" x2="${X(x.hi)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="3"/><circle cx="${X(x.mean)}" cy="${y}" r="6" fill="${c}"/>
      <text x="${X(x.hi) + 8}" y="${y + 4}" font-size="12" fill="${c}">${f3(x.mean)}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Subgroup performance">
    <line x1="${X(overall)}" x2="${X(overall)}" y1="4" y2="${h - 18}" stroke="#13203A" stroke-dasharray="3 3"/>
    <text x="${X(overall)}" y="${h - 4}" font-size="11" text-anchor="middle" fill="#13203A">overall ${f3(overall)}</text>${rows}</svg>`;
}

function renderSurvival() {
  const el = $("#tab-survival");
  const s = R.survival;
  if (!s) { el.innerHTML = claimHead() + `<h2>Survival tests</h2><p class="note">Not run for this audit.</p>`; return; }
  const label = R.protocol.metric_label;
  const block = (t, body) => `<div class="stest"><div><h3>${esc(t.name)}</h3><span class="chip ${esc(t.status)}">${esc((t.status || "").replace("_", " "))}</span>
      <div class="rule">${esc(t.rule || t.reason || "")}</div></div><div>${body}</div></div>`;
  const st = s.stability;
  const sg = s.subgroup;
  const tm = s.temporal;
  el.innerHTML = claimHead() + `<p class="note">These tests run on the fully corrected protocol, with seeds and thresholds fixed in advance. ${esc(s.limitations || "")}</p>
    <div class="survival">
    ${block(st, `${stripPlot(st)}<p class="small">${st.runs} new random splits. ${esc(st.selection || "")} Standard deviation ${f3(st.sd)}; range ${f3(st.min)} to ${f3(st.max)}.</p>`)}
    ${block(sg, sg.status === "not_applicable" ? `<p class="small">${esc(sg.reason)}</p>` :
      `${barsPlot(sg.groups, sg.overall)}<p class="small">${esc(label)} by <code>${esc(sg.column)}</code> on held-out patients, averaged over ${st.runs} re-splits (Nadeau–Bengio CIs; counts are per held-out split because the splits overlap).</p>`)}
    ${block(tm, tm.status === "not_applicable" ? `<p class="small">${esc(tm.reason)}</p>` :
      `<table class="t" style="max-width:520px"><tr><th>Evaluation</th><th>${esc(label)}</th></tr>
       <tr><td>Random splits (corrected)</td><td class="n">${f3(tm.random_split)}</td></tr>
       <tr><td>Train before ${esc(tm.cutoff)}, test after (new ${esc((R.roles || {}).group || "records")} only)</td><td class="n">${f3(tm.value)}</td></tr></table>
       <p class="small">${tm.n_train} training rows, ${tm.n_test} later rows from people not seen in training.</p>`)}
    </div>`;
}

// ---------------------------------------------------------------- 6. certificate
function renderCertificate() {
  const el = $("#tab-certificate");
  const j = encodeURIComponent(JOB);
  el.innerHTML = `<div class="toolbar">
      <a class="btn primary" href="api/jobs/${j}/certificate?download=1">Download certificate (HTML)</a>
      <a class="btn" href="api/jobs/${j}/result.json">Download evidence (JSON)</a>
      ${R.repair && R.repair.ok ? `<a class="btn" href="api/jobs/${j}/corrected.py">Download corrected code</a>` : ""}
      <a class="btn" href="api/jobs/${j}/certificate" target="_blank" rel="noopener">Open printable view</a></div>
    <iframe class="certframe" title="Claim Survival Certificate" src="api/jobs/${j}/certificate"></iframe>`;
  const fr = $(".certframe", el);
  fr.addEventListener("load", () => { try { fr.style.height = fr.contentDocument.documentElement.scrollHeight + 40 + "px"; } catch (e) { /* cross-origin */ } });
}

// ---------------------------------------------------------------- public-demo mode
fetch("api/config").then((r) => r.json()).then((c) => {
  if (c.uploads) return;
  const f = $("#uploadForm");
  f.querySelectorAll("input, button, select").forEach((el) => (el.disabled = true));
  const note = document.createElement("p");
  note.className = "note";
  note.innerHTML = "Uploads are off on this public demo because auditing runs your code. Clone the repository and run it locally to audit your own notebook. The demos on the right run live here.";
  f.insertBefore(note, f.querySelector("label"));
}).catch(() => {});

// ---------------------------------------------------------------- deep links
(function init() {
  const h = location.hash.slice(1);
  if (h) {
    const [job] = h.split("/");
    if (job) load(job).catch(() => {});
  }
})();
