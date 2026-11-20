"""Render the Claim Survival Certificate as a standalone, printable HTML page."""
from __future__ import annotations

import html
import json

VERDICT_COLORS = {"Supported": "#0B6E64", "Overstated": "#B0185E", "Contradicted": "#8A1020", "Insufficient evidence": "#5A6670",
                  "Depends on review": "#A86100"}
STATUS_TEXT = {"material": "Material", "real": "Real, size unclear", "small": "Small but real", "negligible": "Negligible", "uncertain": "Uncertain"}


def _e(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _f(x, d=3):
    if x is None:
        return "—"
    if abs(x) < 0.5 * 10 ** -d:
        x = 0.0
    return f"{x:.{d}f}"


def _title(s) -> str:
    import re
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", _e(s))


def gauge_svg(r: dict, width: int = 300, height: int = 380) -> str:
    """The drain gauge: reported value falling to the verified estimate, gap split by leak."""
    rp = r.get("replay") or {}
    if not rp.get("fidelity_ok"):
        return ""
    chance = (r.get("verdict") or {}).get("chance", 0.5)
    lo_axis, hi_axis = min(chance, (rp["corrected"]["lo"] or chance) - 0.02), 1.0
    top, bot, x0, bw = 24, height - 30, 70, 64

    def Y(v):
        return bot - (v - lo_axis) / (hi_axis - lo_axis) * (bot - top)
    rep = rp["reported"]
    cor = rp["corrected"]
    parts = []
    # axis
    parts.append(f'<line x1="{x0 - 8}" y1="{top}" x2="{x0 - 8}" y2="{bot}" stroke="#D3D9DC"/>')
    t = round(lo_axis + 0.05 - (lo_axis % 0.05), 2)
    while t <= hi_axis + 1e-9:
        parts.append(f'<line x1="{x0 - 12}" y1="{Y(t):.1f}" x2="{x0 - 8}" y2="{Y(t):.1f}" stroke="#9AA6AD"/>'
                     f'<text x="{x0 - 16}" y="{Y(t) + 4:.1f}" text-anchor="end" class="ax">{t:.2f}</text>')
        t = round(t + 0.05, 2)
    # verified band
    parts.append(f'<rect x="{x0}" y="{Y(cor["hi"]):.1f}" width="{bw}" height="{max(2, Y(cor["lo"]) - Y(cor["hi"])):.1f}" fill="#0B6E64" opacity="0.18"/>')
    parts.append(f'<line x1="{x0}" x2="{x0 + bw}" y1="{Y(cor["mean"]):.1f}" y2="{Y(cor["mean"]):.1f}" stroke="#0B6E64" stroke-width="3"/>')
    # gap segments
    segs = [w for w in (rp.get("waterfall") or []) if "delta" in w]
    cur = rep
    shades = ["#B0185E", "#C9457F", "#DB76A0", "#E9A6C1", "#F2CADB", "#F7E0EA"]
    k = 0
    for w in segs:
        d = -w["delta"]
        if d <= 0.0005:
            continue
        y1, y2 = Y(cur), Y(cur - d)
        parts.append(f'<rect x="{x0}" y="{y1:.1f}" width="{bw}" height="{max(1, y2 - y1):.1f}" fill="{shades[min(k, 5)]}"/>')
        if y2 - y1 > 13:
            parts.append(f'<text x="{x0 + bw + 8}" y="{(y1 + y2) / 2 + 4:.1f}" class="lb">−{d:.3f} {_e(w["label"].lower())}</text>')
        cur -= d
        k += 1
    parts.append(f'<line x1="{x0 - 4}" x2="{x0 + bw + 4}" y1="{Y(rep):.1f}" y2="{Y(rep):.1f}" stroke="#13203A" stroke-width="2"/>')
    parts.append(f'<text x="{x0 + bw + 8}" y="{Y(rep) - 6:.1f}" class="big">Reported {rep:.3f}</text>')
    parts.append(f'<text x="{x0 + bw + 8}" y="{Y(cor["mean"]) + 22:.1f}" class="big ok">Verified {cor["mean"]:.3f}</text>')
    parts.append(f'<text x="{x0 + bw + 8}" y="{Y(cor["mean"]) + 38:.1f}" class="ax">95% CI {cor["lo"]:.3f}–{cor["hi"]:.3f}</text>')
    if lo_axis <= chance:
        parts.append(f'<line x1="{x0 - 8}" x2="{x0 + bw + 150}" y1="{Y(chance):.1f}" y2="{Y(chance):.1f}" stroke="#9AA6AD" stroke-dasharray="3 3"/>'
                     f'<text x="{x0 + bw + 150}" y="{Y(chance) - 4:.1f}" text-anchor="end" class="ax">chance</text>')
    return (f'<svg viewBox="0 0 {width + 160} {height}" role="img" aria-label="Reported {rep:.3f} versus verified {cor["mean"]:.3f}">'
            '<style>.ax{font:11px Archivo,system-ui,sans-serif;fill:#5A6670}.lb{font:12px Archivo,system-ui,sans-serif;fill:#13203A}'
            '.big{font:600 15px Archivo,system-ui,sans-serif;fill:#13203A}.ok{fill:#0B6E64}</style>' + "".join(parts) + "</svg>")


def certificate_html(r: dict) -> str:
    v = r.get("verdict") or {}
    verdict = v.get("verdict", "Insufficient evidence")
    color = VERDICT_COLORS.get(verdict, "#5A6670")
    rp = r.get("replay") or {}
    proto = r.get("protocol") or {}
    claim = r.get("claim") or {}
    label = proto.get("metric_label", "metric")
    rows = []
    att = (r.get("replay") or {}).get("attribution") or {}
    review = [f for f in r.get("findings") or [] if f.get("needs_review")]
    for f in r.get("findings") or []:
        if f.get("needs_review"):
            continue
        imp = f.get("impact") or {}
        dag = " <span class='src'>(protocol change: different test rows)</span>" if (att.get(f.get("fix")) or {}).get("changes_test_population") else ""
        if f.get("proposed"):
            dag += " <span class='src'>(proposed: evaluated both ways)</span>"
        rows.append(f"<tr><td>{_title(f['title'])}{dag}<div class='src'>{_e(f.get('source', ''))}</div></td>"
                    f"<td class='n'>{_f(imp.get('mean'))}</td><td class='n'>{_f(imp.get('lo'))} to {_f(imp.get('hi'))}</td>"
                    f"<td class='n'>{_f(imp.get('last'))}</td><td><span class='st st-{_e(imp.get('status', 'na'))}'>{STATUS_TEXT.get(imp.get('status'), '—')}</span></td></tr>")
    surv = r.get("survival") or {}
    srows = []
    for key in ("stability", "subgroup", "temporal"):
        t = surv.get(key) or {}
        if not t:
            continue
        st = t.get("status", "not_applicable")
        if key == "stability":
            det = f"SD {_f(t.get('sd'))} across {t.get('runs')} re-splits (range {_f(t.get('min'))}–{_f(t.get('max'))})"
        elif key == "subgroup" and st != "not_applicable":
            det = "; ".join(f"{_e(g['level'])}: {_f(g.get('mean'))}" + (f" ({_f(g.get('lo'))}–{_f(g.get('hi'))})" if g.get('lo') is not None else " (inconclusive)")
                            for g in t.get("groups", []))
        elif key == "temporal" and st != "not_applicable":
            det = f"{_f(t.get('value'))} on records after {_e(t.get('cutoff'))} vs {_f(t.get('random_split'))} on random splits"
        else:
            det = _e(t.get("reason", ""))
        srows.append(f"<tr><td>{_e(t.get('name'))}</td><td>{det}</td><td><span class='st st-{st}'>{st.replace('_', ' ').capitalize()}</span></td></tr>")
    rep = r.get("repair") or {}
    ver = rep.get("verification") or {}
    inputs = r.get("inputs") or {}
    env = r.get("environment") or {}
    split = (proto.get("split") or {})
    gauge = gauge_svg(r)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claim Survival Certificate · {_e(r.get('id'))}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap" rel="stylesheet">
<style>
:root{{--ink:#13203A;--paper:#F4F6F5;--muted:#5A6670;--rule:#D3D9DC;--leak:#B0185E;--ok:#0B6E64;--warn:#A86100}}
*{{box-sizing:border-box}}body{{margin:0;background:#E9EDEC;color:var(--ink);font:15px/1.55 Archivo,system-ui,sans-serif}}
.sheet{{max-width:880px;margin:32px auto;background:#fff;padding:48px 56px;border:1px solid var(--rule)}}
h1{{font:600 15px Archivo,sans-serif;letter-spacing:.02em;margin:0;color:var(--muted)}}
.claim{{font:400 26px/1.35 'Source Serif 4',Georgia,serif;margin:14px 0 6px;max-width:34em}}
.meta{{color:var(--muted);font-size:13px}}
.stamp{{display:inline-block;margin-top:22px;border:2px solid {color};color:{color};padding:6px 16px;font:700 22px Archivo,sans-serif;transform:rotate(-2deg)}}
.reason{{font:400 17px/1.55 'Source Serif 4',Georgia,serif;margin:16px 0 0;max-width:40em}}
.grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);gap:32px;margin-top:32px;align-items:start}}
h2{{font:600 17px Archivo,sans-serif;margin:36px 0 10px}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}th{{text-align:left;font-weight:600;color:var(--muted);border-bottom:1px solid var(--ink);padding:6px 8px 6px 0}}
td{{border-bottom:1px solid var(--rule);padding:8px 8px 8px 0;vertical-align:top}}td.n{{font-variant-numeric:tabular-nums;white-space:nowrap}}
.src{{color:var(--muted);font-size:12px}}.st{{font-size:12px;font-weight:600;padding:2px 8px;border-radius:3px;white-space:nowrap}}
.st-material,.st-concern{{background:#F7E0EA;color:var(--leak)}}.st-negligible,.st-pass{{background:#DDEFEC;color:var(--ok)}}
.st-small{{background:#F6E8D2;color:var(--warn)}}.st-real{{background:#F7E0EA;color:var(--leak)}}.st-depends{{background:#F6E8D2;color:var(--warn)}}.st-uncertain,.st-not_applicable,.st-inconclusive,.st-na{{background:#ECEFF0;color:var(--muted)}}
.kv{{display:grid;grid-template-columns:auto 1fr;gap:4px 16px;font-size:12.5px;color:var(--muted)}}.kv b{{color:var(--ink);font-weight:500;word-break:break-all}}
.big3{{display:flex;gap:28px;margin-top:6px}}.big3 div{{font-size:13px;color:var(--muted)}}.big3 b{{display:block;font:600 30px Archivo,sans-serif;color:var(--ink);font-variant-numeric:tabular-nums}}
.big3 .ok b{{color:var(--ok)}}.big3 .lk b{{color:var(--leak)}}
.rules{{font-size:12.5px;color:var(--muted)}}.rules b{{color:var(--ink)}}
@media print{{body{{background:#fff}}.sheet{{margin:0;border:0;padding:24px}}}}
@media (max-width:720px){{.sheet{{padding:24px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main class="sheet">
<h1>LeakVerity AI · Claim Survival Certificate</h1>
<p class="claim">“{_e(claim.get('text') or f"Reported {label}: {_f(proto.get('reported'))}")}”</p>
<div class="meta">{_e(inputs.get('code'))} with {_e(inputs.get('data'))} · certificate {_e(r.get('id'))} · {_e(r.get('created'))}</div>
<div class="stamp">{_e(verdict)}{' (provisional)' if v.get('provisional') else ''}{' (conditional)' if v.get('conditional') else ''}</div>
<p class="reason">{_e(v.get('reason', ''))}</p>
{''.join(f"<p class='meta'><b>Reviewer decision:</b> <code>{_e(c)}</code> {_e(w)}.</p>" for k, w in (("known_leaks", "confirmed as not available at prediction time"), ("not_leaks", "marked as legitimate")) for c in ((r.get('options') or {}).get(k) or []))}
{''.join(f"<p class='meta'>Caveat ({_e(c['test'])}): {_e(c['text'])}{' This contradicts the claim as stated.' if c.get('contradicts_claim') else ''}</p>" for c in (v.get('caveats') or []))}
{(lambda sc: f"<table style='margin-top:14px'><tr><th>If the proposed leak is…</th><th>Verdict</th><th>Verified</th></tr><tr><td>confirmed (removed)</td><td>{_e(sc['confirmed']['verdict'])}</td><td class='n'>{_f(((r.get('replay') or {}).get('corrected') or {}).get('mean'))}</td></tr><tr><td>legitimate (kept)</td><td>{_e(sc['legitimate']['verdict'])}</td><td class='n'>{_f((((r.get('replay') or {}).get('legit_scenario')) or {}).get('corrected', {}).get('mean'))}</td></tr></table>")(v['scenarios']) if v.get('scenarios') else ''}
<div class="grid"><div>{gauge}</div><div>
<div class="big3"><div>Reported<b>{_f(rp.get('reported'))}</b></div><div class="ok">Verified<b>{_f((rp.get('corrected') or {}).get('mean'))}</b></div>
<div class="lk">Inflation<b>{_f((rp.get('inflation') or {}).get('from_reported'))}</b></div></div>
<p class="meta">Verified = mean {_e(label)} of the fully corrected protocol over {len(rp.get('seeds') or [])} random splits; 95% CI with the Nadeau–Bengio correction for overlapping splits.
Fidelity check: replaying the original protocol reproduced {_f(rp.get('fidelity_value'), 4)} (reported {_f(rp.get('reported'), 4)}).</p>
<p class="meta">Corrected code {'<b style="color:var(--ok)">verified by execution</b>' if ver.get('verified') else 'not verified'}{f": printed {_f(ver.get('value'), 6)}, replay engine {_f(ver.get('expected'), 6)}" if ver else ''}.</p>
</div></div>
<h2>Leaks found and what each one changed</h2>
<table><tr><th>Finding</th><th>Fair share</th><th>95% CI</th><th>Once others fixed</th><th>Effect</th></tr>{''.join(rows) or '<tr><td colspan=5>No leakage found.</td></tr>'}</table>
<p class="meta">Fair share = exact Shapley attribution of the change across every order of applying the fixes, so interacting leaks are not double-counted.
Rows marked "protocol change" alter which rows are in the test set: their share is a protocol-dependent change in performance under the corrected evaluation, not a proven causal effect of the leak.</p>
{('<h2>Features for the author to review</h2><p class="meta">Flagged by name only, with ordinary predictive power. Not removed, because removing a legitimate predictor would also lower the score: ' + ', '.join('<code>' + _e(f.get('column')) + '</code>' for f in review) + '.</p>') if review else ''}
<h2>Repair verification</h2>
<table><tr><th>Check on the executed corrected code</th><th>Result</th></tr>{''.join(f"<tr><td>{_e(c['check'])}<div class='src'>{_e(c['detail'])}</div></td><td><span class='st st-{'pass' if c['ok'] else 'concern'}'>{'Pass' if c['ok'] else 'Fail'}</span></td></tr>" for c in (ver.get('checks') or [])) or '<tr><td colspan=2>No repair was needed or generated.</td></tr>'}</table>
<h2>Claim survival tests (corrected protocol)</h2>
<table><tr><th>Test</th><th>Result</th><th>Status</th></tr>{''.join(srows)}</table>
<p class="meta">{_e(surv.get('limitations', ''))}</p>
<h2>Evidence record</h2>
<div class="kv"><span>Notebook SHA-256</span><b>{_e(inputs.get('code_sha256'))}</b><span>Data SHA-256</span><b>{_e(inputs.get('data_sha256'))}</b>
<span>Protocol</span><b>{_e(' → '.join(proto.get('flow') or []))}</b><span>Split</span><b>{_e(split.get('via', 'train_test_split'))}, test size {_e(split.get('test_size'))}, seed {_e(split.get('random_state'))}, seeds replayed {_e(rp.get('seeds'))}</b>
<span>Model</span><b>{_e(proto.get('model'))} · {_e(json.dumps(proto.get('model_params')))}</b><span>Environment</span><b>{_e(', '.join(f'{k} {v}' for k, v in env.items()))}</b>
<span>AI</span><b>{_e(('enabled: ' + str((r.get('ai') or {}).get('model'))) if (r.get('ai') or {}).get('enabled') else 'rule-based fallback (no API key)')}; AI never produces numbers</b></div>
<h2>Verdict rules (fixed before analysis)</h2>
<div class="rules">{''.join(f'<p><b>{_e(a)}.</b> {_e(b)}</p>' for a, b in (v.get('rules') or []))}</div>
</main></body></html>"""
