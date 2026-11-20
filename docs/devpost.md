# Devpost submission draft — LeakVerity AI

**Tagline:** Find the leak. Measure the damage. Test the claim.

**Track:** Scientific Research

## Inspiration

Machine-learning results are published and deployed on the strength of a single number. A model that reports AUC 0.94 gets a paper, a press release, sometimes a hospital deployment. But data leakage, where information from the test set or the future slips into training, is pervasive. A study of more than 100,000 public notebooks found it everywhere. Tools that flag leakage patterns already exist. What they can't tell you is whether the result is still true once the leak is fixed. A 2026 study of 2,047 datasets showed that leakage types differ in impact by orders of magnitude, and the most famous one (fitting a scaler before the split) barely matters. Researchers need to know which leaks moved their result, and by how much.

## What it does

You upload a notebook and its dataset. LeakVerity:

1. **Reproduces** the experiment in a sandbox and traces every split, fit and metric.
2. **Investigates** leaks using execution evidence. For example, it can report that "60 of 682 test rows are exact copies of training rows" or that "332 patients are on both sides of the split". An AI reading of the features proposes anything that looks recorded after the prediction point. A proposal is evaluated both ways, and only you can confirm it.
3. **Repairs** the notebook, keeping your data preparation, and **executes** the corrected code to verify it.
4. **Measures** the damage. It first proves its reconstruction reproduces your exact number (otherwise it abstains). Then it re-runs every combination of fixes over several splits and attributes the inflation to each leak with exact Shapley values and corrected confidence intervals.
5. **Tests the claim** for stability, subgroup performance and time-based validity.
6. **Certifies**: pre-registered rules issue *Supported / Overstated / Contradicted / Insufficient evidence*, or *Depends on review* when the verdict hinges on a proposal you must confirm, with input hashes and every number traceable.

On the demo notebook the verdict first reads *Depends on review*: the claim fails if the follow-up feature is a leak, and holds if it is not. Once the feature is confirmed, the claim of AUC 0.94 becomes a verified 0.709, and the true value on fresh patients is 0.713. The certificate shows the inflation came almost entirely from one feature recorded after discharge, that the textbook scaler leak changed nothing, and that the model fails patients aged 65 and over, so "all adult patients" is not supported.

## How we built it

Python, pandas, scikit-learn, SciPy, FastAPI, and a hand-built web interface with SVG charts. A runtime tracer wraps scikit-learn estimators, splitters and metrics inside a sandboxed subprocess (no network, resource limits, no secrets). A protocol reconstructor turns the trace into a replayable experiment specification. The replay engine runs counterfactual experiments, Shapley attribution and Nadeau–Bengio intervals. The repair generator rewrites the evaluation code, and the sandbox executes it for verification. The AI layer (Claude by default) reasons about feature availability and explains findings, with guards that reject invented columns and any number not present in the verified evidence.

## Challenges we ran into

- **Making a replay trustworthy.** It isn't enough to re-implement "something similar". We added a fidelity gate: the reconstruction must reproduce the author's own reported number, or no verdict is issued.
- **Interacting leaks.** Fixing leaks one at a time double-counts their effects. Patient overlap barely matters while the future feature is present, but it does matter once that feature is removed. Exact Shapley attribution over all fix combinations solved this.
- **Honest uncertainty.** Repeated random splits overlap, so naive confidence intervals are too narrow. We used the Nadeau–Bengio correction, plus a one-sided paired test for the directional question "is the claim inflated?".
- **Our own benchmarks caught our bugs.** Model candidates inside `make_pipeline` were treated as identical; groups were lost when preprocessing preceded a group split; tuple and function parameters (`MinMaxScaler(feature_range=(0, 1))`, `SelectKBest(f_classif)`) could not be replayed; a SMOTE repair referenced the wrong label variable. An independent set of hand-written notebooks on real data found the last three.
- **Measurement can't tell a leak from a strong feature.** Removing either lowers the score. Real data has genuinely near-perfect features, so no AI or rule-based suspicion removes a feature on its own: it is proposed, the verdict is computed both ways, and the user decides. <!-- add your own challenges -->

## Accomplishments we're proud of

On 24 generated studies with known ground truth, LeakVerity reached **100% detection precision**, **100% recall** and **0% false alarms** on clean notebooks. Its verified estimate was **4× closer** to the true performance than the reported number (mean error 0.021 vs 0.092), the truth fell inside its 95% CI 96% of the time, and it **never overturned an honest claim**. Verdicts in the 10 studies with a planted future feature stay at *Depends on review* until the proposal is confirmed; after confirmation, 88% of verdicts match the ground truth. On 12 independent hand-written notebooks (mostly real public data), judged against an independent cross-validated reference, it detected leaks exactly in 12/12, matched the reference verdict in 11/12, and verified 7/7 repairs. The one mismatch resolves in Thorough mode.

## What we learned

**Detecting a problem is not the same as proving its impact.** Several planted leaks changed nothing measurable, and the most dramatic inflation came from one feature. We also learned to trust verification over generation: every repair is executed, every number comes from a re-run, and when the tool can't reproduce the experiment it says so. <!-- add your personal learning: e.g. what was new to you -->

## What's next

- Replay for K-fold cross-validation protocols and `GridSearchCV` pipelines.
- Regression and multiclass metrics.
- Broader pandas-level leakage beyond the known look-ahead patterns (whole-dataset statistics computed before the split).
- A GitHub Action that attaches a Claim Survival Certificate to every pull request that changes a model's evaluation.
- Container-per-audit sandboxing for public deployment.

## Built with

python, scikit-learn, pandas, numpy, scipy, imbalanced-learn, fastapi, uvicorn, anthropic-claude, javascript, svg

## AI disclosure

See `docs/ai_usage.md`. The product uses an LLM for semantic reasoning and explanations, under guards; no reported number comes from a language model. Development used Claude (Anthropic) as a coding assistant. <!-- describe accurately -->
