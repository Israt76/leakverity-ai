# LeakVerity

**Find the leak. Measure the damage. Test the claim.**

LeakVerity checks whether a machine-learning result survives once data leakage is fixed. Upload a notebook and its dataset. LeakVerity runs the notebook in a sandbox, records what the experiment actually did, finds leaks, repairs them, re-runs the experiment with every combination of fixes, and issues a **Claim Survival Certificate**: *Supported*, *Overstated*, *Contradicted*, *Insufficient evidence*, or *Depends on review* when the answer hinges on a suspected leak only a person can confirm.

> Evidence-based checks identify possible leaks. Re-executed code produces every number. Pre-registered rules decide the verdict; a person confirms uncertain information.


## The demo in one paragraph

A synthetic hospital-readmission notebook claims *"a test AUC of 0.94, which supports deploying it at discharge for all adult patients."* LeakVerity reproduces 0.9390 exactly, then finds five problems from execution evidence:

- the scaler was fitted on all rows;
- 60 test rows are copies of training rows;
- 332 patients sit on both sides of the split;
- `followup_ed_visits_30d` looks like it is recorded after discharge (a *proposed* leak: suspected, not removed, until a person confirms it);
- the best of 10 seeds was picked on the test set.

Because the follow-up feature is only suspected, LeakVerity evaluates the claim both ways and reports **Depends on review**: Overstated if the feature is a leak, Supported (0.929) if it is legitimate. One click confirms it, and the audit re-runs with the decision recorded on the certificate. With it confirmed, re-running every combination of fixes on 6 splits gives a verified AUC of **0.709** (95% CI 0.679–0.739); the model's true AUC on fresh patients is 0.713. The Shapley allocation assigns most of the observed performance change in this synthetic demonstration to the `future_features` repair category, with a smaller share for test-set seed selection; the preprocessing category has negligible observed effect. These allocations are per repair category under the tested protocols, not isolated causal effects of individual flagged features. The corrected notebook is executed and checked against its own execution trace: it reproduces the replay engine's number to six decimals, nothing is fitted before the split, no record or patient is on both sides, the future feature is gone, and all seeds are reported (7/7 checks). For patients 65+ the AUC is only 0.60, so the claim about "all adult patients" fails. **Verdict once confirmed: Overstated.** A leak-free control notebook gets **Supported**, with zero findings.

## Does it work? A benchmark with known ground truth

`benchmarks/run_benchmark.py` generates 24 studies: random combinations of planted leaks, clean controls, and a decoy feature that sounds post-hoc but is legitimate. Each study comes from a known process, so the *true* AUC of the honestly trained model is measured on 5,000 fresh synthetic patients. These results use the rule-based feature checks and 3 replay splits per study.

| Measure | Result |
|---|---|
| Leak detection precision | 100% |
| Leak detection recall | 100% |
| False alarms on clean notebooks | 0% |
| Decoy feature wrongly flagged | 0% |
| Replays that reproduced the reported number | 100% |
| Corrected code verified by execution | 100% |
| Mean error of the reported score vs truth | 0.092 |
| Mean error of LeakVerity's verified score vs truth | 0.021 |
| Truth inside LeakVerity's 95% CI | 96% |
| Verdicts left as *Depends on review* | 42% |
| Headline verdict agreement with ground truth | 46% |
| Verdict agreement once the proposed leak is confirmed | 88% |
| Honest claims wrongly overturned | 0% |
| Overstated claims caught (once proposals are confirmed) | 77% |

**Read the verdict rows carefully.** Every study with a planted future feature (10 of 24) ends at *Depends on review*, because LeakVerity no longer removes a suspected feature on its own judgement: it proposes it, shows that the verdict changes, and asks. That is why headline agreement is 46%. Once the proposal is confirmed (as a reviewer would, since these features are post-outcome by construction), agreement is 88%. The remaining misses are cases where the true inflation was small (0.03–0.05), within the sampling noise of the test sets. LeakVerity's error is conservative: it never overturned an honest claim. Full per-scenario results: [`benchmarks/results/BENCHMARK.md`](benchmarks/results/BENCHMARK.md).

### Independent evaluation (not from the generator)

Twelve hand-written notebooks in varied coding styles, mostly on real bundled datasets (breast cancer, wine). They cover SMOTE before the split, SelectKBest on pure noise, a leak inside a helper function, cross-validation and `GridSearchCV` (where LeakVerity must abstain), a future-peeking rolling mean built in pandas, an ambiguous centred window over rows with no time order (which must be proposed, not removed), and a clean control with row-wise features. Expected verdicts come from an independent reference: each model re-implemented honestly and evaluated by repeated cross-validation. Results: **exact leak detection 12/12, verdict matches the reference 11/12, repairs verified 7/7, 0 false findings.** The pandas-level look-ahead leak (c09) was a miss before static look-ahead detection was added after review. The one remaining mismatch is the pure-noise notebook (claimed AUC 0.76): at the default 3 splits LeakVerity says *Insufficient evidence* because its CI still contains 0.76; in Thorough mode it returns **Contradicted** (verified 0.502). On real data, genuinely strong features such as `worst_radius` get a note, never a removal. See [`benchmarks/results/INDEPENDENT.md`](benchmarks/results/INDEPENDENT.md).

## How it works

| Stage | What happens |
|---|---|
| 1. Reproduce | The notebook runs unmodified in a sandbox. A tracer records every split (exact rows), fit (rows seen, overlap with the test set) and metric (and which labels it used). |
| 2. Investigate | Execution evidence, data checks and rule-based feature screening produce findings, each with code location, evidence and a fix. Suspected features (from name rules or look-ahead code without time-order evidence) are **proposed**, never removed automatically. |
| 3. Repair | The corrected notebook keeps the author's data preparation and rewrites the evaluation (`# FIX` marks every change). It is **executed under the tracer**: its output must match the replay engine, and its own trace must show every targeted leak gone. |
| 4. Measure | **Fidelity gate:** the reconstructed protocol must reproduce the reported number, or LeakVerity abstains. Then every combination of fixes is re-run on several splits. The observed score difference is allocated across **repair categories** using exact Shapley values, with **Nadeau–Bengio corrected** intervals. A category can cover multiple findings, so its share is a joint, protocol-dependent allocation—not a separately established causal effect for each feature. |
| 5. Test the claim | On the corrected protocol, with the same model selection as the corrected result: stability across 20 re-splits, subgroup performance, and time-based validation (train on the past, test on later patients). Failures that contradict the claim's stated scope change the verdict; others are listed as caveats. |
| 6. Certify | Pre-registered rules produce the verdict for each scenario (proposed leaks confirmed or legitimate): **Supported**, **Overstated** (real signal, but the claim does not survive), **Contradicted** (the claimed value is ruled out *and* no above-chance performance is shown), or **Insufficient evidence** (unsupported protocol, a claim about a metric the notebook does not compute, or too uncertain to confirm or rule out the claim). If the scenarios disagree, the verdict is **Depends on review**. "Chance" is defined per metric (for example 2p/(1+p) for F1). If any repair fails verification, the verdict is marked **provisional**. Intervals for bounded metrics such as AUC are computed on the logit scale, so they never leave [0, 1]. The certificate records input hashes, environment, protocol, seeds and every number. |

Details: [`docs/methodology.md`](docs/methodology.md).

**Leaks covered:** preprocessing or feature selection fitted before the split; resampling (SMOTE) before the split; duplicate records across train and test; the same person or group on both sides; features unavailable at prediction time; look-ahead features built in pandas (centred windows, negative shifts, backward fill, target encoding); model or hyper-parameter selection on the test set.

## Run it

```bash
pip install -r requirements.txt
LEAKVERITY_ALLOW_UPLOADS=1 uvicorn app.main:app --port 7860   # open http://localhost:7860 (uploads on for local use)
```

Click **Open the saved audit** for the instant demo, or **Run it live** (about 2 minutes on 2 CPU cores). The Docker image used for public hosting turns uploads off (demos still run live), because auditing executes the uploaded code. See [`docs/security.md`](docs/security.md). Deployment to Hugging Face Spaces or Render: [`docs/deploy.md`](docs/deploy.md).

```bash
python -m pytest -q tests                  # run the test suite
python benchmarks/run_benchmark.py 24 7    # about 5 minutes
python benchmarks/independent/build_data.py && python benchmarks/run_independent.py
```

## Scope and limitations

Supported: pandas + scikit-learn notebooks or scripts (imbalanced-learn, XGBoost and LightGBM estimators too), tabular binary classification, hold-out splits (`train_test_split`, `ShuffleSplit`, `StratifiedShuffleSplit`, `GroupShuffleSplit`). K-fold cross-validation loops are detected, but LeakVerity abstains rather than replaying them. Pandas-level leaks are caught only when they match a known look-ahead pattern; other whole-dataset statistics computed before the split (for example `x - x.mean()`) are not seen. Whether a feature is available at prediction time is a semantic judgement; replay measures its effect but cannot decide legitimacy, so suspected features are proposed and the user decides. See [`docs/limitations.md`](docs/limitations.md) and [`docs/security.md`](docs/security.md).

## Prior work

Leakage detection and quick fixes already exist: Yang et al. (ASE 2022); NBLyzer (Subotić et al., 2022); LeakageDetector 2.0 (ICSME 2025); LeakLens. Impact varies hugely by leak type (Roth, 2026, *Which Leakage Types Matter?*). LeakVerity's contribution is the claim-centred verification workflow: runtime evidence, a fidelity gate, measured repair-category effects with interactions, repairs verified by execution, and pre-registered verdicts. See [`docs/prior_art.md`](docs/prior_art.md).


## Project structure

```
leakverity/        engine: tracer, sandbox, spec, replay, findings, feature screening, repair, survival, verdict, report
app/               FastAPI server and the web interface (app/static)
examples/          demo dataset, leaky and clean notebooks, saved audits
benchmarks/        ground-truth benchmark and results
tests/             pytest suite
docs/              methodology, limitations, prior art, security, deploy, demo script
```

**Data.** The demo study and generated benchmark are synthetic and non-identifiable. The independent evaluation also uses two real public datasets bundled with scikit-learn: Breast Cancer Wisconsin (Diagnostic) and Wine, both from the UCI Machine Learning Repository. No real patient records are used.

MIT licence. © 2026 Israt Jahan Aunika.
