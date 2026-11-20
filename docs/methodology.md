# Methodology

LeakVerity answers one question about a machine-learning result: **does the reported number survive once the experiment is run without leakage?** Every number it reports comes from re-executed code. The AI layer only reads and explains.

## 1. Reproduce (execution trace)

The notebook runs unmodified in a sandboxed subprocess. A tracer wraps pandas and scikit-learn and records:

| Event | What is recorded |
|---|---|
| `read_csv` | file, shape, columns |
| `train_test_split`, `ShuffleSplit`, `GroupShuffleSplit`, K-fold splitters | exact train/test row positions, seed, stratification, groups, and how many test rows are byte-identical copies of training rows |
| every `fit`, `fit_transform`, `fit_resample` | class, parameters, rows and columns seen, and after the split how many of those rows belong to the test set |
| every metric call | metric, value, and a hash of the labels it was computed on (so test-set metrics can be told apart) |

The **design matrix** is the input of the earliest preprocessing step fitted before the split, or the split input if there is none. This keeps the author's own feature engineering while letting LeakVerity re-run everything that happens afterwards.

## 2. Reconstruct the protocol

From the trace, LeakVerity rebuilds the experiment at the scikit-learn level: steps fitted before the split, the split, steps fitted on training rows, the model candidates and their parameters, and the metric that was reported. If several models were scored on the same test labels and the best was reported, that is recorded as **test-set selection** (seed selection if only the random seed varied, otherwise hyper-parameter selection).

## 3. Investigate

| Rule | Evidence | Fix used in replay |
|---|---|---|
| LV-PREPROCESSING | a transformer or selector was fitted on rows that include the test set | fit it inside the training pipeline |
| LV-RESAMPLING | a sampler such as SMOTE ran before the split | resample training rows only |
| LV-DUPLICATES | test rows that are the *same source record* as a training row: every column identical, IDs included. Different people with identical features are not duplicates. | drop records exported more than once (keep the first) |
| LV-GROUP-OVERLAP | the same person or entity on both sides of a random split | group-aware split |
| LV-FUTURE-FEATURE | a feature the **user confirmed** is not available at prediction time (`known_leaks`, or the *Confirm* button) | drop the feature |
| LV-PROPOSED-LEAK | a feature the AI or rule-based reader *suspects* is unavailable at prediction time (post-outcome name, cryptic code read as post-outcome, near-perfect univariate AUC) | **proposed**: evaluated both with and without the feature (see §7) |
| LV-LOOKAHEAD | static code analysis: a model feature assigned from a centred rolling window, a negative `shift`, backward fill, or `groupby(...)[target].transform` (built in pandas before the split), **with** verifiable evidence: time order (a date column the rows are sorted by, or explicit sorting), or target encoding, which leaks whatever the order | drop the feature (conservative; measured by replay) |
| LV-LOOKAHEAD-REVIEW | the same patterns without time-order evidence. The finding reports cross-partition mixing (how many training rows' windows include test rows) | **proposed**: evaluated both ways |
| LV-NOTE-STRONG-FEATURE | a feature that is only strong (high univariate AUC) with nothing else suspicious | none: a note, never affects the verdict |
| LV-TEST-SELECTION | several candidates compared on the test set | average over seeds, or choose hyper-parameters by 3-fold CV on training data |

A semantic flag is a **hypothesis**. Its effect is measured, never assumed. But measurement cannot tell a leak from a legitimately strong predictor, because removing either lowers the score. So **no AI or rule-based suspicion removes a feature by itself.** A suspected feature becomes a *proposed* fix: the replay runs every combination with and without it, and the verdict is computed for both scenarios. Only the user (or verifiable code evidence such as time order) turns a proposal into a definite removal. Users can confirm features either way (`known_leaks` / `not_leaks`, or the buttons on each proposed finding, which re-run the audit).

## 4. Verified counterfactual replay

1. **Fidelity.** The reconstructed protocol is replayed as written with the author's seed. It must reproduce the notebook's own reported number within 0.001, or LeakVerity **abstains** ("Insufficient evidence").
2. **Counterfactuals.** Every subset of the fixes is applied and the experiment re-run on *J* random splits (the author's seed plus *J*−1 fixed seeds). Test-set selection is evaluated from the same runs.
3. **Attribution.** For each split, the change from the as-written to the fully corrected result is divided among the fixes with **exact Shapley values**: each fix's average marginal effect over every order of applying the fixes. Shares add up to the total, and interacting leaks are not double-counted. LeakVerity also reports each fix's effect *alone* and *once all other leaks are fixed*.
4. **Uncertainty.** Repeated random splits overlap, so naive standard errors are too small. Intervals use the **Nadeau–Bengio corrected resampled t-interval**: variance × (1/J + n_test/n_train).
5. **Classification** from the fair-share 95% CI, with materiality threshold 0.005:
   - *Material*: the whole CI is above 0.005, or the effect once the other leaks are fixed is.
   - *Real, size unclear*: the CI excludes zero but reaches above the threshold.
   - *Small but real*: the CI excludes zero and lies entirely below the threshold.
   - *Negligible*: the whole CI lies inside ±0.005.
   - *Uncertain*: anything else.
6. **Protocol changes.** Removing duplicates and group-aware splitting change which rows are in the test set. Their shares are labelled in the interface and certificate as *protocol change: different test rows*, i.e. changes under a corrected evaluation protocol, not isolated causal effects measured on the same rows.

## 5. Repair, verified by execution

The corrected script keeps the author's code up to the first leaky statement and rewrites the evaluation protocol, marking every change with `# FIX`. It is then run in the sandbox under the same tracer, and the repair is **Verified** only if every check passes on the corrected run's *own* execution trace:

| Check | Evidence from the corrected run's trace |
|---|---|
| Reproduces the replay engine's number | printed result matches the corrected value for the same split to within 1e-4 |
| Nothing fitted on the full data before the split | no transformer or sampler fit precedes the split |
| No step after the split is fitted on test rows | every fit's rows are matched against the test rows |
| No source record in both train and test | split positions are mapped back to the source records |
| No group in both train and test | split positions are mapped back to the group column |
| Removed features absent | columns (or width) of every model fit |
| Test set used once / all seeds reported | number of metric calls on the test labels |

## 6. Claim survival tests (fully corrected protocol, fixed in advance)

| Test | Procedure | Concern if |
|---|---|---|
| Seed and split stability | 20 new random splits evaluating **the model the corrected protocol reports**: seed selection rotates through the candidate seeds; hyper-parameter selection re-runs inner CV on each split's training data | SD > 0.03 |
| Subgroup performance | AUC per subgroup level on each held-out split, Nadeau–Bengio CIs across splits. Counts are per split (the splits overlap), plus the number of unique people seen | a subgroup is > 0.05 below overall and its CI excludes the overall value; fewer than 15 cases per class per split = inconclusive |
| Time-based validation | train before the 75th-percentile date, test after, new groups only, same corrected selection | > 0.03 below the random-split estimate |

A test missing the data it needs is reported as **Not applicable**, never as a pass. Stability and time-based tests with fewer than 50 test rows (or too few cases of a class) are **Inconclusive**. Every survival report carries its limitations: the tests use the uploaded data only, subgroup counts are per split, and passing them is not external validation.

When there are proposed leaks, the survival tests are run **twice**: once on the protocol that removes the proposed features and once on the protocol that keeps them, so each scenario's verdict uses its own evidence.

## 7. Verdict rules

| Verdict | Rule |
|---|---|
| Supported | the claim is within the verified 95% CI + 0.01, there is no reliable paired inflation, the corrected CI excludes chance, and no survival test fails against the claim's scope. Non-contradicting survival concerns are listed as caveats. |
| Overstated | the claim does not survive but signal remains (the corrected CI excludes chance, or its point estimate is more than 0.05 above chance). "Does not survive" means: the claim is above the verified CI + 0.01; or the paired inflation has a one-sided 95% lower bound > 0.01 and the claim exceeds the verified mean by > 0.02; or a survival test fails against the claim's scope. |
| Contradicted | the claimed value is ruled out, the corrected CI includes chance, **and** the corrected point estimate is within 0.05 of chance. Reported as "the result does not stand", not as proof that the model has no signal. |
| Insufficient evidence | fidelity failed, the protocol is unsupported, or replay failed; or the corrected estimate is too uncertain to confirm or rule out the claim (its CI contains both the claim and chance). |

| Depends on review | there are proposed (unconfirmed) leaks and the verdict differs depending on whether they are leaks. Both scenario verdicts and their reasons are shown. If both scenarios agree, that verdict is issued with "holds whether or not the proposed leak is confirmed". |

**Claim metric.** The claim must be about a metric the notebook actually computes on the test set. If the claim names AUC but the notebook only reports accuracy, LeakVerity abstains rather than verifying a different number.

**Chance level.** "Above chance" is defined per metric: AUC and balanced accuracy 0.5; accuracy the majority-class rate; average precision and precision the positive rate *p*; F1 the F1 of always predicting positive, 2*p*/(1+*p*). Recall has no meaningful chance level (predicting all positive gives 1.0), so for recall LeakVerity never issues *Contradicted* and only judges whether the claim survives.

**Provisional.** If any required repair fails generation or verification, the verdict is marked *provisional* and the reason is shown. The numbers still come from the replay engine.

**Bounded intervals.** For metrics in [0, 1], the Nadeau–Bengio interval is computed on the logit scale, with values capped half a case from 0 and 1, and transformed back. Intervals can therefore never exceed the metric's range.

*Revision history.* The Contradicted rule was tightened after an external technical review. Earlier versions called a claim contradicted whenever the corrected CI included chance, which confuses "not demonstrated" with "contradicted". Every benchmark in this repository was re-run with the final rules.

## 8. The AI layer and its guards

| AI task | Guard |
|---|---|
| Suggest which features are unavailable at prediction time | must name existing columns; suggestions are only *proposed* leaks, evaluated both ways; the user decides |
| Extract the claim (metric, value, scope) | the sentence must occur verbatim in the notebook and the value inside that sentence; otherwise the rule-based reader is used |
| Explain findings for the reader | the text may only contain numbers present in the verified evidence, otherwise the template text is kept |

Without an API key, rule-based readers with the same interfaces are used, and the interface says so.
