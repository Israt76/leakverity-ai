# Independent evaluation

Twelve hand-written notebooks, not produced by the development generator, in varied coding styles, on real public datasets bundled with scikit-learn (breast cancer, wine) and new synthetic ones. Expected *findings* were written before the first run (c09 was later re-scored more strictly). Expected *verdicts* come from an independent reference: an honest re-implementation of each model evaluated by repeated cross-validation (`benchmarks/independent/references.py`), a different method from LeakVerity's hold-out replay. Rule-based reader, 3 replay splits.

- Exact leak detection: **12/12** cases (TP 7, FP 0, FN 0)
- Verdict matches the independent reference: **11/12**
- Repairs verified by execution: **7/7**

| Case | What it tests | Expected leaks | Found | Reported | LeakVerity verified | Reference (CV) | Verdict | Expected verdict |
|---|---|---|---|---|---|---|---|---|
| c01_bc_clean | real data, leak-free; strong genuine features must not be removed | none | none | 0.996 | 0.995 | 0.995 | Supported | Supported |
| c02_bc_scaler_first | textbook scaler leak; module-style imports, NumPy arrays; impact should be negligible | preprocessing | preprocessing | 0.998 | 0.997 | 0.995 | Supported | Supported |
| c03_noise_selectkbest | feature selection on all data with pure-noise features; truth is chance | preprocessing | preprocessing | 0.760 | 0.539 | 0.482 | Insufficient evidence | Contradicted |
| c04_smote_first | SMOTE before the split | resampling | resampling | 0.996 | 0.897 | 0.893 | Overstated | Overstated |
| c05_cross_val | cross-validation protocol: must abstain | none | none | — | — | — | Insufficient evidence | Insufficient evidence |
| c06_gridsearch | GridSearchCV object: must abstain, not guess | none | none | — | — | — | Insufficient evidence | Insufficient evidence |
| c07_wine_tuned_on_test | k tuned on the test set inside a helper function; accuracy metric | test_selection | test_selection | 0.981 | 0.944 | 0.977 | Supported | Supported |
| c08_bc_exported_twice | records exported twice (same sample_id) | duplicates | duplicates | 0.973 | 0.996 | 0.990 | Supported | Supported |
| c09_sensor_centered_rolling | future-peeking (centred) rolling mean built in pandas before scikit-learn | future_features | future_features | 0.771 | 0.737 | 0.647 | Overstated | Overstated |
| c10_bc_clean_groupless_proba | leak-free with internal early stopping | none | none | 0.984 | 0.986 | 0.991 | Supported | Supported |
| c11_bc_centered_no_time | AMBIGUOUS: centred window, no time order, computed before the split: proposed (not removed), evaluated both ways | proposed_features | proposed_features | 0.998 | 0.996 | 0.995 | Supported (holds in both scenarios) | Supported (holds in both scenarios) |
| c12_bc_rowwise_features | clean control: row-wise engineered features; unconditional Supported expected | none | none | 1.000 | 0.997 | 0.995 | Supported | Supported |

**Notes.**
- **c09** (future-peeking rolling mean built in pandas) was a miss in an earlier version. Static look-ahead detection (`leakverity/lookahead.py`: centred windows, negative shifts, backward fill, target encoding) was added after review and now catches it; removing the feature is measured as material and the verdict is Overstated.
- **c11** is deliberately ambiguous, not clean: a centred window over rows with no time order, computed before the split, so test rows shape training features. LeakVerity reports the cross-partition mixing, treats the feature as a *proposed* leak instead of removing it, evaluates the claim with and without it, and reports Supported in both scenarios. **c12** is the clean control: row-wise features only, unconditional Supported.
- **c03** (pure-noise features): with the default 3 replay splits on 60 test rows, the corrected CI (0.297–0.765) still contains the claimed 0.76, so LeakVerity returns *Insufficient evidence* rather than *Contradicted*. Run in Thorough mode (6 splits), the same notebook gives **Contradicted** (verified 0.502, 95% CI 0.365–0.639). Scored as a mismatch at the default setting.
