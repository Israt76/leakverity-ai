# Prior work and how LeakVerity differs

LeakVerity did not invent leakage detection, automated fixes, or the idea of measuring leakage impact. It builds on:

- **Yang, Brower-Sinning, Lewis & Kästner (ASE 2022), *Data Leakage in Notebooks: Static Detection and Better Processes*.** Static analysis of overlap, preprocessing and multi-test leakage; found leakage pervasive across 100,000+ public notebooks.
- **Subotić et al. (ICSE-SEIP 2022; SOAP 2022) and Drobnjaković, Subotić & Urban (2022).** NBLyzer and an abstract-interpretation leakage analysis for notebooks.
- **LeakageDetector 2.0 (Stevens Institute of Technology, ICSME 2025 tool track).** A VS Code extension that detects overlap, preprocessing and multi-test leakage and offers quick fixes and assisted corrections.
- **LeakLens and FeatureLeakageLens (PyPI).** Train/test validity audits, AST checks for preprocessing leakage, and feature-leakage audits.
- **Roth (2026), *Which Leakage Types Matter?* (arXiv:2604.04199).** Counterfactual experiments on 2,047 tabular datasets showing that leakage types differ by orders of magnitude, with normalisation leakage negligible and selection leakage dominant.
- **Kapoor & Narayanan (2023), *Leakage and the reproducibility crisis in ML-based science*.** A taxonomy of leakage and its effect on published findings.
- **Nadeau & Bengio (2003), *Inference for the Generalization Error*.** The corrected resampled t-test used for every interval.

## What is different here

Existing tools answer "does this notebook contain a leakage pattern?" LeakVerity answers "**does this specific claim survive once the leaks are fixed, and which leaks actually mattered?**"

1. **Execution evidence instead of code patterns alone.** Splits, fits and metrics are traced at runtime: which rows each step saw, how many test rows duplicate training rows, which candidates were scored on the test labels.
2. **Fidelity-gated replay.** Before any counterfactual, the reconstructed protocol must reproduce the author's own number, or the tool abstains.
3. **Measured impact of repair categories, with interactions.** Every combination of supported fixes is re-run; the observed score difference is allocated across repair categories with exact Shapley values and corrected CIs. Several findings may share a category, and changes to the evaluation population are protocol-dependent, not isolated causal effects of individual leaks. A category with negligible observed impact is reported as such, consistent with Roth (2026).
4. **Repairs verified by execution.** The corrected notebook is run, and its output must match the replay engine.
5. **Claim-level verdicts.** Pre-registered rules turn the evidence, plus stability, subgroup and time-based survival tests, into Supported / Overstated / Contradicted / Insufficient evidence, issued as a certificate with input hashes.

LeakVerity's novelty is this combination and its verification discipline, not any single detector.
