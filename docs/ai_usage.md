# AI usage disclosure

STEM2Connect requires AI assistance to be disclosed and attributed. This project uses AI in two distinct ways.

## 1. AI inside the product

| Component | Model | What it does | What it may not do |
|---|---|---|---|
| Feature audit (`leakverity/semantic.py`) | Configurable: Anthropic Claude (default `claude-sonnet-5-5`) or OpenAI | Suggests which features would not be available at the prediction point, with a reason and confidence | Name columns that do not exist (rejected); remove a feature (suggestions are only proposals, evaluated both ways; the user confirms); decide impact (measured by replay) |
| Claim reading | Same | Identifies the claimed metric, value and scope | Supply a value not present in the text |
| Explanations | Same | Rewrites verified findings in plain language | Introduce any number not already in the evidence (number guard) |

Without an API key, rule-based readers with the same interfaces are used, and the interface labels which reader ran. **No number on any LeakVerity screen or certificate is produced by a language model.**

## 2. AI assistance during development

> This disclosure distinguishes project decisions and reviews from AI-generated implementation.

- **Ideation and selection.** Candidate ideas (ResultGuard, LeakHunter/LeakVerity, PeerReviewShield, ChartCheck, ClaimStress) were discussed with two AI assistants: ChatGPT (OpenAI) and Claude (Anthropic). I compared their arguments against the judging criteria, the judging panel and the existing gallery, and chose LeakVerity and its scope (pandas + scikit-learn, binary classification, hold-out splits).
- **Implementation.** Most of the code (tracer, sandbox, replay engine, statistics, repair generator, survival tests, verdict rules, web interface, tests, benchmarks) and the documentation were written by **Claude (Anthropic)** as a coding agent, working from my instructions and decisions.
- **Review and correction.** I commissioned several rounds of independent technical review of the code from ChatGPT and had every issue they raised checked against the source and fixed. Among them: survival tests evaluating the wrong model, verdicts ignoring failed survival tests, an over-strong "Contradicted" rule, unbounded AUC intervals, a missed look-ahead leak, AI suspicions removing features without confirmation, claims checked against the wrong metric, and uploads enabled by default. The fixes were re-validated by the test suite and by re-running all benchmarks.
- **Validation.** Every result reported in this repository comes from code that was executed: 41 automated tests, the generated benchmark, the independent benchmark, and the saved demo audits.
- **My own work.** I selected the final problem, project name and supported scope; chose verified repair/replay and the Claim Survival Certificate as the central deliverables; reviewed successive versions and requested independent technical audits; directed revisions based on those findings; and published the public project repository. I did not manually write most of the source code: Claude generated the bulk of the implementation and initial documentation. I am responsible for checking the accuracy of project claims and the final competition submission.
## Evidence status

The saved demo, generated benchmark, and handwritten benchmark were run with the **rule-based reader**. The optional LLM integration exists, but the published AI comparison documents do **not** include a completed live-model result. No measured AI-versus-rules improvement is claimed.

## Data

The demo study and the generated benchmark use synthetic, non-identifiable data from `leakverity/synth.py`. The independent evaluation also uses two **real public datasets bundled with scikit-learn**: Breast Cancer Wisconsin (Diagnostic) and Wine, both from the UCI Machine Learning Repository (credit: UCI and scikit-learn). Its remaining cases use small synthetic datasets built in `benchmarks/independent/build_data.py`. No real patient records are used or distributed.
