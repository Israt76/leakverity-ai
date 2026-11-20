# AI reader vs rule-based reader: ambiguous feature names

15 features, 7 of them post-discharge leaks. Task: 30-day readmission predicted at discharge.

Proposed = flagged as a suspected leak; LeakVerity then evaluates the claim with and without it. Notes = strength-only observations that never affect the verdict.

| Reader | Precision (proposals) | Recall (proposals) | Missed leaks | Legitimate features proposed | Notes |
|---|---|---|---|---|---|
| rule-based | 56% | 71% | ed_ct_30, readm_billing_code | followup_appt_scheduled, lab_result_count, outcome_risk_score_admit, prior_readmissions | readm_billing_code |

_AI reader not run: set ANTHROPIC_API_KEY or OPENAI_API_KEY and re-run to add it._

## Features

| Feature | Truth | Meaning |
|---|---|---|
| `followup_ed_visits_30d` | leak | ER visits in the 30 days after discharge |
| `ed_ct_30` | leak | ER visit count, 30 days after discharge (cryptic name) |
| `readm_billing_code` | leak | billing code assigned at the readmission |
| `days_to_next_adm` | leak | days until the next admission |
| `cm_call_outcome` | leak | result of the care-manager call made after discharge |
| `snf_claims_30d` | leak | nursing-facility claims in the 30 days after discharge |
| `rx_fill_gap_post` | leak | medication refill gap after discharge |
| `discharge_hour` | legitimate | hour of discharge |
| `followup_appt_scheduled` | legitimate | whether a follow-up appointment was booked before discharge |
| `prior_readmissions` | legitimate | readmissions in the previous year |
| `outcome_risk_score_admit` | legitimate | risk score computed at admission |
| `lab_result_count` | legitimate | number of lab results during the stay |
| `max_troponin_stay` | legitimate | highest troponin during the stay |
| `disp_cd` | legitimate | discharge disposition code (known at discharge) |
| `length_of_stay` | legitimate | length of stay |
