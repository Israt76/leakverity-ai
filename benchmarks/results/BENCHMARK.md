# LeakVerity benchmark

24 generated studies (6 clean controls), rule-based reader (no AI key), 3 replay splits each. Ground truth = the honestly trained model evaluated on 5,000 fresh synthetic patients from the same process.

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
| Headline verdict agreement with ground truth | 46% |
| Verdicts left as *Depends on review* (semantic proposal changes the verdict) | 42% |
| Verdict agreement once the proposed leak is confirmed | 88% |
| Honest claims wrongly overturned | 0% |
| Overstated claims caught (once proposals are confirmed) | 77% |

## Per leak type

| Leak | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|
| preprocessing | 100% | 100% | 10 | 0 | 0 |
| duplicates | 100% | 100% | 11 | 0 | 0 |
| group_overlap | 100% | 100% | 17 | 0 | 0 |
| future_features | 100% | 100% | 10 | 0 | 0 |
| test_selection | 100% | 100% | 10 | 0 | 0 |

## Scenarios

| ID | Model | Planted | Found | Reported | Verified (95% CI) | Truth | Verdict | Ground truth |
|---|---|---|---|---|---|---|---|---|
| S00 | lr | none | none | 0.722 | 0.731 (0.685–0.772) | 0.728 | Supported | Supported |
| S01 | rf | duplicates, future_features, group_overlap, preprocessing, test_selection | duplicates, future_features, group_overlap, preprocessing, test_selection | 0.942 | 0.648 (0.492–0.778) | 0.706 | Depends on review | Overstated |
| S02 | rf | future_features, group_overlap | future_features, group_overlap | 0.896 | 0.755 (0.672–0.824) | 0.711 | Depends on review | Overstated |
| S03 | lr | group_overlap, preprocessing | group_overlap, preprocessing | 0.734 | 0.756 (0.636–0.846) | 0.734 | Supported | Supported |
| S04 | lr | none | none | 0.715 | 0.732 (0.637–0.810) | 0.745 | Supported | Supported |
| S05 | lr | duplicates, group_overlap | duplicates, group_overlap | 0.748 | 0.727 (0.707–0.746) | 0.738 | Supported | Supported |
| S06 | lr | group_overlap, preprocessing, test_selection | group_overlap, preprocessing, test_selection | 0.754 | 0.715 (0.605–0.804) | 0.738 | Supported | Supported |
| S07 | lr | group_overlap, test_selection | group_overlap, test_selection | 0.713 | 0.707 (0.684–0.729) | 0.735 | Supported | Supported |
| S08 | lr | none | none | 0.710 | 0.720 (0.640–0.789) | 0.731 | Supported | Supported |
| S09 | lr | future_features, group_overlap, preprocessing, test_selection | future_features, group_overlap, preprocessing, test_selection | 0.936 | 0.710 (0.508–0.855) | 0.733 | Depends on review | Overstated |
| S10 | lr | future_features, group_overlap, test_selection | future_features, group_overlap, test_selection | 0.861 | 0.732 (0.578–0.846) | 0.729 | Depends on review | Overstated |
| S11 | lr | duplicates, future_features, group_overlap, preprocessing, test_selection | duplicates, future_features, group_overlap, preprocessing, test_selection | 0.948 | 0.750 (0.641–0.835) | 0.740 | Depends on review | Overstated |
| S12 | rf | none | none | 0.726 | 0.744 (0.640–0.827) | 0.712 | Supported | Supported |
| S13 | lr | duplicates, future_features, group_overlap, test_selection | duplicates, future_features, group_overlap, test_selection | 0.869 | 0.716 (0.616–0.800) | 0.738 | Depends on review | Overstated |
| S14 | rf | duplicates, group_overlap, preprocessing | duplicates, group_overlap, preprocessing | 0.760 | 0.737 (0.545–0.870) | 0.726 | Supported | Overstated |
| S15 | rf | duplicates, future_features, group_overlap, preprocessing, test_selection | duplicates, future_features, group_overlap, preprocessing, test_selection | 0.954 | 0.747 (0.573–0.868) | 0.719 | Depends on review | Overstated |
| S16 | rf | none | none | 0.674 | 0.688 (0.634–0.737) | 0.718 | Supported | Supported |
| S17 | lr | duplicates, future_features, group_overlap | duplicates, future_features, group_overlap | 0.915 | 0.740 (0.685–0.788) | 0.738 | Depends on review | Overstated |
| S18 | rf | duplicates, group_overlap, preprocessing | duplicates, group_overlap, preprocessing | 0.766 | 0.764 (0.702–0.818) | 0.721 | Supported | Overstated |
| S19 | lr | duplicates, future_features, group_overlap, preprocessing | duplicates, future_features, group_overlap, preprocessing | 0.977 | 0.737 (0.617–0.831) | 0.740 | Depends on review | Overstated |
| S20 | lr | none | none | 0.708 | 0.724 (0.630–0.802) | 0.736 | Supported | Supported |
| S21 | rf | duplicates, group_overlap, test_selection | duplicates, group_overlap, test_selection | 0.753 | 0.752 (0.702–0.797) | 0.719 | Supported | Overstated |
| S22 | rf | future_features, preprocessing, test_selection | future_features, preprocessing, test_selection | 0.858 | 0.730 (0.563–0.853) | 0.706 | Depends on review | Overstated |
| S23 | lr | duplicates, group_overlap | duplicates, group_overlap | 0.747 | 0.749 (0.533–0.888) | 0.730 | Supported | Supported |
