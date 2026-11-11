# Independent Final Benchmark Results Audit

## Decision

READY_WITH_DISCLOSED_LIMITATIONS

## Scope and method

Read-only audit of the existing final benchmark report against `raw/scenario_manifest.csv`, `raw/per_run_results.csv`, and `raw/per_stage_results.csv`. No benchmark execution, tuning, source/config/result edits, or report edits. Percentiles use nearest-rank `ceil(p*N)`; medians use the conventional midpoint for even N. Missing values are excluded and counted, never imputed. Machine-readable detail and named checks are in `audit_checks.json`.

## Independent held-out results

Convergence/status/failure/validity denominators are N_all=48 per approach. Each continuous metric gives available count and its own denominator (available values); missing count is shown. Position error and residuals in metres; runtime seconds; iteration counts in iterations.

| Approach | N / converged (%) | Numerical statuses | Fail / numfail | Valid / invalid / incomplete | Position error: n/miss; median / mean / p90 / p95 / max | Residual RMS: n/miss; median / mean / p90 / p95 / max | Runtime: n/miss; median / mean / p90 / max | Iterations: n/miss; median / mean / p90 / max |
|---|---:|---|---:|---:|---|---|---|---|
| GN | 48 / 33 (68.8%) | `{"converged":33,"rank_deficient":14,"line_search_failed":1}` | 1 / 0 | 0 / 0 / 48 | 48/48; 973.018 / 6.24075e+07 / 6.21491e+06 / 2.19046e+07 / 2.9169e+09 | 48/48; 14.7632 / 321.787 / 999.16 / 1036.19 / 1771.16 | 48/48; 0.00052005 / 0.00124563 / 0.0015485 / 0.0198093 | 48/48; 2 / 3.79167 / 5 / 52 |
| LM | 48 / 31 (64.6%) | `{"converged":31,"rank_deficient":13,"numerical_failure":2,"max_iterations":2}` | 4 / 2 | 0 / 0 / 48 | 48/48; 918.715 / 298965 / 1.18573e+06 / 2.82562e+06 / 3.87034e+06 | 48/48; 44.0433 / 343.981 / 999.937 / 1036.19 / 1771.16 | 48/48; 0.0015805 / 0.00375434 / 0.008034 / 0.0293351 | 48/48; 4 / 11.6667 / 20 / 100 |
| BFGS | 48 / 33 (68.8%) | `{"converged":33,"rank_deficient":13,"max_iterations":2}` | 2 / 0 | 0 / 0 / 48 | 48/48; 918.715 / 211123 / 1.18573e+06 / 1.70798e+06 / 2.82562e+06 | 48/48; 44.0433 / 346.261 / 999.937 / 1036.19 / 1771.16 | 48/48; 0.00205075 / 0.00371441 / 0.0087878 / 0.0218939 | 48/48; 11 / 18.4375 / 45 / 100 |
| initial_selection_only | 48 / 33 (68.8%) | `{"converged":33,"unobservable":13,"max_iterations":2}` | 2 / 0 | 0 / 0 / 48 | 35/48; 1089.73 / 409803 / 1.32559e+06 / 3.76643e+06 / 3.87034e+06 | 35/48; 0.0683692 / 210.714 / 735.723 / 796.256 / 999.937 | 48/48; 0.00045455 / 0.00200512 / 0.0019355 / 0.028652 | 48/48; 2 / 6.58333 / 5 / 100 |
| fixed_gn_adaptive | 48 / 33 (68.8%) | `{"converged":33,"unobservable_geometry":14,"line_search_failed":1}` | 1 / 0 | 0 / 0 / 48 | 48/48; 973.018 / 6.24075e+07 / 6.21491e+06 / 2.19046e+07 / 2.9169e+09 | 48/48; 14.7632 / 321.787 / 999.16 / 1036.19 / 1771.16 | 48/48; 0.0008007 / 0.00146269 / 0.00183 / 0.0187543 | 48/48; 2 / 3.79167 / 5 / 52 |
| full_hybrid | 48 / 33 (68.8%) | `{"converged":33,"unobservable_geometry":13,"max_iterations":2}` | 2 / 0 | 0 / 0 / 48 | 48/48; 918.715 / 339564 / 1.18573e+06 / 2.82562e+06 / 4.88738e+06 | 48/48; 44.0433 / 343.49 / 999.937 / 1036.19 / 1771.16 | 48/48; 0.00075945 / 0.00328087 / 0.0031138 / 0.0557811 | 48/48; 2 / 10.75 / 5 / 200 |

Validity accounting identity was checked per approach: N_valid + N_invalid + N_incomplete = N_all. It passes for all six; every test row is incomplete, with zero complete assessments. Numerical status counts are exact raw strings.

## Report-versus-raw comparison

**Main table:** 48 checks across table row coverage, convergence counts/rates, validity counts, metric values, and failure totals; all pass. Values were matched at the precision displayed in the report.

**Geometry table:** all 24 approach/category rows match scenario count, converged count and position-error median/p90.

**Scenario and split counts:** 192 scenarios, 144 development, 48 test, 1152 run rows, 1035 stage rows; report and metadata match.

**Status/failure counts:** exact numerical status counts match `validation_checks.json`; all failure / numerical-failure counts in the report match raw flags.

**Validity:** N counts and accounting match. Fit status remains not-evaluated when an estimate exists; 13 initial-selection-only rows have no state and are not-assessable. Neither is treated as a pass.

**Missing metrics:** test missing counts are GN: position/RMS/max-abs 0/0/0, LM: position/RMS/max-abs 0/0/0, BFGS: position/RMS/max-abs 0/0/0, initial_selection_only: position/RMS/max-abs 13/13/13, fixed_gn_adaptive: position/RMS/max-abs 0/0/0, full_hybrid: position/RMS/max-abs 0/0/0. The report?s 13 missing initial-selection-only outputs and 48/48 availability for the other methods match.

**Hybrid:** report transition and stage count statements match raw stage rows: 33 one-stage, 2 two-stage, 0 three-stage; transitions {'Levenberg-Marquardt -> BFGS': 2}. Row `solver_switches` agrees.

### Discrepancies

No numerical mismatch was found in the main results table, geometry table, validity accounting, failure counts, scenario/split counts, missing metric statement, or hybrid transitions. One report completeness omission: it describes the held-out results as paired but provides no paired difference estimates, pair denominators, or exclusion reasons. This audit supplies them below; before paper integration, either add these audit values to the report or avoid implying that paired comparisons were reported there.

## Paired comparison audit

Pairing is by identical held-out `scenario_id`; differences are alternative minus GN in position error (m), restricted to pairs where both numerical statuses are `converged` and both position errors are present. Shared denominator is 48 for every comparison. No significance test is performed.

| Alternative | Shared N | Both converged | Excluded | Exclusion reasons | Median paired difference (m) | Mean paired difference (m) |
|---|---:|---:|---|---|---:|---:|
| LM | 48 | 30 | 18 | {"GN_not_converged":15,"alternative_not_converged":3} | -2.69401e-07 | -940324 |
| BFGS | 48 | 32 | 16 | {"GN_not_converged":15,"alternative_not_converged":1} | -5.44814e-08 | -881554 |
| initial_selection_only | 48 | 32 | 16 | {"GN_not_converged":15,"alternative_not_converged":1} | 0 | -881554 |
| fixed_gn_adaptive | 48 | 33 | 15 | {"GN_not_converged":15} | 0 | 0 |
| full_hybrid | 48 | 32 | 16 | {"GN_not_converged":15,"alternative_not_converged":1} | 0 | -881554 |

The paired comparisons are descriptive and do not establish statistical significance or superiority.

## Hybrid transition verification

- Held-out full-hybrid scenarios represented: 35 of 48.
- One stage: {hybrid["scenarios_one_stage"]}; two stages: {hybrid["scenarios_two_stages"]}; three stages: {hybrid["scenarios_three_stages"]}; other stage counts: {hybrid["other_stage_counts"]}.
- Transitions: Levenberg-Marquardt -> BFGS=2.
- Unexpected transitions: [].
- Row-level switch counts: {0: 46, 1: 2}. Stage count equals switches + 1 for every scenario represented in stage output: True. Solver sequence field agrees with stage names for 35 scenarios; unparseable 0; disagreements 0.

## Scientific interpretation review

- Numerical convergence is kept separate from accuracy and validity; position error is truth-based evaluation and not a validity-assessor input.
- Residual RMS is distinct from position error; null residual thresholds mean no acceptance decision. Incomplete assessments are not described as software failures.
- Geometry results are conditioned on verified geometry categories; they should not be treated as general GPS geometry without the synthetic-model limitation.
- Runtime uses one invocation per scenario with the benchmark timing method; no independent-run uncertainty is available.
- The report makes no superiority claim and correctly states synthetic ENU rays are not orbital ephemerides/real visibility.
- Validity completeness is 0/48 per algorithm. The reported validity policy therefore cannot support complete valid/invalid rates.

## Limitations and readiness

There are no raw-to-report numeric discrepancies. Held-out comparisons have only 48 paired scenarios, and this audit uses descriptive paired differences without inferential tests. Synthetic ENU ray geometry limits external validity. The fit thresholds remain unset, so a complete validity decision is unavailable. The original report omits explicit paired estimates despite mentioning pairing; that should be disclosed or added before paper integration.

**Final recommendation: READY_WITH_DISCLOSED_LIMITATIONS.** The underlying tables are numerically consistent and scientifically cautious, with the paired-results omission disclosed as a paper-integration limitation.
