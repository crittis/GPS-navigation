# Final Held-Out Benchmark Report


## 1. Experiment objective

Execute and validate the established six-approach synthetic GPS benchmark on the frozen design, with held-out results reserved for final evaluation.

## 2. Exact configuration and provenance

- Git baseline: `ffa0a845f094de70ad4ba7ab5031e1ff6d1a756f`. Pre-execution test status supplied in the brief: 233 passed in 2.14s; tests not rerun.
- Benchmark version 0.1.0; Python 3.14.0; NumPy 2.5.4; Windows / Python 3.14.0 venv; execution UTC 2026-11-10T09:38:12.579258+00:00.
- Design source: `results/pilot_v2/experiment_metadata.json`. Full solver and hybrid settings are recorded in `experiment_metadata.json`.
- Seed 20261110; 192 scenarios; test fraction 0.25; factors: satellites [4,8], noise SD [0.3,3.0] m, position perturbation [10,500] m, clock perturbation 100 m, geometry [good,moderate,poor,rank_deficient], outlier rates [0,0.5,1], outlier magnitude 1000 m.
- Validity: terrestrial altitude [-1000,100000] m; waiver false; condition cutoff 1,000,000 (engineering screening threshold); rank tolerance 1e-12; both fit thresholds null. Null means no fit decision, not pass.
- Approaches: GN, LM, BFGS, initial_selection_only, fixed_gn_adaptive, full_hybrid.

## 3. Benchmark execution details

Existing engine and paired shared scenarios. `perf_counter` outer timing, no warm-up, generation and file I/O excluded, selection/orchestration included; stage times are solver-reported.

## 4. Dataset and split integrity checks

- 192 scenarios; development 144, test 48; 1152 run rows and 1035 stage rows.
- Unique IDs per split: {'development': True, 'test': True}; no leakage: True.
- Six rows per scenario: True; unique scenario/algorithm pairs: True; run IDs match manifest: True.
- Seed sequence/configuration: True; all 96 combinations in full dataset/development: True; test full combinations: False (48/96, expected design coverage). Factor levels are represented in both splits.
- Per-algorithm row coverage: {'GN': 192, 'LM': 192, 'BFGS': 192, 'initial_selection_only': 192, 'fixed_gn_adaptive': 192, 'full_hybrid': 192}.

## 5. Held-out results

Every approach denominator N=48. Convergence, failure and validity counts use all rows. Error, residual, runtime and iteration statistics use available values among those 48, including nonconverged outcomes. Position error, residual RMS and maximum absolute residual are available for 48/48 rows for GN, LM, BFGS, fixed_gn_adaptive and full_hybrid; for initial_selection_only they are available for 35/48 rows because 13 outputs have no assessable estimate. Runtime and iterations are available for 48/48 for every approach. Nearest-rank percentiles; errors in metres; runtime in seconds.

| Approach | Converged / 48 | Valid / invalid / incomplete | Position error median / mean / p90 / p95 / max | Residual RMS median / mean / p90 / p95 / max | Runtime median / mean / p90 / max | Iterations median / mean / p90 / max | Failures / numerical failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| GN | 33/48 (68.8%) | 0 / 0 / 48 | 973.01787 / 62407492.314936 / 6214910.695764 / 21904644.672277 / 2916903600.126636 | 14.763171 / 321.786545 / 999.159959 / 1036.189147 / 1771.157805 | 0.00052005 / 0.00124563 / 0.001549 / 0.019809 | 2.0 / 3.7917 / 5.0 / 52.0 | 1 / 0 |
| LM | 31/48 (64.6%) | 0 / 0 / 48 | 918.714675 / 298965.291752 / 1185729.967912 / 2825622.924225 / 3870338.872711 | 44.043349 / 343.980516 / 999.937404 / 1036.189147 / 1771.157805 | 0.0015805 / 0.00375434 / 0.008034 / 0.029335 | 4.0 / 11.6667 / 20.0 / 100.0 | 4 / 2 |
| BFGS | 33/48 (68.8%) | 0 / 0 / 48 | 918.714675 / 211122.991326 / 1185729.968434 / 1707984.066569 / 2825622.924304 | 44.043349 / 346.261057 / 999.937404 / 1036.189147 / 1771.157805 | 0.00205075 / 0.00371441 / 0.008788 / 0.021894 | 11.0 / 18.4375 / 45.0 / 100.0 | 2 / 0 |
| initial_selection_only | 33/48 (68.8%) | 0 / 0 / 48 | 1089.734595 / 409802.746784 / 1325591.461797 / 3766434.057537 / 3870338.872711 | 0.068369 / 210.71432 / 735.72322 / 796.256399 / 999.937404 | 0.00045455 / 0.00200512 / 0.001935 / 0.028652 | 2.0 / 6.5833 / 5.0 / 100.0 | 2 / 0 |
| fixed_gn_adaptive | 33/48 (68.8%) | 0 / 0 / 48 | 973.01787 / 62407492.314936 / 6214910.695764 / 21904644.672277 / 2916903600.126636 | 14.763171 / 321.786545 / 999.159959 / 1036.189147 / 1771.157805 | 0.0008007 / 0.00146269 / 0.00183 / 0.018754 | 2.0 / 3.7917 / 5.0 / 52.0 | 1 / 0 |
| full_hybrid | 33/48 (68.8%) | 0 / 0 / 48 | 918.714675 / 339563.63549 / 1185729.968433 / 2825622.924308 / 4887379.956873 | 44.043349 / 343.489698 / 999.937404 / 1036.189147 / 1771.157805 | 0.00075945 / 0.00328087 / 0.003114 / 0.055781 | 2.0 / 10.75 / 5.0 / 200.0 | 2 / 0 |

## Failure and numerical-status accounting

Failure count uses the benchmark `is_failure` field; numerical failures use `is_numerical_failure`. Exact test status counts are preserved in `validation_checks.json`. Failures / numerical failures: GN 1 / 0; LM 4 / 2; BFGS 2 / 0; initial_selection_only 2 / 0; fixed_gn_adaptive 1 / 0; full_hybrid 2 / 0.

## 6. Validity-assessment accounting

For every approach, N_all=48, N_complete=0, N_valid=0, N_invalid=0, N_incomplete=48. N_converged is 33 for GN, BFGS, initial_selection_only, fixed_gn_adaptive and full_hybrid, and 31 for LM. The identity N_valid + N_invalid + N_incomplete = N_all holds for each. Fit is `not_evaluated` wherever a finite estimate exists; the 13 initial_selection_only rows without an estimate are `not_assessable`. Neither status is a pass. Incomplete is a policy assessment outcome, not a software failure. Numerical convergence is separate. The assessor receives observations, estimate, numerical status and policy, never truth.

## 7. Position-error analysis

Position error is truth-based evaluator distance, not validity input. All-row summaries above include nonconverged estimates. These descriptive paired outcomes have no inferential tests; no superiority claim is made.

## 8. Residual analysis

Residual RMS is final-estimate fit (m), distinct from position error. Maximum absolute residual is exported diagnostically. Null fit limits mean neither metric provides an acceptance decision; low residual does not prove physical correctness.

## 9. Runtime analysis

Single invocation timing per scenario/algorithm under the benchmark timing method. No independent-repeat uncertainty estimate.

## 10. Geometry-conditioned analysis

Test split grouped by verified geometry category; all-row position-error median / p90 (m).

| Approach | Geometry | N | Converged | Position error median / p90 m |
|---|---|---:|---:|---:|
| GN | good | 21 | 21 | 842.450142 / 1564.618288 |
| GN | moderate | 9 | 9 | 74578.114235 / 2825622.924308 |
| GN | poor | 5 | 3 | 21904644.672277 / 2916903600.126636 |
| GN | rank_deficient | 13 | 0 | 647.520038 / 1182.512854 |
| LM | good | 21 | 19 | 842.450142 / 1564.618281 |
| LM | moderate | 9 | 9 | 74578.114225 / 2825622.924225 |
| LM | poor | 5 | 3 | 15844.05899 / 3870338.872711 |
| LM | rank_deficient | 13 | 0 | 647.520038 / 1182.512854 |
| BFGS | good | 21 | 21 | 842.450142 / 1564.618289 |
| BFGS | moderate | 9 | 9 | 74578.114234 / 2825622.924304 |
| BFGS | poor | 5 | 3 | 15843.147381 / 1712372.177247 |
| BFGS | rank_deficient | 13 | 0 | 647.520038 / 1182.512854 |
| initial_selection_only | good | 21 | 21 | 842.450142 / 1564.618288 |
| initial_selection_only | moderate | 9 | 9 | 74578.114235 / 2825622.924308 |
| initial_selection_only | poor | 5 | 3 | 15844.05899 / 3870338.872711 |
| initial_selection_only | rank_deficient | 13 | 0 | unavailable / unavailable |
| fixed_gn_adaptive | good | 21 | 21 | 842.450142 / 1564.618288 |
| fixed_gn_adaptive | moderate | 9 | 9 | 74578.114235 / 2825622.924308 |
| fixed_gn_adaptive | poor | 5 | 3 | 21904644.672277 / 2916903600.126636 |
| fixed_gn_adaptive | rank_deficient | 13 | 0 | 647.520038 / 1182.512854 |
| full_hybrid | good | 21 | 21 | 842.450142 / 1564.618288 |
| full_hybrid | moderate | 9 | 9 | 74578.114235 / 2825622.924308 |
| full_hybrid | poor | 5 | 3 | 15844.05899 / 4887379.956873 |
| full_hybrid | rank_deficient | 13 | 0 | 647.520038 / 1182.512854 |

## 11. Hybrid transition analysis

Full-hybrid stage transition counts are derived from held-out per-stage rows.
- Transitions: {'Levenberg-Marquardt -> BFGS': 2}.
- Scenarios by stage count: {1: 33, 2: 2}.

## 12. Limitations and unavailable metrics

- All validity outcomes are incomplete because fit limits are unset; no complete validity rates are claimed.
- No statistical inference or independent benchmark repeat was performed.
- Geometry is synthetic ENU rays, not orbital ephemerides or real visibility.
- Test split covers 48/96 combinations but includes every factor level. Missing/nonfinite counts are in `validation_checks.json`; none are imputed.

## 13. Interpretation

Results are descriptive. Convergence is not validity; low residual is not truth accuracy. Incomplete validity prevents validity ranking; paired error/runtime differences do not establish superiority.

## 14. Reproducibility information

Full configuration, seed, execution timestamp, versions, timing method and split coverage are in `experiment_metadata.json`; scenario-level seeds and assignments are in the manifest. This verifies provenance and internal integrity, not independent reproduction.

## 15. Held-out use statement

Held-out results were not used for tuning. No source changes were made and tests were not rerun.

## Preservation and generated files

Source Git status: `?? results/final_benchmark/`; pilot-v2 remained unchanged. Generated files:
- `experiment_metadata.json`
- `raw/per_run_results.csv`
- `raw/per_stage_results.csv`
- `raw/scenario_manifest.csv`
- `summaries/development_ablation_summary.csv`
- `summaries/development_geometry_summary.csv`
- `summaries/development_initial_perturbation_summary.csv`
- `summaries/development_noise_summary.csv`
- `summaries/development_outlier_condition_summary.csv`
- `summaries/development_overall_summary.csv`
- `summaries/development_satellite_count_summary.csv`
- `summaries/test_ablation_summary.csv`
- `summaries/test_geometry_summary.csv`
- `summaries/test_initial_perturbation_summary.csv`
- `summaries/test_noise_summary.csv`
- `summaries/test_outlier_condition_summary.csv`
- `summaries/test_overall_summary.csv`
- `summaries/test_satellite_count_summary.csv`
- `validation_checks.json`
- `FINAL_BENCHMARK_REPORT.md`
