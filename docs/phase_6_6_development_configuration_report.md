# Phase 6.6 — Development Configuration Review

## Executive summary

This review used only rows labeled `development` in the saved `results/pilot_v2` manifest and per-run CSV. It did not inspect, summarize, compare, or use held-out test outcomes. The analysis retained the existing split and data; no scenarios were regenerated or repartitioned.

The development manifest contains 144 scenarios and the run CSV contains 864 rows: 144 per each of the six algorithms, with no duplicate `(scenario_id, algorithm)` keys. The development split covers all 96 configured parameter combinations and every configured factor level.

The data support retaining `[-1,000 m, 100,000 m]` only as the explicitly temporary synthetic terrestrial screening profile. They support `1e6` as a clear engineering geometry-screening boundary between the pilot's verified poor and moderate/good *initial* geometries, but do not validate final-state behavior. The fit thresholds remain unresolved: the archived raw files lack maximum absolute residuals and validity assessment fields, the residual distribution changes substantially with realized outlier conditions, and no application-level fit acceptance requirement or measurement covariance policy is supplied. Do not run the held-out evaluation with the current unconfigured fit limits and claim complete validity assessments.

The proposed JSON artifact is [phase_6_6_proposed_configuration.json](./phase_6_6_proposed_configuration.json). It is explicitly pending review and is not a production/default configuration.

## Inspected implementation and artifacts

- [solution_validity.py](../src/gps_hybrid/solution_validity.py): `SolutionValidityConfig` fields are `operating_domain`, optional `max_residual_rms_m`, optional `max_absolute_residual_m`, `condition_number_threshold` (default `1e6`), and `rank_tolerance` (default `1e-12`). `synthetic_benchmark_validity_config()` selects the temporary terrestrial interval but leaves both fit thresholds unset. Under the approved policy, this makes fit `not_evaluated` and a full validity assessment incomplete.
- [benchmarking.py](../src/gps_hybrid/benchmarking.py): `BenchmarkConfig` is a dataclass in this module; there is no `src/gps_hybrid/config.py`. The Python API accepts an explicit `validity_config`. The CLI exposes output directory, scenario count, seed, test fraction, and plotting only; it has no fit-threshold flags. `_run_row()` calls validity assessment with observations, final estimate, and numerical status before reading truth for evaluator position/clock errors.
- [geometry.py](../src/gps_hybrid/geometry.py): final geometry can be obtained from the final state by `diagnose_geometry`; its SVD rank tolerance defaults to `1e-12`. It exposes rank, condition number, and PDOP. The existing benchmark raw columns `numerical_rank`, `condition_number`, and `pdop` in this saved pilot describe initial geometry, not the final estimate.
- [selection.py](../src/gps_hybrid/selection.py): `SelectionConfig.condition_number_threshold` defaults to `1e8`. That is an initial solver-selection policy, not the validity cutoff; no selection behavior is proposed to change.
- [hybrid.py](../src/gps_hybrid/hybrid.py): numerical status remains its own result field and the hybrid's stage/transition policy is separate from validity classification.
- [scenario.py](../src/gps_hybrid/scenario.py): generated truth is at latitude 37°, longitude -122°, WGS-84 ellipsoidal altitude 20 m. Pseudorange noise is generated with `rng.normal(0, pseudorange_noise_std_m, satellite_count)`; benchmark outliers are separate signed fixed-magnitude modifications. Directions are synthetic local ENU rays, not real GPS visibility/orbital geometry.
- [solution_validity_policy.md](./solution_validity_policy.md) and [README.md](../README.md), plus the development-only files [per_run_results.csv](../results/pilot_v2/raw/per_run_results.csv), [scenario_manifest.csv](../results/pilot_v2/raw/scenario_manifest.csv), and `summaries/development_*_summary.csv`.

The development raw CSV header has no `validity_` columns, estimated ECEF/geodetic state fields, final-state geometry fields, or maximum-absolute-residual field. It does contain final `residual_rms_m`, solver-reported RMS, truth-derived evaluator errors, and initial-geometry fields. Thus validity outcomes and their eligible populations cannot be reconstructed from this archive. Missing assessments are **unavailable**, not zero valid, zero invalid, or an inferred incomplete count.

## Development data provenance and coverage

The reviewed source is the existing Phase 6.5B pilot-v2 dataset (configured seed `20261110`; 192 scenarios configured). Only its `split == development` rows were retained for analysis: 144 manifest rows and 864 run rows. No held-out test outcome was used.

| Factor | Configured levels represented in development |
|---|---|
| Satellite count | 4, 8 |
| Pseudorange noise standard deviation | 0.3 m, 3.0 m |
| Initial position perturbation | 10 m, 500 m |
| Requested geometry | good, moderate, poor, rank_deficient |
| Configured outlier rate | 0.0, 0.5, 1.0 |
| Full parameter combinations | 96 of 96 |

Each algorithm has 144 rows. There are no duplicate scenario/algorithm keys. The six names in the raw data are `GN`, `LM`, `BFGS`, `initial_selection_only`, `fixed_gn_adaptive`, and `full_hybrid`.

Requested and verified geometry are not interchangeable. In the 144 development scenarios, requested counts were good 35, moderate 31, poor 43, and rank-deficient 35; verified counts were good 51, moderate 15, poor 43, and rank-deficient 35. Sixteen requested moderate scenarios verified as good. This is retained as a coverage/data-quality observation, not silently recoded.

Realized outlier conditions from the development manifest:

| Configured rate | Scenarios | Realized none | Realized partial | Realized all |
|---:|---:|---:|---:|---:|
| 0.0 | 47 | 47 | 0 | 0 |
| 0.5 | 49 | 3 | 44 | 2 |
| 1.0 | 48 | 0 | 0 | 48 |

The `0.5` rate therefore includes a few no-outlier and all-observation realizations; configured rate must not be substituted for realized condition.

## Numerical outcomes and evaluator accuracy

These are historical raw statuses from the saved pilot, not current-code re-executions. Each rate uses 144 development rows for that algorithm. `position_error_m` and `clock_bias_error_m` use scenario truth only as post-solve evaluator metrics; neither is an input to validity or configuration selection.

| Algorithm | Converged | Other saved statuses |
|---|---:|---|
| GN | 87 (60.4%) | rank_deficient 56; max_iterations 1 |
| LM | 84 (58.3%) | rank_deficient 35; max_iterations 23; numerical_failure 2 |
| BFGS | 84 (58.3%) | rank_deficient 35; max_iterations 25 |
| Initial-selection-only | 86 (59.7%) | unobservable 35; max_iterations 23 |
| Fixed-GN adaptive | 87 (60.4%) | unobservable_geometry 56; max_iterations 1 |
| Full hybrid | 88 (61.1%) | unobservable_geometry 35; max_iterations 21 |

Across all algorithms, `N_all = 864` and saved numerical convergence count is `N_converged = 516` (59.7%). These status counts do not measure validity or physical correctness.

For converged rows, position error median / 90th percentile / maximum (metres) and clock-bias error median / 90th percentile (metres) were:

| Algorithm | Position error median / p90 / max | Clock-bias error median / p90 |
|---|---:|---:|
| GN | 1,646 / 20,058,431 / 21,996,489 | 392 / 20,058,428 |
| LM | 1,377 / 448,753 / 1,730,467 | 242 / 15,172 |
| BFGS | 1,303 / 66,226 / 1,730,467 | 184 / 5,127 |
| Initial-selection-only | 1,347 / 341,927 / 1,730,467 | 242 / 12,745 |
| Fixed-GN adaptive | 1,646 / 20,058,431 / 21,996,489 | 392 / 20,058,428 |
| Full hybrid | 1,377 / 749,899 / 2,418,729 | 253 / 99,172 |

These large errors reinforce that numerical convergence and accuracy are separate quantities. They do not establish that any one algorithm is faster, more accurate, or more reliable. The saved pilot predates the final LM overflow robustness change, so its historical LM failure count should not be presented as a current implementation rate.

## Residual and geometry evidence

The saved independently recomputed `residual_rms_m` is available for 827/864 runs; 37 values are missing. Solver-reported RMS is available for 757/864; 107 values are missing. Among available recomputed RMS values across all algorithms and statuses, median was 143.74 m, p90 966.16 m, p95 1,022.51 m, p99 1,580.20 m, and maximum 1,933.67 m. These are correlated run rows because algorithms share scenarios; they are descriptive, not independent observations or a recommended acceptance quantile.

Realized outlier condition shows why a single data-quantile cutoff is not defensible by itself:

| Realized condition | RMS values available | Median RMS | p90 RMS | Maximum RMS |
|---|---:|---:|---:|---:|
| None | 289 | 0.24 m | 95.39 m | 966.16 m |
| Partial observations | 251 | 469.10 m | 902.92 m | 1,933.67 m |
| All observations | 287 | 741.94 m | 1,021.56 m | 1,621.99 m |

These pooled values include six algorithm outcomes per scenario and available non-converged results. The residual RMS distribution does not distinguish a noisy but acceptable observation set from an injected-outlier case without the intended measurement-noise/outlier policy. The raw archive has no residual vectors or maximum absolute residuals, so the maximum-residual distribution and threshold failure population cannot be calculated. A low residual is not evidence of physical correctness; saved converged solutions exhibit large truth-based errors.

Other development factor breakdowns (same pooled run-level caveat):

| Factor level | RMS available | Median RMS | p90 RMS | Maximum RMS |
|---|---:|---:|---:|---:|
| 4 satellites | 408 | 0.000003 m | 948.36 m | 1,621.99 m |
| 8 satellites | 419 | 484.96 m | 983.02 m | 1,933.67 m |
| Noise 0.3 m | 402 | 174.32 m | 983.09 m | 1,621.99 m |
| Noise 3.0 m | 425 | 143.74 m | 959.89 m | 1,933.67 m |
| Position perturbation 10 m | 418 | 95.08 m | 940.73 m | 1,022.51 m |
| Position perturbation 500 m | 409 | 289.02 m | 1,003.47 m | 1,933.67 m |

These are descriptive associations only. Geometry, outliers, solver, and realization vary within each factor slice; the differences do not isolate a causal effect or supply an acceptance threshold.

The available condition and PDOP fields describe the **initial** geometry. By requested geometry, initial condition-number ranges were:

| Requested category | Initial condition-number range | Initial PDOP median (where defined) |
|---|---:|---:|
| good | 2.06–2.89 | 1.29 |
| moderate | 3.61–5,006.20 | 15.28 |
| poor | 186,051,511–219,025,719 | 49,996,127 |
| rank_deficient | unavailable (rank 1) | unavailable |

At `1e6`, 0/35 good, 0/31 moderate, and 43/43 poor requested scenarios exceed the cutoff in initial geometry; rank-deficient scenarios have no finite condition number and must remain a separate category. This is strong evidence for a transparent pilot-category screening cutoff at the **initial** state. Final estimated states and final-state geometry were not saved, so the development data cannot establish how many final solutions would be poor-observability, rank deficient, or unavailable under validity's required final-state evaluation.

## Validity assessment counts and denominators

The archive has no validity result columns, final estimates, or geodetic coordinates. Consequently, the following cannot be reported from this dataset: `N_complete`; valid, invalid, and incomplete counts; physical-domain evaluated/failure/unavailable counts; final-state poor-observability and rank-deficiency denominators; configured-threshold fit failures; or their rates. In particular, it would be incorrect to label unavailable validity results as incomplete outcomes or count them as passes.

The current default synthetic validity configuration leaves both fit limits `None`. If used for a new run without explicit fit limits, every otherwise measurable fit reports `not_evaluated`, making the overall assessment incomplete under the policy. Therefore `N_complete` would not represent a complete validity experiment until explicit, approved fit limits are configured.

For a future assessment, preserve the policy denominators separately for each development/test split and algorithm: `N_all` is every run row, `N_converged` is exact numerical status `converged`, `N_complete` is complete validity assessments; physical-domain rates use only successfully converted states with a bounded domain; poor-observability rates use only full-rank geometry with finite condition; rank-deficiency rates exclude insufficient observations and unavailable ranks; measurement-fit failure rates require at least one configured threshold and computable residual statistics. The current files do not permit recomputing these populations. The validity module accepts no truth data, and `_run_row()` calls it before evaluator-only truth error calculations.

## Configuration proposal

| Setting | Proposed value | Evidence / rationale | Decision |
|---|---|---|---|
| Physical domain | `terrestrial`, inclusive `-1000 m` to `100000 m` WGS-84 ellipsoidal altitude | Generated synthetic truth is fixed at 20 m. The wide interval is consistent with that synthetic setup and the established diagnostic convention, but estimated altitudes are absent from the archive. | Ready to freeze only as a temporary synthetic-benchmark screening profile. Not a universal terrestrial or aviation constraint. Do not constrain/project solver estimates. |
| `max_residual_rms_m` | **Unresolved (`null`)** | Available RMS varies substantially by realized outlier condition; measurement noise has two configured levels and injected outliers have fixed signed magnitude. No service-level residual criterion or covariance/weighting model is approved. | Requires explicit measurement acceptance requirements and a designed development-only calibration/verification experiment. Do not select a pooled quantile to maximize validity. |
| `max_absolute_residual_m` | **Unresolved (`null`)** | The raw archive has no residual vector or maximum-absolute-residual field. A threshold cannot be evaluated or defensibly calibrated from RMS alone. | Requires residual-vector output in a future authorized development experiment and an explicit acceptance rationale. |
| `condition_number_threshold` | `1e6` | In initial development geometry, this boundary separates every verified poor case (all above `1.86e8`) from all observed good/moderate cases (moderate max about `5.01e3`). It matches the implementation-defined poor-category boundary. Final-state diagnostics are absent. | Propose for freeze as an engineering screening policy, clearly labeled and independently reviewable; not a universal constant or a solver-selection threshold. Final-state outcome counts remain unverified. |
| `rank_tolerance` | `1e-12` | Existing geometry/SVD convention; no development evidence warrants changing it. Rank deficiency remains distinct from poor conditioning. | Retain; no change proposed. |
| Unrestricted waiver | `allow_unrestricted_domain_waiver = false` | Policy requires explicit waiver and reports unrestricted as `not_applicable`, never a pass. Current synthetic study is terrestrial. | Freeze as false; no waiver enabled. |
| Benchmark design | Preserve pilot-v2 factors and split provenance for this proposal; do not regenerate/repartition here | Development covers all 96 combinations and all configured factor levels. The final experiment size/power target is not specified by this phase. | No scenario-count or split change proposed. Existing CLI cannot set validity thresholds; any future run needing them must use explicit `BenchmarkConfig(validity_config=...)` through the Python API unless CLI support is separately approved. |

The JSON proposal intentionally leaves both fit limits null and sets `freeze_ready` to false. This reflects the evidence gap rather than a zero threshold. Thresholds must not be tuned against held-out test outcomes.

## Limitations, unresolved decisions, and next step

1. **Fit acceptance requirement:** what residual RMS and maximum absolute residual represent acceptable measurement fit for this synthetic application? The policy should be tied to requirements, the known noise/outlier process, and preferably an explicit covariance/noise model—not to validity-rate maximization.
2. **Measurement output:** an authorized development-only run or saved artifacts must provide maximum absolute residuals and final-state estimates/geometry. The existing saved pilot cannot be retroactively converted into these metrics without regenerating/re-executing scenarios, which this phase did not do.
3. **Representativeness and inference:** 144 scenarios over 96 combinations provide factor coverage but limited replication (mostly one or two scenarios per combination); shared scenarios across six algorithms create paired, correlated results. Do not treat the 864 rows as independent trials.
4. **Version alignment:** historical LM statuses predate the 6.5F damping robustness change. They remain useful provenance, not a current-code reliability estimate.
5. **Physical screening:** `[-1,000,100,000] m` is temporary and broad. It is not a general definition of terrestrial altitude and is not an aviation operating envelope.
6. **Freeze readiness:** the physical profile, rank tolerance, condition cutoff, and no-waiver semantics can be reviewed as proposed policy choices. Fit limits remain unresolved; therefore the complete validity policy is not ready for final held-out evaluation unless stakeholders explicitly approve an alternative that reports fit as not evaluated and accepts incomplete assessments.

No code defect requiring a source change was identified. No solver, algorithm, solver-selection policy, hybrid transition, scenario generator, split, default configuration, or pilot output was changed. No held-out test result influenced this report.

## Commands and validation

Development-only CSV analysis used PowerShell here-strings piped to the selected project interpreter. Each analysis parsed the saved CSV and retained only rows whose `split` value was `development`; printed output contained no held-out values. The analyzed files were `results/pilot_v2/raw/scenario_manifest.csv` and `results/pilot_v2/raw/per_run_results.csv`.

Source/document inspection included `Get-Content` on the policy, README, implementation modules, and development summary CSV; `rg -n` located configuration, CLI, and classification definitions. No benchmark, test suite, regeneration, or repartition command was run. The proposed JSON is validated separately after creation.
