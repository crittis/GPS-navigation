# Phase 6.5 Pilot Benchmark Audit

## 1. Pilot configuration

The pilot used the existing `BenchmarkConfig` and `run_benchmark` interface; no
benchmark code or algorithm was changed for this execution.

| Parameter | Value |
|---|---|
| Scenario count | 48 |
| Seed | 20261109 |
| Test fraction | 0.25 |
| Satellite counts | 4, 8 |
| Noise standard deviations (m) | 0.3, 3.0 |
| Position perturbations (m) | 10.0, 500.0 |
| Clock-bias perturbation (m) | 100.0 |
| Requested geometry | good, moderate, poor, rank_deficient |
| Configured outlier rates | 0.0, 0.5, 1.0 |
| Outlier magnitude (m) | 1000.0 |
| Require both splits | true |

Exact execution command:

```powershell
D:\MyProjects\DAAPRO2\.venv\Scripts\python.exe -c "from gps_hybrid.benchmarking import BenchmarkConfig, run_benchmark; c=BenchmarkConfig(scenario_count=48, seed=20261109, test_fraction=0.25, satellite_counts=(4,8), noise_stds_m=(0.3,3.0), position_perturbations_m=(10.0,500.0), geometry_categories=('good','moderate','poor','rank_deficient'), outlier_rates=(0.0,0.5,1.0), require_both_splits=True); r=run_benchmark(c, output_dir=r'results/pilot'); print('scenarios=',len(r.manifest_rows),'runs=',len(r.run_rows),'stages=',len(r.stage_rows)); print('split_counts=',r.metadata['split_counts'])"
```

Outputs were written under `results/pilot/`. The run produced 48 scenarios, 288
per-run rows, and 259 per-stage rows.

## 2. Scenario and split counts

| Split | Scenarios | Algorithm rows |
|---|---:|---:|
| Development | 36 | 216 |
| Test | 12 | 72 |
| Total | 48 | 288 |

The manifest, metadata, and raw run CSV agree on these counts. Every scenario
has exactly six algorithm rows. There are zero duplicate `(scenario_id,
algorithm)` keys.

## 3. Coverage findings

The metadata describes the split procedure as largest-remainder proportional
allocation by parameter stratum, with seeded tie-breaking.

Configured factor coverage:

| Factor | Development | Test |
|---|---|---|
| Satellite count | 4 only; 8 missing | 4 only; 8 missing |
| Noise standard deviation | 0.3 and 3.0 represented | 0.3 and 3.0 represented |
| Position perturbation | 10.0 and 500.0 represented | 10.0 and 500.0 represented |
| Requested geometry | all four represented | all four represented |
| Configured outlier rate | 0.0, 0.5, 1.0 represented | 0.0, 0.5, 1.0 represented |

There are 96 configured full parameter combinations. Development contains 36
observed combinations and is missing 60; test contains 12 and is missing 84.
Neither split claims complete full-combination coverage. The satellite-count
factor is also incomplete in both splits, and the metadata reports the missing
configured value explicitly.

Realized outlier conditions were present in the raw data: 16 scenarios had no
affected observations, 15 had partial affected observations, and 17 had all
observations affected.

## 4. Algorithm-level results

The following status counts are from the raw per-run CSV, not inferred from
summary filenames.

| Algorithm | Development converged | Test converged | Development statuses | Test statuses |
|---|---:|---:|---|---|
| GN | 23/36 | 9/12 | 23 converged, 12 rank-deficient, 1 max-iterations | 9 converged, 3 rank-deficient |
| LM | 23/36 | 8/12 | 23 converged, 10 rank-deficient, 3 max-iterations | 8 converged, 2 rank-deficient, 2 max-iterations |
| BFGS | 22/36 | 8/12 | 22 converged, 10 rank-deficient, 4 max-iterations | 8 converged, 2 rank-deficient, 2 max-iterations |
| Initial selection only | 23/36 | 8/12 | 23 converged, 10 unobservable, 3 max-iterations | 8 converged, 2 unobservable, 2 max-iterations |
| Fixed-GN adaptive | 23/36 | 9/12 | 23 converged, 12 unobservable-geometry, 1 max-iterations | 9 converged, 3 unobservable-geometry |
| Full hybrid | 23/36 | 8/12 | 23 converged, 10 unobservable-geometry, 3 max-iterations | 8 converged, 2 unobservable-geometry, 2 max-iterations |

Across all algorithms there were 187 converged, 23 max-iterations, 78
unobservable/rank-deficient, and 0 numerical-failure rows. Missing estimate
metrics occurred for 12 unobservable rows. Final residual RMS is populated for
276 rows and solver-reported residual RMS for 252 rows; both columns are
present in the raw CSV and missing values remain blank where unavailable.

The development and test summary files are separate. Each split's overall
summary contains six algorithm rows with 36 or 12 scenario-algorithm rows,
respectively. The paired ablation files report 36 available pairs for each
comparison in development and 12 in test. No paired comparison mixes split
membership.

Hybrid audit checks found no sequence or switch-count mismatch between the raw
run rows and stage records for either hybrid algorithm.

## 5. Numerical sanity checks

Representative findings from the raw rows:

- Good geometry converged normally in the inspected representative case, with
  approximately 0.24 m position error, 0.17 m clock-bias error, and residual
  RMS around `3.45e-9` m.
- Moderate geometry produced converged rows with approximately 1.7 million m
  position error and 1.7 million m clock-bias error while residual RMS was near
  `1.9e-9` m in one representative scenario.
- Poor full-rank geometry produced converged rows with errors from millions to
  tens of millions of metres. One inspected GN row had approximately 4.14
  million m position error and residual RMS around `2.08e-9` m; another had
  approximately 21.98 million m position error and residual RMS around
  `1.86e-9` m.
- Rank-deficient scenarios produced no converged rows. They were reported as
  rank-deficient or unobservable, as expected.
- No-outlier, partial-outlier, and all-observation-outlier scenarios were all
  generated and represented in the raw data.
- The inspected converged rows retained finite iteration counts, function
  evaluations, and nonnegative execution times.

The large-error/low-residual combinations are a material numerical warning:
`converged` does not by itself establish accurate receiver recovery in the
poorly conditioned synthetic geometries. No algorithm, threshold, or model was
changed to improve these pilot metrics.

## 6. Failures and warnings

- Numerical failures: 0 in the pilot.
- Maximum-iteration outcomes: present in GN/LM/BFGS and hybrid comparisons as
  reported above.
- Rank-deficient/unobservable outcomes: present and retained; they were not
  dropped from denominators or summaries.
- Test execution emitted one existing NumPy overflow warning from the LM
  augmented damping calculation during a difficult benchmark test. It did not
  fail execution.
- Matplotlib was not installed, so optional plotting was not run.

## 7. Metadata/raw discrepancies

No discrepancies were found in the audited invariants:

- metadata, manifest, and raw split counts agree;
- all six algorithms occur once per scenario;
- all CSV files were read successfully;
- split-specific summary row counts match their filtered raw rows;
- paired counts equal the number of matching scenario IDs within each split;
- hybrid sequences and switch counts match stage records.

## 8. Full-benchmark readiness

The output generation and audit plumbing are structurally ready: files are
complete, split-specific, readable, and internally consistent. The pilot is not
ready to serve as an unqualified accuracy benchmark for solver comparison,
because poorly conditioned cases show implausibly large position and clock
errors despite convergence statuses and very small residual RMS. This is an
audit finding for review, not an automatic algorithm correction. The full
2,000–3,000-scenario benchmark was not launched.
