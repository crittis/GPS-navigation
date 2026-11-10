# Phase 6.5B Pilot Audit

No source implementation was changed for this execution. The original
[pilot](../pilot/) was preserved.

## 1. First-pilot limitation

`generate_benchmark_scenarios()` constructs the ordered Cartesian product:

```text
(satellite_count, noise, position_perturbation, geometry, outlier_rate)
```

It assigns combination `index % len(combinations)`. The first pilot had 96
configured combinations but only 48 scenarios, so it stopped halfway through
the ordered product. The first satellite-count block contains 48 combinations,
which explains why the first pilot contained only 4-satellite scenarios. No
generator defect was found: increasing the count to 192 produces two complete
cycles and is sufficient to exercise both satellite-count groups.

## 2. Pilot configuration and execution

The requested configuration was used through the existing
`BenchmarkConfig`/`run_benchmark` interface:

| Parameter | Value |
|---|---|
| Scenario count | 192 |
| Seed | 20261110 |
| Test fraction | 0.25 |
| Satellite counts | 4, 8 |
| Noise standard deviations (m) | 0.3, 3.0 |
| Position perturbations (m) | 10.0, 500.0 |
| Geometry categories | good, moderate, poor, rank_deficient |
| Outlier rates | 0.0, 0.5, 1.0 |
| Clock-bias perturbation | Existing default, 100.0 m |
| Outlier magnitude | Existing default, 1000.0 m |
| Require both splits | true |

Exact command:

```powershell
D:\MyProjects\DAAPRO2\.venv\Scripts\python.exe -c "from gps_hybrid.benchmarking import BenchmarkConfig, run_benchmark; c=BenchmarkConfig(scenario_count=192, seed=20261110, test_fraction=0.25, satellite_counts=(4,8), noise_stds_m=(0.3,3.0), position_perturbations_m=(10.0,500.0), geometry_categories=('good','moderate','poor','rank_deficient'), outlier_rates=(0.0,0.5,1.0), require_both_splits=True); r=run_benchmark(c, output_dir=r'results/pilot_v2'); print('scenarios=',len(r.manifest_rows),'runs=',len(r.run_rows),'stages=',len(r.stage_rows)); print('split_counts=',r.metadata['split_counts'])"
```

Execution produced 192 scenarios, 1,152 per-run rows, and 1,031 per-stage
rows. The output directory did not exist before execution and was not
overwritten.

## 3. Counts and coverage

| Split | Scenarios | Algorithm rows | Manifest satellite counts |
|---|---:|---:|---|
| Development | 144 | 864 | 4: 71, 8: 73 |
| Test | 48 | 288 | 4: 25, 8: 23 |
| Total | 192 | 1,152 | 4: 96, 8: 96 |

There are 96 configured parameter combinations and exactly two generated
scenarios per combination. The generated data contains all 96 combinations.
The development split contains all 96 combinations; the test split contains 48
combinations and reports 48 missing combinations in metadata. This is expected
because only 48 of the 192 scenarios are held out.

All configured factor levels were represented in both splits:

- satellite counts: 4 and 8;
- noise: 0.3 and 3.0 m;
- position perturbations: 10.0 and 500.0 m;
- requested geometry: all four categories;
- configured outlier rates: 0.0, 0.5, and 1.0.

Realized outlier condition counts were retained separately from configured
rates. Both no-outlier, partial-outlier, and all-observation-outlier cases were
present in the raw manifest.

## 4. Structural audit

- Metadata, manifest, and raw split counts agree: 144/48 scenarios and
  864/288 algorithm rows.
- All six algorithms occur exactly once per scenario.
- There are zero duplicate `(scenario_id, algorithm)` keys.
- Every scenario has six algorithm rows.
- Missing metrics remain blank. There are 52 rows without position error or
  final residual RMS, and 148 without solver-reported residual RMS.
- Status totals are: 712 converged, 166 rank-deficient, 118
  unobservable-geometry, 48 unobservable, 102 max-iterations, 4 numerical
  failures, and 2 line-search failures.
- Development and test summary files are separate. Each overall summary has
  144 or 48 rows per algorithm, respectively.
- Paired ablations report 144 available pairs in development and 48 in test for
  each comparison. Pairing uses matching scenario IDs within the same split.
- Hybrid solver sequences and switch counts matched the corresponding stage
  records for both hybrid algorithms.
- All generated CSV files and `experiment_metadata.json` were read
  successfully.

## 5. Algorithm-level status results

| Split | Algorithm | Status counts | Large-error converged rows |
|---|---|---|---:|
| Development | GN | 87 converged, 56 rank-deficient, 1 max-iterations | 25 |
| Development | LM | 84 converged, 35 rank-deficient, 23 max-iterations, 2 numerical-failure | 7 |
| Development | BFGS | 84 converged, 35 rank-deficient, 25 max-iterations | 6 |
| Development | Initial selection only | 86 converged, 35 unobservable, 23 max-iterations | 7 |
| Development | Fixed-GN adaptive | 87 converged, 56 unobservable-geometry, 1 max-iterations | 25 |
| Development | Full hybrid | 88 converged, 35 unobservable-geometry, 21 max-iterations | 9 |
| Test | GN | 33 converged, 14 rank-deficient, 1 line-search-failed | 7 |
| Test | LM | 31 converged, 13 rank-deficient, 2 max-iterations, 2 numerical-failure | 4 |
| Test | BFGS | 33 converged, 13 rank-deficient, 2 max-iterations | 4 |
| Test | Initial selection only | 33 converged, 13 unobservable, 2 max-iterations | 4 |
| Test | Fixed-GN adaptive | 33 converged, 14 unobservable-geometry, 1 line-search-failed | 7 |
| Test | Full hybrid | 33 converged, 13 unobservable-geometry, 2 max-iterations | 4 |

## 6. Large-error convergence investigation

The threshold was position error greater than 1,000,000 m. The first pilot had
43 affected rows across 12 scenarios. The new pilot has 109 affected rows
across 34 scenarios.

The following compact tables enumerate every affected scenario. Where multiple
algorithms share a scenario, the `algorithms` field lists every affected
algorithm and metric ranges cover those affected rows. The authoritative
per-algorithm values remain in each pilot's raw `per_run_results.csv`.

### First pilot (`results/pilot`)

| Scenario | Split | Algorithms | Sat | Geometry requested/verified | Condition | PDOP | Position error range (m) | Clock error range (m) | RMS range (m) | Iterations | Sequence(s) | Perturbation | Outliers |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| 00004 | dev | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.73e6 | 1.73e6 | 1.86e-9..1.91e-7 | 4..42 | GN/LM/BFGS | 10 | partial |
| 00005 | dev | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.19e6 | 1.18e6 | 1.63e-9..2.34e-7 | 4..42 | GN/LM/BFGS | 10 | all |
| 00006 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 4.14e6 | 4.14e6 | 2.08e-9 | 4 | GN | 10 | none |
| 00008 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.32e6 | 2.32e6 | 2.13e-9 | 4 | GN | 10 | all |
| 00018 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 1.34e7 | 1.34e7 | 1.86e-9 | 6 | GN | 500 | none |
| 00020 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 1.86e-9 | 17 | GN | 500 | all |
| 00029 | dev | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 1.61e-9..2.65e-6 | 4..42 | GN/LM/BFGS | 10 | all |
| 00031 | test | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 2.63e-9 | 15 | GN | 10 | partial |
| 00041 | test | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 1.86e-9..2.66e-6 | 5..41 | GN/LM/BFGS | 500 | all |
| 00042 | test | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 1.81e7 | 1.81e7 | 2.63e-9 | 6 | GN | 500 | none |
| 00043 | dev | GN, LM, initial-selection-only, fixed-GN, full-hybrid | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.39e6..2.20e7 | 3.19e5..2.20e7 | 1.86e-9..248.6 | 8..98 | GN/LM | 500 | partial |
| 00044 | test | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 1.77e7 | 1.77e7 | 0 | 7 | GN | 500 | all |

### New pilot (`results/pilot_v2`)

| Scenario | Split | Algorithms | Sat | Geometry requested/verified | Condition | PDOP | Position error range (m) | Clock error range (m) | RMS range (m) | Iterations | Sequence(s) | Perturbation | Outliers |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| 00004 | test | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.19e6 | 1.18e6 | 1.76e-9..3.12e-9 | 4..45 | GN/LM/BFGS | 10 | partial |
| 00005 | dev | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 6.98e-10..3.21e-8 | 4..44 | GN/LM/BFGS | 10 | all |
| 00007 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 0 | 9 | GN | 10 | partial |
| 00008 | test | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 0 | 8 | GN | 10 | all |
| 00018 | test | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 6.21e6 | 6.21e6 | 9.31e-10 | 5 | GN | 500 | none |
| 00019 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 0 | 20 | GN | 500 | partial |
| 00028 | dev | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 3.03e-9..2.64e-6 | 5..48 | GN/LM/BFGS | 10 | partial |
| 00030 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 1.55e7 | 1.55e7 | 1.32e-9 | 6 | GN | 10 | none |
| 00031 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 1.25e7 | 1.25e7 | 3.23e-9 | 6 | GN | 10 | partial |
| 00040 | dev | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 2.33e-9..2.65e-6 | 4..46 | GN/LM/BFGS | 500 | partial |
| 00044 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 2.63e-9 | 10 | GN | 500 | all |
| 00054 | dev | GN, fixed-GN | 8 | poor/poor | 2.00e8 | 3.53e7 | 2.68e6 | 2.68e6 | 0.0478 | 4 | GN | 10 | none |
| 00055 | dev | full-hybrid | 8 | poor/poor | 2.13e8 | 3.77e7 | 2.42e6 | 3.23e5 | 248.7 | 120 | LM→BFGS | 10 | partial |
| 00066 | dev | GN, fixed-GN | 8 | poor/poor | 2.19e8 | 3.87e7 | 4.21e6 | 4.21e6 | 0.110 | 4 | GN | 500 | none |
| 00067 | dev | GN, fixed-GN, full-hybrid | 8 | poor/poor | 2.00e8 | 3.53e7 | 2.37e6..2.20e7 | 2.40e5..2.20e7 | 13.0..253.0 | 7..102 | GN or LM→BFGS | 500 | partial |
| 00090 | dev | GN, fixed-GN | 8 | poor/poor | 1.92e8 | 3.40e7 | 1.97e7 | 1.97e7 | 0.581 | 7 | GN | 500 | none |
| 00100 | dev | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.73e6 | 1.73e6 | 2.46e-9..4.59e-9 | 4..42 | GN/LM/BFGS | 10 | partial |
| 00101 | test | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 2.83e6 | 2.82e6 | 2.68e-9..2.80e-8 | 4..37 | GN/LM/BFGS | 10 | all |
| 00102 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 3.80e6 | 3.80e6 | 2.98e-9 | 4 | GN | 10 | none |
| 00104 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 0 | 8 | GN | 10 | all |
| 00114 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 1.03e7 | 1.03e7 | 9.31e-10 | 8 | GN | 500 | none |
| 00124 | test | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.18e6 | 1.18e6 | 0..2.31e-7 | 5..39 | GN/LM/BFGS | 10 | partial |
| 00125 | test | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.33e6 | 1.32e6 | 2.29e-9..2.65e-6 | 4..42 | GN/LM/BFGS | 10 | all |
| 00128 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 1.86e-9 | 10 | GN | 10 | all |
| 00136 | dev | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 3.30e-9..2.65e-6 | 4..45 | GN/LM/BFGS | 500 | partial |
| 00137 | dev | GN, LM, BFGS, initial-selection-only, fixed-GN, full-hybrid | 4 | moderate/moderate | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 1.99e-9..2.68e-6 | 4..41 | GN/LM/BFGS | 500 | all |
| 00138 | dev | GN, fixed-GN | 4 | poor/poor | 2.00e8 | 5.00e7 | 2.06e7 | 2.06e7 | 0 | 7 | GN | 500 | none |
| 00150 | dev | GN, fixed-GN | 8 | poor/poor | 1.99e8 | 3.52e7 | 1.26e7 | 1.26e7 | 0.179 | 5 | GN | 10 | none |
| 00151 | dev | GN, fixed-GN | 8 | poor/poor | 1.90e8 | 3.35e7 | 2.20e7 | 2.20e7 | 469.1 | 17 | GN | 10 | partial |
| 00174 | dev | GN, fixed-GN | 8 | poor/poor | 2.01e8 | 3.56e7 | 1.63e7 | 1.63e7 | 1.28 | 5 | GN | 10 | none |
| 00175 | dev | GN, fixed-GN | 8 | poor/poor | 1.89e8 | 3.34e7 | 2.20e7 | 2.19e7 | 451.6 | 17 | GN | 10 | partial |
| 00176 | test | GN, fixed-GN | 8 | poor/poor | 2.05e8 | 3.62e7 | 2.19e7 | 2.19e7 | 697.8 | 52 | GN | 10 | all |
| 00186 | dev | GN, fixed-GN | 8 | poor/poor | 1.92e8 | 3.40e7 | 8.86e6 | 8.86e6 | 1.08 | 6 | GN | 500 | none |
| 00188 | dev | LM, initial-selection-only, full-hybrid | 8 | poor/poor | 1.98e8 | 3.50e7 | 1.03e6 | 3.89e4 | 998.5 | 97 | LM | 500 | all |

## 7. Four- versus eight-satellite comparison

The second pilot corrected the first-pilot coverage limitation: both satellite
counts occur in both splits. The large-error rows remain dominated by poor
geometry in both groups.

- Four-satellite rows account for most affected cases, including the moderate
  geometry cases with approximately 1.18–2.83 million m position error and
  residual RMS near `1e-9` to `1e-6` m.
- Eight-satellite rows also show the issue in poor geometry. Their affected
  errors range from approximately 2.4 million to 219 million m, with RMS from
  sub-metre values to hundreds of metres depending on the solver outcome.
- The 8-satellite cases do not eliminate the issue; they change its magnitude
  and expose additional hybrid/LM transition cases.
- Both 10 m and 500 m initial position perturbations appear among affected
  rows. No-outlier, partial-outlier, and all-outlier conditions also appear.
- GN and fixed-GN adaptive account for many affected rows, but LM, BFGS,
  initial-selection-only, and full hybrid are also affected in moderate or
  transition cases. This is an association in this pilot, not evidence of
  causation.

## 8. Does the low-residual/high-position-error issue persist?

Yes. It persists in the representative 192-scenario pilot and is not explained
by the first pilot's missing 8-satellite group. It is strongly associated in
these data with the synthetic poor/moderate geometry categories, very high
condition numbers (roughly `5e3` for moderate cases and `1e8` for poor cases),
and very high PDOP. The pilot also shows affected rows under multiple noise,
perturbation, and outlier conditions, so those factors cannot be isolated as a
cause from this execution alone.

## 9. Remaining methodological concerns

- Solver convergence status and low residual RMS are not sufficient accuracy
  checks for these ill-conditioned geometries.
- The synthetic geometry categories are intentionally extreme and are not a
  claim about real satellite visibility or orbital geometry.
- The test split has only 48 scenarios and 48 of 96 full combinations; it is
  factor-representative but not full-combination-complete.
- The pilot emitted the existing LM overflow warning during damping
  calculations. It did not prevent output generation, but it remains a warning
  for review.
- Matplotlib was unavailable, so optional plotting was not executed.

## 10. Full-benchmark readiness recommendation

The scenario coverage and output-audit plumbing are ready for further review,
and the requested 192-scenario pilot successfully exercised both satellite
counts. The full benchmark is **not recommended yet for unqualified numerical
accuracy comparison**, because the low-residual/high-position-error behavior
persists in both 4- and 8-satellite poor/moderate cases. No algorithms,
thresholds, or held-out results were tuned or changed. The 2,000–3,000-scenario
experiment was not launched.
