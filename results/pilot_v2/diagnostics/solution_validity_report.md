# Phase 6.5C — GPS Solution Validity and Numerical Diagnosis

## 1. Executive summary

This diagnostic-only review found a confirmed low-residual/large-position-error failure mode. In the first pilot, 43 converged rows exceeded 1,000,000 m position error, across 12 distinct scenarios. In pilot v2, 109 converged rows exceeded the threshold, across 34 scenarios. The first pilot contains only four-satellite scenarios; pilot v2 contains both four- and eight-satellite scenarios.

The four-satellite case demonstrates a distinct low-residual solution: for `scenario-00007`, Gauss–Newton reached an exact-fit state with objective approximately zero and position error approximately 21,990,956 m, while the objective at truth was approximately 1,000,014. This is expected to be possible with four observations and four unknowns, especially with outliers: a zero residual only proves that the four equations fit the estimated state, not that the state is the physical receiver state.

The eight-satellite cases show that the issue is not limited to exact four-equation interpolation. Affected eight-satellite rows occur primarily in the intentionally poor, extremely ill-conditioned geometry. Some have small residuals but multi-million-metre errors; others retain substantial residuals and are nevertheless marked converged under the current numerical stopping semantics.

No source implementation was changed. The LM overflow warning is reproducible in isolation. It occurs while forming the augmented damping block, after repeated rejected steps have made the damping parameter enormous. The affected run is returned as a numerical failure, not as convergence.

These findings do not establish general solver robustness, and they are not a recommendation to tune thresholds using the held-out split. The pilots are not ready to justify a full benchmark without a review of solution-validity policy and the synthetic geometry design.

## 2. Source-code trace

Line references are to the inspected working tree.

| Concern | Location and observed implementation |
|---|---|
| State and units | [`gps_model.py`](../../src/gps_hybrid/gps_model.py:1) documents ECEF coordinates in metres. The four-vector is `[x, y, z, b]`; `b` is receiver clock bias expressed in metres. [`scenario.py`](../../src/gps_hybrid/scenario.py:90) constructs the truth state with the clock bias as the fourth component. |
| Residual | [`gps_model.py`](../../src/gps_hybrid/gps_model.py:32) evaluates `rho_i - ||s_i-p|| - b`, in metres. |
| Jacobian | [`gps_model.py`](../../src/gps_hybrid/gps_model.py:38) uses the line-of-sight position derivatives and a `-1` clock-bias column. |
| Objective | [`gps_model.py`](../../src/gps_hybrid/gps_model.py:45) uses `0.5 * r.T @ r`. |
| Initial state | [`scenario.py`](../../src/gps_hybrid/scenario.py:108) independently perturbs the three ECEF position components and clock bias. Benchmark reconstruction uses [`benchmarking.py`](../../src/gps_hybrid/benchmarking.py:250). |
| Position constraints | None are imposed by `validate_scenario` or any solver. [`scenario.py`](../../src/gps_hybrid/scenario.py:121) validates shapes, finiteness, satellite count, and ranges, but does not constrain an estimated state to the Earth or to an altitude interval. |
| Convergence | [`solvers/base.py`](../../src/gps_hybrid/solvers/base.py:156) preserves a gradient check and combines it with the separately scaled step/objective checks. The position scales are `6.4e6` m and the clock scale is `1000` m ([`solvers/base.py`](../../src/gps_hybrid/solvers/base.py:173)). The result is `grad_ok or (step_ok and f_ok)`; a step alone is insufficient. |
| GN numerical solve | [`solvers/gauss_newton.py`](../../src/gps_hybrid/solvers/gauss_newton.py:27) uses direct `numpy.linalg.lstsq`; its line search and stopping calls are at lines 32–61. |
| LM numerical solve | [`solvers/levenberg_marquardt.py`](../../src/gps_hybrid/solvers/levenberg_marquardt.py:27) forms the damping and scaling, solves the augmented least-squares system at line 35, computes the gain ratio at lines 41–45, and grows damping after rejection at lines 62–76. |
| BFGS numerical solve | [`solvers/bfgs.py`](../../src/gps_hybrid/solvers/bfgs.py:28) uses a direct Hessian solve and curvature safeguards. |
| Hybrid geometry and acceptance | [`hybrid.py`](../../src/gps_hybrid/hybrid.py:295) diagnoses initial geometry, [`hybrid.py`](../../src/gps_hybrid/hybrid.py:367) diagnoses final geometry, and [`hybrid.py`](../../src/gps_hybrid/hybrid.py:377) accepts convergence only after the final observability check. |
| Final benchmark metrics | [`benchmarking.py`](../../src/gps_hybrid/benchmarking.py:511) independently recomputes residual RMS from a finite final state; position and clock errors are formed in [`benchmarking.py`](../../src/gps_hybrid/benchmarking.py:596). |
| Geometry diagnostics | [`geometry.py`](../../src/gps_hybrid/geometry.py:100) evaluates the existing GPS Jacobian at the supplied estimate, computes SVD rank/condition, and derives ENU PDOP. Benchmark initial geometry is evaluated at [`benchmarking.py`](../../src/gps_hybrid/benchmarking.py:273). |

The synthetic scenario generator creates satellite directions from templates in [`scenario.py`](../../src/gps_hybrid/scenario.py:79). These are synthetic directions, not claims about real satellite visibility or orbital geometry.

## 3. Pilot data and reconstruction method

The inspected artifacts were:

- [`results/pilot/raw/per_run_results.csv`](../raw/per_run_results.csv) and its scenario manifest and stage CSVs.
- [`results/pilot_v2/raw/per_run_results.csv`](../raw/per_run_results.csv) and its scenario manifest and stage CSVs.

Scenarios were reconstructed from their manifest values using `generate_benchmark_scenario`, preserving the saved seed, split, satellite count, noise, position perturbation, requested geometry, outlier rate, clock perturbation, and the approved default outlier magnitude. Residuals, objectives, geodetic coordinates, and diagnostic Jacobians were then evaluated outside the solver. Saved raw CSVs do not contain singular-value vectors, initial/final ECEF states, or observation vectors; those values are therefore marked unavailable in the tables below unless explicitly reconstructed for a representative case.

The threshold used for this report is `position_error_m > 1,000,000`. “Converged” means the raw row’s `is_converged` flag, not an independent claim of physical correctness.

## 4. Large-error scenario tables

The following tables contain one row per distinct affected scenario. Metric ranges cover the affected converged algorithms in that scenario; the raw per-run CSV remains authoritative for individual algorithm rows. `algorithms` lists only affected converged rows. The condition number and PDOP are the final diagnostic values recorded for the run; the scenario manifest also records initial values. Singular-value vectors were not persisted in the pilot artifacts and are therefore unavailable per row.

### First pilot

There were 43 affected rows in 12 scenarios. All had four observations.

| scenario | split | requested/verified | outlier condition | algorithms | condition | PDOP | position error m | clock error m | residual RMS m | iterations | sequence |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 00004 | development | moderate/moderate | partial | GN, LM, BFGS, selection, fixed-GN, hybrid | 5.01e3 | 1.25e3 | 1.73e6 | 1.73e6 | 1.86e-9..1.91e-7 | 4..42 | GN |
| 00005 | development | moderate/moderate | all | GN, LM, BFGS, selection, fixed-GN, hybrid | 5.01e3 | 1.25e3 | 1.19e6 | 1.18e6 | 1.63e-9..2.34e-7 | 4..42 | GN |
| 00006 | development | poor/poor | none | GN, fixed-GN | 2.00e8 | 5.00e7 | 4.14e6 | 4.14e6 | 2.08e-9 | 4 | GN |
| 00008 | development | poor/poor | all | GN, fixed-GN | 2.00e8 | 5.00e7 | 2.32e6 | 2.32e6 | 2.13e-9 | 4 | GN |
| 00018 | development | poor/poor | none | GN, fixed-GN | 2.00e8 | 5.00e7 | 1.34e7 | 1.34e7 | 1.86e-9 | 6 | GN |
| 00020 | development | poor/poor | all | GN, fixed-GN | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 1.86e-9 | 17 | GN |
| 00029 | development | moderate/moderate | all | GN, LM, BFGS, selection, fixed-GN, hybrid | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 1.61e-9..2.65e-6 | 4..42 | GN |
| 00031 | test | poor/poor | partial | GN, fixed-GN | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 2.63e-9 | 15 | GN |
| 00041 | test | moderate/moderate | all | GN, LM, BFGS, selection, fixed-GN, hybrid | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 1.86e-9..2.66e-6 | 5..41 | GN |
| 00042 | test | poor/poor | none | GN, fixed-GN | 2.00e8 | 5.00e7 | 1.81e7 | 1.81e7 | 2.63e-9 | 6 | GN |
| 00043 | development | poor/poor | partial | GN, LM, selection, fixed-GN, hybrid | 2.00e8 | 5.00e7 | 2.39e6..2.20e7 | 3.19e5..2.20e7 | 1.86e-9..249 | 8..98 | GN |
| 00044 | test | poor/poor | all | GN, fixed-GN | 2.00e8 | 5.00e7 | 1.77e7 | 1.77e7 | 0 | 7 | GN |

### Pilot v2

There were 109 affected rows in 34 scenarios. Four-satellite scenarios account for 22 affected scenarios and eight-satellite scenarios for 12 affected scenarios.

| scenario | split | sat | requested/verified | outlier condition | algorithms | condition | PDOP | position error m | clock error m | RMS m | iterations | sequence |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 00004 | test | 4 | moderate/moderate | partial | GN,LM,BFGS,selection,fixed-GN,hybrid | 5.01e3 | 1.25e3 | 1.19e6 | 1.18e6 | 1.76e-9..3.12e-9 | 4..45 | GN |
| 00005 | development | 4 | moderate/moderate | all | GN,LM,BFGS,selection,fixed-GN,hybrid | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 6.98e-10..3.21e-8 | 4..44 | GN |
| 00007 | development | 4 | poor/poor | partial | GN,fixed-GN | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 0 | 9 | GN |
| 00008 | test | 4 | poor/poor | all | GN,fixed-GN | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 0 | 8 | GN |
| 00018 | test | 4 | poor/poor | none | GN,fixed-GN | 2.00e8 | 5.00e7 | 6.21e6 | 6.21e6 | 9.31e-10 | 5 | GN |
| 00019 | development | 4 | poor/poor | partial | GN,fixed-GN | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 0 | 20 | GN |
| 00028 | development | 4 | moderate/moderate | partial | GN,LM,BFGS,selection,fixed-GN,hybrid | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 3.03e-9..2.64e-6 | 5..48 | GN |
| 00030 | development | 4 | poor/poor | none | GN,fixed-GN | 2.00e8 | 5.00e7 | 1.55e7 | 1.55e7 | 1.32e-9 | 6 | GN |
| 00031 | development | 4 | poor/poor | partial | GN,fixed-GN | 2.00e8 | 5.00e7 | 1.25e7 | 1.25e7 | 3.23e-9 | 6 | GN |
| 00040 | development | 4 | moderate/moderate | partial | GN,LM,BFGS,selection,fixed-GN,hybrid | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 2.33e-9..2.65e-6 | 4..46 | GN |
| 00044 | development | 4 | poor/poor | all | GN,fixed-GN | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 2.63e-9 | 10 | GN |
| 00054 | development | 8 | poor/poor | none | GN,fixed-GN | 2.00e8 | 3.53e7 | 2.68e6 | 2.68e6 | .0478 | 4 | GN |
| 00055 | development | 8 | poor/poor | partial | full hybrid | 2.13e8 | 3.77e7 | 2.42e6 | 3.23e5 | 249 | 120 | LM→BFGS |
| 00066 | development | 8 | poor/poor | none | GN,fixed-GN | 2.19e8 | 3.87e7 | 4.21e6 | 4.21e6 | .11 | 4 | GN |
| 00067 | development | 8 | poor/poor | partial | GN,fixed-GN,full hybrid | 2.00e8 | 3.53e7 | 2.37e6..2.20e7 | 2.40e5..2.20e7 | 13..253 | 7..102 | GN |
| 00090 | development | 8 | poor/poor | none | GN,fixed-GN | 1.92e8 | 3.40e7 | 1.97e7 | 1.97e7 | .581 | 7 | GN |
| 00100 | development | 4 | moderate/moderate | partial | GN,LM,BFGS,selection,fixed-GN,hybrid | 5.01e3 | 1.25e3 | 1.73e6 | 1.73e6 | 2.46e-9..4.59e-9 | 4..42 | GN |
| 00101 | test | 4 | moderate/moderate | all | GN,LM,BFGS,selection,fixed-GN,hybrid | 5.01e3 | 1.25e3 | 2.83e6 | 2.82e6 | 2.68e-9..2.80e-8 | 4..37 | GN |
| 00102 | development | 4 | poor/poor | none | GN,fixed-GN | 2.00e8 | 5.00e7 | 3.80e6 | 3.80e6 | 2.98e-9 | 4 | GN |
| 00104 | development | 4 | poor/poor | all | GN,fixed-GN | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 0 | 8 | GN |
| 00114 | development | 4 | poor/poor | none | GN,fixed-GN | 2.00e8 | 5.00e7 | 1.03e7 | 1.03e7 | 9.31e-10 | 8 | GN |
| 00124 | test | 4 | moderate/moderate | partial | GN,LM,BFGS,selection,fixed-GN,hybrid | 5.01e3 | 1.25e3 | 1.18e6 | 1.18e6 | 0..2.31e-7 | 5..39 | GN |
| 00125 | test | 4 | moderate/moderate | all | GN,LM,BFGS,selection,fixed-GN,hybrid | 5.01e3 | 1.25e3 | 1.33e6 | 1.32e6 | 2.29e-9..2.65e-6 | 4..42 | GN |
| 00128 | development | 4 | poor/poor | all | GN,fixed-GN | 2.00e8 | 5.00e7 | 2.20e7 | 2.20e7 | 1.86e-9 | 10 | GN |
| 00136 | development | 4 | moderate/moderate | partial | GN,LM,BFGS,selection,fixed-GN,hybrid | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 3.30e-9..2.65e-6 | 4..45 | GN |
| 00137 | development | 4 | moderate/moderate | all | GN,LM,BFGS,selection,fixed-GN,hybrid | 5.01e3 | 1.25e3 | 1.33e6 | 1.33e6 | 1.99e-9..2.68e-6 | 4..41 | GN |
| 00138 | development | 4 | poor/poor | none | GN,fixed-GN | 2.00e8 | 5.00e7 | 2.06e7 | 2.06e7 | 0 | 7 | GN |
| 00150 | development | 8 | poor/poor | none | GN,fixed-GN | 1.99e8 | 3.52e7 | 1.26e7 | 1.26e7 | .179 | 5 | GN |
| 00151 | development | 8 | poor/poor | partial | GN,fixed-GN | 1.90e8 | 3.35e7 | 2.20e7 | 2.20e7 | 469 | 17 | GN |
| 00174 | development | 8 | poor/poor | none | GN,fixed-GN | 2.01e8 | 3.56e7 | 1.63e7 | 1.63e7 | 1.28 | 5 | GN |
| 00175 | development | 8 | poor/poor | partial | GN,fixed-GN | 1.89e8 | 3.34e7 | 2.20e7 | 2.19e7 | 452 | 17 | GN |
| 00176 | test | 8 | poor/poor | all | GN,fixed-GN | 2.05e8 | 3.62e7 | 2.19e7 | 2.19e7 | 698 | 52 | GN |
| 00186 | development | 8 | poor/poor | none | GN,fixed-GN | 1.92e8 | 3.40e7 | 8.86e6 | 8.86e6 | 1.08 | 6 | GN |
| 00188 | development | 8 | poor/poor | all | LM,selection,full hybrid | 1.98e8 | 3.50e7 | 1.03e6 | 3.89e4 | 998 | 97 | LM |

## 5. Representative four-satellite investigation

`scenario-00007` in pilot v2 is a four-satellite, poor-geometry, partial-outlier case. Its truth state was reconstructed as:

```text
[-2702593.0646, -4325052.9993, 3817405.1966, 75.0]
```

The GN final state was:

```text
[15946681.0075, -15978667.6321, 3818943.9881, 21990898.6018]
```

The truth geodetic coordinates were latitude `0.6457718 rad`, longitude `-2.1293017 rad`, altitude `20.0 m`. The final estimate was latitude `0.1678903 rad`, longitude `-0.7864001 rad`, altitude approximately `16,517,846 m`. Position error was `21,990,956 m`; clock-bias error was `21,990,824 m`.

At truth, the residual vector was approximately:

```text
[-0.13799, 0.23595, 1000.07747, 999.93660]
```

and the objective was `1,000,014.11596`. At the GN final state, the residual vector was numerically zero and the objective was zero. The independently recomputed residual RMS was zero.

The initial Jacobian singular values were approximately:

```text
[2.8284271071, 3.0000019056e-4, 1.0000006527e-4, 1.4142153841e-8]
```

with rank 4, condition number `1.9999974e8`, and PDOP `4.9999937e7`. At the final estimate, the singular values became approximately `[2.7416964, 0.6485591, 0.2397442, 0.0706740]`, condition number `38.7935`, and PDOP `11.2389`. Thus the final local Jacobian looked much better conditioned even though the final state was physically implausible; final local conditioning is not a validity certificate.

Multiple diagnostic GN initial states for the same observations (`truth`, saved initial, truth plus a 1,000,000 m Cartesian perturbation, an origin-like state, and a far state) all reached the same or numerically identical wrong exact-fit branch. This experimentally demonstrates a distinct low-residual solution relative to the truth for this observation set; it is not merely suspected. The corresponding LM trials did not produce the same exact-fit branch, which also shows solver dependence in this ill-conditioned case.

## 6. Eight-satellite investigation

The eight-satellite failures are distinct from the four-equation interpolation argument. A representative no-outlier case, `scenario-00054`, had truth objective `0.04827` and final GN objective `0.00914`; its final residual RMS was about `0.0478 m`, but the position error was `2,677,514 m` and the estimated altitude was approximately `539,157 m`. Its initial singular values were approximately `[4.0000000, 4.2533e-4, 1.4221e-4, 2.0043e-8]`, condition `1.9957e8`, PDOP `3.5279e7`; final condition remained `1.5395e8`, PDOP `2.7214e7`.

Representative `scenario-00055` had eight satellites, poor geometry, partial outliers, and a full-hybrid sequence `Levenberg-Marquardt → BFGS`. Truth objective was `1,000,249.4878`; final objective was `247,366.3774`, with residual RMS about `249 m`, position error `2,418,729 m`, and clock-bias error `322,852 m`. Its final residuals remained distributed at approximately 227–283 m in magnitude, so this is not an exact-fit four-equation phenomenon. The final condition was approximately `2.0975e8` and PDOP `3.7079e7`.

Across the v2 table, eight-satellite affected cases are concentrated in poor requested/verified geometry and occur with no, partial, and all-observation outlier conditions. They occur at both 10 m and 500 m initial position perturbations. The raw results show GN/fixed-GN most often, with LM/BFGS and hybrid transitions in some cases. This is evidence of association with severe conditioning and the synthetic poor-geometry construction, not proof of causation.

## 7. Physical-validity findings

The generated truth receiver has geodetic altitude `20 m`. Estimated altitudes in affected examples include approximately `136,258 m` (`scenario-00004`), `16,517,846 m` (`scenario-00007`), `539,157 m` (`scenario-00054`), and `1,943,990 m` (`scenario-00055`). With the WGS84 conversion used by [`coordinates.py`](../../src/gps_hybrid/coordinates.py:37), distance from the reference ellipsoid is represented diagnostically by the absolute geodetic altitude.

For this report only, I use a generous Earth-proximate plausibility bound of `-1,000 m <= estimated altitude <= 100,000 m`. This is not a solver constraint and is not a claim that every receiver is at sea level. It is a diagnostic bound chosen to make multi-hundred-kilometre and multi-million-kilometre excursions visibly implausible relative to the generated truth altitude. The representative affected estimates above fail this bound. Small residual RMS therefore does not establish physical validity.

## 8. Convergence semantics

For `scenario-00007` GN, the final `has_converged` call returned true at iteration 9. Reconstructing the predicate from [`solvers/base.py`](../../src/gps_hybrid/solvers/base.py:156) gave:

```text
gradient ratio = 0.0                    <= gradient tolerance: true
scaled step    = 4.97e-7                <= scaled step limit: true
objective reduction = 3.27e-9           <= relative objective test: false
```

The result was therefore accepted through the gradient branch (and also had a small scaled step), not because a small step alone was sufficient. The final Jacobian was full rank. It was not physically plausible, and the convergence predicate does not test physical plausibility or agreement with truth. In general, `converged` in this implementation means numerical stationarity according to the configured residual/Jacobian criteria plus the final geometry checks in the hybrid controller; it does not mean a physically valid receiver solution.

For the representative moderate four-satellite case `scenario-00004`, GN similarly returned true with gradient ratio approximately `2.79e-9` and a small scaled step, while the objective-reduction check was false. This confirms that the gradient branch can accept a state whose residual is numerically flat around a wrong branch.

## 9. LM overflow warning

The warning operation is exactly [`levenberg_marquardt.py`](../../src/gps_hybrid/solvers/levenberg_marquardt.py:33):

```python
np.diag(np.sqrt(damping * scales))
```

The isolated reconstruction used v2 `scenario-00064`, the eight-satellite moderate partial-outlier case whose raw LM row is a numerical failure. Re-running the approved solver with warnings captured produced:

```text
RuntimeWarning: overflow encountered in multiply
status: numerical_failure
reason: SVD did not converge in Linear Least Squares
iterations: 48
function evaluations: 49
history records: 48
maximum recorded damping: 3.100429997645449e294
```

Repeated rejected steps recorded `gain_ratio=nan` and drove damping from approximately `7.95e4` at the first observed rejected record to approximately `3.10e294`. The logged input to the `sqrt` call was finite through the preceding iteration and then became `inf` for the four-element damping block because `damping * scales` overflowed. The warning is therefore reproducible and produces non-finite intermediate values. In this case the subsequent least-squares operation raises `SVD did not converge`; the solver catches it and reports numerical failure. It was not recovered as convergence and was not silently reported as a valid estimate.

The same pattern is reflected in the raw v2 numerical-failure rows (`scenario-00064`, `00088`, `00172`, and `00181`), whose LM failure reasons include invalid gain-ratio history or the resulting numerical linear-algebra failure. No warning was suppressed or implementation behavior changed during this investigation.

## 10. Confirmed defects versus hypotheses

### Confirmed

1. Four observations can produce a numerically exact, physically wrong state. This was demonstrated experimentally for `scenario-00007`.
2. The current convergence status does not encode physical plausibility; the affected GN state passed the gradient/stationarity criterion and final rank check.
3. Large-error convergence persists with eight observations, so the issue is not exclusively the four-equation/four-unknown interpolation case.
4. Poor synthetic geometry is strongly associated with the affected eight-satellite rows: their condition numbers are approximately `1e8`–`2e8` and PDOP is approximately `3e7`–`5e7`.
5. The LM damping multiplication overflow warning is reproducible and leads to non-finite intermediate values in the isolated affected case.

### Not established

1. The pilot does not prove that outliers, initial perturbation, or a particular solver independently causes the failures; these factors co-occur with geometry strata and the scenario design.
2. The evidence does not establish that every low-residual wrong state is a second physical solution. Only the `scenario-00007` GN branch was experimentally demonstrated as a distinct exact-fit solution.
3. The pilot does not establish general algorithmic robustness or production GPS performance. Satellite directions are synthetic and are not orbital/visibility data.

## 11. Recommended next steps, ranked

1. **Highest severity:** define an explicit solution-validity policy separate from numerical convergence, including how physically implausible states and ambiguous low-residual roots are reported. This should be reviewed before interpreting benchmark convergence rates as solution correctness.
2. **High:** review the synthetic poor-geometry templates and whether their extreme condition numbers are intended for the benchmark. Keep any change separate from held-out evaluation and document it as methodology, not solver tuning.
3. **High:** add a diagnostic or result field distinguishing numerical stationarity from physically validated positioning, without silently relabeling existing solver statuses.
4. **Medium:** review LM rejection/damping safeguards and warning behavior. The current isolated case is correctly rejected as numerical failure, but repeated damping growth produces overflow before termination.
5. **Medium:** retain independent final residual recomputation and add state/altitude/Jacobian diagnostics to future benchmark artifacts so the analysis does not require scenario reconstruction.
6. **Low:** investigate broader multi-start behavior across representative strata. Any conclusions should be reported as empirical sensitivity analysis, not used to tune held-out thresholds.

## 12. Reproducibility and commands

No diagnostic script file was created. The following read-only commands were executed from `D:\MyProjects\DAAPRO2` using the project interpreter:

```powershell
Get-ChildItem results\pilot_v2\raw, results\pilot\raw
Get-Content src\gps_hybrid\solvers\levenberg_marquardt.py | Select-Object -First 110
& .venv\Scripts\python.exe -c "import csv; print(next(csv.DictReader(open('results/pilot_v2/raw/scenario_manifest.csv'))))"
```

Two inline Python diagnostics reconstructed scenarios from the manifests, reran approved solver calls without writing outputs, recomputed metrics, captured warnings, logged the LM damping-block square-root inputs, and wrapped `has_converged` to report its predicate components. The exact isolated LM command was an inline script invoked as:

```powershell
@' ...diagnostic Python body... '@ | & .venv\Scripts\python.exe -
```

It targeted `scenario-00064` and produced the warning and numerical-failure values reported above. The convergence diagnostic targeted v2 `scenario-00007` and `scenario-00004` with the approved GN solver.

No benchmark outputs were overwritten, no full 2,000–3,000-scenario experiment was launched, and no tests were modified or suppressed. Because this phase made no source-code change, no pytest run was required for the diagnostic deliverable; the commands above are the executed diagnostic validation.
