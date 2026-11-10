# GPS Solution-Validity Policy — Design Proposal

## 1. Problem statement

The current solver status answers a numerical question: did the iteration process satisfy its convergence rule (or terminate with another solver status)? It does not, by itself, answer whether the pseudorange observations fit acceptably, whether the state is observable, or whether the estimated location is inside the application’s operating region.

The Phase 6.5C pilot audit demonstrated why these distinctions matter: a four-observation/four-unknown GN solve reached a zero residual with a position error exceeding 21,000 km; poor-geometry eight-observation cases also produced large errors. Therefore a small residual and a `converged` status cannot be used as proxies for a correct receiver location.

This proposal adds a separate validator, downstream of the approved solvers and hybrid controller. It must not change solver results, solver status semantics, the GPS equations, or the hybrid switching policy.

## 2. Existing architecture assessment

The inspected implementation has useful boundaries to reuse:

- [`models.py`](../src/gps_hybrid/models.py) separates `GPSObservations` from `ScenarioTruth`. `Scenario` bundles both for controlled experiment generation, while solver calls receive observations and an initial estimate.
- [`gps_model.py`](../src/gps_hybrid/gps_model.py) owns the approved residual, Jacobian, and objective. The residual is in metres and is independently evaluable for any finite state and observation set.
- [`coordinates.py`](../src/gps_hybrid/coordinates.py) implements WGS-84 geodetic/ECEF conversion; altitude and Cartesian coordinates are in metres.
- [`geometry.py`](../src/gps_hybrid/geometry.py) already returns typed `GeometryDiagnostics`, including satellite count, numerical rank, singular values, condition number, PDOP, full-rank and insufficient-observation indicators. Geometry is estimated at a supplied receiver state and uses the existing Jacobian.
- [`solvers/base.py`](../src/gps_hybrid/solvers/base.py) defines the immutable `SolverResult` and existing statuses, including convergence, maximum iterations, line-search failure, rank deficiency, and numerical failure. Keep these unchanged.
- [`hybrid.py`](../src/gps_hybrid/hybrid.py) returns a `HybridResult` with final state, solver status, initial/final geometry, and stage/transition history. Its final observability checks are solver-controller safeguards, not a physical operating-domain test.
- [`benchmarking.py`](../src/gps_hybrid/benchmarking.py) already recomputes final residual RMS from the final finite state; it separately calculates truth-based position and clock errors for evaluation. Current run rows record initial geometry data, solver status, final residual RMS, solver-reported RMS, and truth-based error metrics.

### Recommended architectural location

Add a standalone `solution_validity.py` module alongside `geometry.py`. It should accept only:

1. `GPSObservations`;
2. the estimated state (or an explicit unavailable state);
3. the existing numerical status; and
4. a validity configuration.

It should reuse `residual`, `diagnose_geometry`, and `ecef_to_geodetic`, and return its own immutable diagnostics/result. It should not call or modify a solver, construct a `ScenarioTruth`, or import benchmark scenario types. Keeping the layer separate avoids changing approved interfaces and makes it usable by non-benchmark receiver code.

At the benchmark boundary, call this layer after `_execute_algorithm` and before building the run row. The evaluator may separately calculate truth-based accuracy; the validity function must not receive truth. Existing columns and summaries remain unchanged, with validity fields added alongside them.

## 3. Proposed result model

Prefer a small configuration plus an immutable result composed of explicit dimensions rather than one Boolean. One possible typed shape is:

```python
@dataclass(frozen=True)
class MeasurementFitDiagnostics:
    residual_rms_m: float | None
    maximum_absolute_residual_m: float | None
    residual_median_absolute_m: float | None  # optional
    residual_p95_absolute_m: float | None     # optional
    status: str                                # pass/fail/not_evaluated/not_assessable
    failure_reasons: tuple[str, ...]

@dataclass(frozen=True)
class PhysicalPlausibilityDiagnostics:
    domain: str
    latitude_rad: float | None
    longitude_rad: float | None
    altitude_m: float | None
    status: str                                # pass/fail/not_applicable/not_assessable
    failure_reasons: tuple[str, ...]

@dataclass(frozen=True)
class SolutionValidityResult:
    numerical_status: str                     # unchanged solver/controller status
    measurement_fit: MeasurementFitDiagnostics
    geometry: GeometryDiagnostics | None
    physical: PhysicalPlausibilityDiagnostics
    classification: str                       # primary summary classification
    flags: frozenset[str]                     # all applicable diagnostic outcomes
    assessment_status: Literal[
        "valid_under_configured_policy",
        "invalid_under_configured_policy",
        "incomplete_assessment",
    ]
    explanations: tuple[str, ...]
```

Names are illustrative, not a demand for this exact public API. The essential semantics are:

- Preserve the source numerical status verbatim. Do not rewrite `converged` to `valid` or turn a physical failure into a solver failure.
- Recompute residuals once from the final finite state and observations, using the approved `residual` function. Derive RMS and maximum absolute residual from that same vector. If state or observations are invalid/unavailable, retain `None`, not a fabricated zero.
- Attach the existing `GeometryDiagnostics` object, rather than recalculating and copying its rank, condition, and PDOP into parallel fields. If there are insufficient observations, preserve that explicit distinction. If Jacobian evaluation fails, retain an explanation and no geometry result.
- Physical assessment requires a finite three-coordinate ECEF position that can be converted to geodetic coordinates. It must not use residual quality or truth as a substitute for domain assessment.
- `flags` must allow simultaneous outcomes, such as `measurement_fit_failed` and `physically_implausible`. `classification` may provide a deterministic headline for tables, but must never discard the detailed flags.
- The overall assessment has exactly three semantics: `valid_under_configured_policy`, `invalid_under_configured_policy`, and `incomplete_assessment`. The numerical solver status remains a separate field and is never overwritten by this assessment.
- A complete assessment requires: a `converged` numerical status; a finite final four-state estimate; at least four valid observations; successful residual and geometry calculations; an explicit condition-number policy threshold; a configured measurement-fit acceptance threshold; and a physical-domain outcome. For `unrestricted_ecef`, the physical outcome is `not_applicable`; the assessment is complete only if configuration explicitly allows that domain check to be waived. Merely selecting unrestricted mode does not make the physical check pass.
- `valid_under_configured_policy` means all required checks were evaluated and passed. `invalid_under_configured_policy` means all required checks were evaluable and at least one failed. `incomplete_assessment` means a required check could not be evaluated, including a non-converged numerical result, invalid/unavailable estimate, geometry/residual calculation failure, or an unevaluated required fit threshold.
- A disabled fit threshold has status `not_evaluated`; it is not a pass and makes the assessment incomplete. An unrestricted ECEF domain has physical status `not_applicable`, not a physical-plausibility pass. It may be explicitly accepted as a waived dimension in configuration, but must remain distinguishable in the result and reports.
- Keep `flags` as a set of independent observations. The primary classification is only a compact headline and must never hide the numerical status or simultaneous dimension flags.

The raw `SolverResult` and `HybridResult` remain intact and remain the sources of solver accounting. The validity result is an additional assessment record.

## 4. Measurement fit and geometric observability

### Measurement fit

For the initial implementation, always expose independently recomputed final residual RMS and maximum absolute residual when a finite final state is available. Optional median/p95 absolute residuals can help distinguish a broad poor fit from a small number of outliers, but are not required initially.

Thresholds must be configurable, with separate optional limits such as:

- `max_residual_rms_m: float | None`
- `max_absolute_residual_m: float | None`

When both limits are configured, both must pass for fit status `pass`; failure of either configured limit makes fit status `fail`. An unconfigured limit is ignored. If neither limit is configured, report the statistics and set fit status to `not_evaluated`; do not treat this as passed. No default numerical acceptance thresholds are proposed. A single universal RMS threshold is not justified by the current benchmark: noise levels vary, outliers can be present, and the current geometry diagnostic assumes equal independent measurement variance rather than using a measurement covariance model. A future statistically normalized/weighted residual test must require an explicit measurement-noise or covariance model before implementation. Low residual must never imply correct position.

### Geometric observability

Call `diagnose_geometry(observations, estimated_state, rank_tolerance)` on the final estimate and retain its result. Classification should distinguish:

- insufficient observations (`satellite_count < 4`);
- rank deficient (`not full_rank`);
- full rank but poorly conditioned (full rank, condition above the configured policy threshold);
- full rank and below the policy threshold.

The existing geometry module uses SVD rank tolerance `1e-12` by default and the benchmark category implementation uses condition boundaries `1e3` and `1e6`. The validator should make the poor-conditioning threshold configurable; reusing `1e6` as an initial engineering policy choice is consistent with the existing benchmark category, not a universal mathematical constant. Preserve poor conditioning as an independent diagnostic flag even when numerical convergence, fit, rank, or physical-domain checks fail. Distinguish insufficient observations, rank deficiency, full-rank poor conditioning, and full-rank conditioning within the configured policy threshold. Preserve numerical rank and condition values. PDOP remains the existing unweighted, unit-variance geometry measure and is undefined for rank-deficient geometry.

## 5. Proposed operating-domain configuration

Use an explicit mode, not an implicit universal GPS constraint. An initial design could be:

```python
@dataclass(frozen=True)
class OperatingDomain:
    name: Literal["terrestrial", "aviation", "unrestricted_ecef"]
    minimum_altitude_m: float | None
    maximum_altitude_m: float | None
    allow_unrestricted_domain_waiver: bool = False
    # Optional future geographic bounds belong here only if an application needs them.
```

Interpret altitude as WGS-84 geodetic altitude relative to the reference ellipsoid, calculated with `ecef_to_geodetic`. This is a domain-membership diagnostic, not a constraint or projection of the solver state. Reject invalid/non-finite or reversed bounds at configuration construction. `unrestricted_ecef` must not silently impose altitude bounds; it can still report converted geodetic coordinates when conversion is defined. The proposed explicit `allow_unrestricted_domain_waiver` flag defaults to false and is meaningful only for unrestricted mode. Without the flag, the physical-domain requirement is unmet and overall assessment is `incomplete_assessment`; with it, physical status remains `not_applicable`, is marked `waived`, and is never recorded as passed.

Suggested policies:

| Mode | Proposed use and default policy |
|---|---|
| `terrestrial` | Configurable inclusive altitude interval. For the current synthetic benchmark only, retain `-1,000 m` through `100,000 m` as a temporary experimental screening profile: the generated truth is at 20 m WGS-84 altitude, and the generous upper bound is intended to flag the pilot's extreme excursions. This interval is not a general definition of terrestrial receiver altitude, is not a universal default, and must not be silently applied to other applications. |
| `aviation` | Separate configurable altitude interval appropriate to the supported operating envelope. Do not claim one universal aviation maximum: aircraft class, service, and application requirements differ. Require the application to select/configure the upper bound; a provisional implementation may use a clearly labeled example profile but should not silently make it normative. |
| `unrestricted_ecef` | No altitude acceptance/rejection bounds. Report physical-domain status as `not_applicable`, not passed. Without the explicit waiver flag, assessment is incomplete. With the flag, report the physical dimension as waived and `not_applicable`; it is not a pass. Still report geodetic altitude where conversion is defined, and still run fit and geometry checks. |

The synthetic benchmark's current `ScenarioConfig` fixes the generated receiver at latitude 37 degrees, longitude -122 degrees, altitude 20 m, and creates synthetic satellite directions. The current synthetic benchmark must explicitly select the temporary terrestrial screening profile `[-1,000, 100,000] m`; it must not select unrestricted mode. That interval must not become the validator's implicit default. If future scenarios vary altitude or include aircraft/space users, the benchmark/application must select an explicitly appropriate domain; the validator must not infer domain from hidden truth.

Thresholds should be frozen using requirements and development data before a final held-out run. Do not choose them to improve test-set accuracy or validity rates.

## 6. Classification rules

Retain the three assessment outcomes, one deterministic primary classification for compact summary tables, and all independent dimension flags and values. Assessment status is determined as follows:

- `incomplete_assessment` if the numerical result is not converged, a required diagnostic cannot be computed, the estimate is invalid/unavailable, required geometry cannot be established, measurement fit is `not_evaluated`, or the physical-domain result is not assessable.
- `invalid_under_configured_policy` if the assessment is complete and one or more required checks fail.
- `valid_under_configured_policy` only if the assessment is complete and every required check passes. An unrestricted domain is explicitly `not_applicable`; it does not constitute a physical pass. Without `allow_unrestricted_domain_waiver=True`, unrestricted mode leaves the physical requirement unmet and the assessment incomplete. With the flag, report the physical dimension as `not_applicable` and `waived`, never as passed; only this explicit waiver permits the other configured checks to determine overall validity.

For primary classification, apply the incomplete rule first. Insufficient observations, non-convergence, and unassessable required dimensions produce primary classification `incomplete_assessment`; retain explicit flags such as `insufficient_observations` and `not_numerically_converged`. For complete assessments, use this headline precedence:

1. `geometry_unobservable` when the final Jacobian is rank deficient;
2. `poor_observability` when it is full rank but crosses the configured condition limit;
3. `measurement_fit_unacceptable` when a configured fit threshold fails;
4. `physically_implausible` when the selected operating domain is violated;
5. `invalid_under_configured_policy` when one or more required checks fail;
6. `valid_under_configured_policy` when the assessment is complete and all required checks pass.

For any incomplete assessment, use `incomplete_assessment` as its primary classification while retaining all available diagnostic flags, including geometry and measurement findings. For a non-converged numerical result, retain its exact solver status in `numerical_status` and add a `not_numerically_converged` flag; its assessment status is incomplete, not valid or invalid. For multiple complete failures, primary precedence chooses only the headline; all failed dimensions remain present in `flags` and their detailed values.

The primary label does not make dimensions mutually exclusive. For instance, a converged result may carry both `poor_observability` and `physically_implausible`, plus `measurement_fit_failed`. Preserve all flags and each numerical diagnostic. A rank-deficient solve and insufficient observations are not interchangeable. A non-converged result may still have residual/geometry statistics if a finite final state exists, but must be counted as non-converged and incomplete, never valid.

“Valid” means valid under the explicitly configured policy, not proven correct. The policy cannot identify a wrong root if that root fits the data, is locally observable, and lies inside the allowed domain.

## 7. Truth-data separation

The validator interface must not accept `Scenario`, `ScenarioTruth`, or a truth coordinate. It receives only observations and the estimate/status/configuration. Keep it in a module that does not import scenario-generation classes. Tests should inspect the public signature and/or use a fake input object whose truth property raises if accessed, proving validity evaluation has no truth dependency.

Benchmark code continues to have access to truth solely for evaluator metrics such as 3D position error and clock-bias error. Those metrics are computed after validity assessment and never passed back into the validator. In a real receiver, truth is unavailable; the same checks still operate using residual fit, observable geometry, and the receiver/operator’s configured operating domain. Consequently, the validator can flag implausible/poorly supported solutions but cannot compute accuracy or prove that a plausible low-residual root is the true root.

## 8. Benchmark integration and reporting proposal

Keep existing raw columns and summary files intact. Add independent columns, preferably with a stable `validity_` prefix:

- `validity_classification`, `validity_assessment_status` (one of the three explicit outcomes), `validity_flags`, `validity_domain`;
- `validity_residual_rms_m`, `validity_max_abs_residual_m`, `validity_fit_status` (`pass`/`fail`/`not_evaluated`/`not_assessable`), plus configured fit-limit outcomes;
- `validity_numerical_rank`, `validity_condition_number`, `validity_pdop`, `validity_full_rank`;
- `estimated_latitude_rad`, `estimated_longitude_rad`, `estimated_altitude_m`, `physical_domain_status` (`pass`/`fail`/`not_applicable`/`not_assessable`).

The current raw row already has `residual_rms_m`, `solver_reported_residual_rms_m`, initial `numerical_rank`, `condition_number`, and `pdop`. Avoid duplicating those values unless explicitly renaming/versioning is necessary. The key change is to make final-state geometry and the validity classification available, while preserving current initial-geometry fields and solver metrics. CSV empty cells should continue to represent unavailable metrics; do not replace missing values with zero.

Add split-specific summary outputs, keeping development and test separate. For each algorithm and split, publish integer counts and denominators as well as rates. Define `N_all` as every scenario/algorithm run row in that split; `N_converged` as rows whose original solver/controller status is exactly `converged`; and `N_complete` as rows with a complete validity assessment. Report:

- **Valid solutions:** `valid_count / N_all` and separately `valid_count / N_converged`. The first is the valid-solution rate across all attempts; the second is validity conditional on numerical convergence. Always report `valid_count`, both denominators, and the number of converged rows. If a denominator is zero, rate is missing, not zero.
- **Invalid solutions:** `invalid_count / N_complete`; also report `invalid_count / N_all` as a fraction of all attempts if useful, clearly labeled. Incomplete assessments are excluded from the complete-assessment denominator and counted separately.
- **Incomplete assessments:** `incomplete_count / N_all`, with the number and reasons. Non-converged outcomes are a separately reported subset: `nonconverged_count / N_all`, retaining exact solver statuses. By definition, non-converged results are incomplete assessments but must not be added twice when producing mutually exclusive outcome totals.
- The mutually exclusive assessment accounting is `valid_count + invalid_count + incomplete_count = N_all`; `nonconverged_count` is a subset of `incomplete_count`, not an additional outcome category.

### Diagnostic-rate populations and denominators

Calculate every count independently for each `(split, algorithm)` group; never combine development and test rows. For each dimension, publish `evaluated_count`, `failure_count`, `unavailable_count`, and `not_applicable_or_waived_count` where relevant. A dimension's rate is `failure_count / evaluated_count`; when `evaluated_count == 0`, the rate is missing, not zero. Rows that cannot be evaluated are never included as passes or failures.

| Diagnostic rate | Eligible/evaluated population (denominator) | Failure numerator | Unavailable / separately counted rows |
|---|---|---|---|
| Physical-implausibility rate | Run rows with a finite estimated state, successful geodetic conversion, and a configured bounded domain (`terrestrial` or `aviation`) whose bounds can be evaluated. This diagnostic denominator is independent of numerical-convergence status. | Eligible rows outside the configured altitude/domain bounds. | Invalid/unavailable states and conversion failures count as unavailable. `unrestricted_ecef` rows with the explicit waiver count as `not_applicable_or_waived`, not in the denominator; unrestricted without waiver is incomplete and unavailable. |
| Poor-observability rate | Run rows with a finite estimated state and successfully computed final geometry that is full rank with a finite condition number. This conditioning diagnostic denominator is independent of fit and physical outcomes. | Eligible full-rank rows whose condition number exceeds the configured policy threshold. | Invalid/unavailable states, Jacobian failures, insufficient observations, and rank-deficient geometry are excluded from this denominator and counted separately. In particular, rank-deficient geometry is not miscounted as poor conditioning. |
| Rank-deficiency rate | Run rows with a finite estimated state, at least four observations, and successfully computed final Jacobian rank. This diagnostic denominator is independent of numerical-convergence status. | Eligible rows whose numerical rank is less than four. | Invalid/unavailable states, insufficient observations, and Jacobian failures are counted separately as unavailable for this rate. Insufficient observations have their own count/rate over all run rows and must not be called rank-deficient. |
| Measurement-fit-failure rate | Run rows with a finite estimated state and successfully recomputed residual statistics, with at least one fit threshold configured. This diagnostic denominator is independent of numerical-convergence status. Every configured RMS/max-absolute-residual limit is evaluated; a row passes only if all configured limits pass. | Eligible rows failing one or more configured limits. | Invalid/unavailable states or residual-evaluation failures are unavailable. If no limit is configured, all otherwise measurable rows have fit status `not_evaluated`; report their count separately, with evaluated denominator zero and failure rate missing. |

Report the overall denominators `N_all`, `N_converged`, and `N_complete` beside these per-dimension populations. This distinguishes a low failure rate based on few assessable rows from broad coverage. Preserve complete-assessment totals and each dimension's unavailable/not-applicable counts in metadata and split-specific summary CSVs.

Also report:

- numerical convergence rate (unchanged status-based metric, clearly named);
- valid-solution rate among all runs and among numerically converged runs using the denominators defined above;
- physically implausible rate;
- poor-observability and rank-deficient rates;
- measurement-fit failure rate among cases where configured/evaluable;
- counts of missing/unassessable values and counts for each solver failure status; report the number evaluated as well as the failure denominator for each diagnostic rate;
- position error conditional on `valid_under_configured_policy`;
- position error across all numerically converged solutions, regardless of validity, retained as a complementary result.

Use the same denominators and missingness rules in metadata and CSV summaries. Report both numerator and denominator, not only percentages. Failure/unobservable rows remain in status accounting even when their error metrics are missing. Report position-error statistics conditional on valid solutions using only rows classified `valid_under_configured_policy`; separately report position-error statistics for every numerically converged row, including invalid and incomplete assessments when the error metric exists. A validity-conditional accuracy statistic must not exclude or hide the unconditional converged-solution statistic.

## 9. Test plan

Use deterministic observations/states, and test each dimension independently as well as combinations:

1. **Exact-fit but implausible four-satellite estimate:** reuse/reconstruct the pilot’s demonstrated four-observation wrong root. Assert numerical status stays `converged`, fit is excellent, physical domain fails, and the classification/flags do not call it valid. Assert no truth is needed by the validator.
2. **Eight-satellite poor geometry:** finite solution with full rank but condition above configured threshold. Assert `poor_observability`, and independently test the fit outcome and physical domain rather than assuming one determines the others.
3. **Good geometry and plausible estimate:** synthetic full-rank observations and a plausible ECEF estimate should pass all explicitly enabled policy checks.
4. **Rank-deficient geometry:** assert rank-deficient is distinct from fewer than four observations; PDOP remains unavailable as in the geometry API.
5. **Insufficient observations:** assert explicit insufficient-data classification rather than rank-deficient or generic numerical failure.
6. **Non-converged solver output:** preserve exact source status (e.g. `max_iterations` or `line_search_failed`), retain any computable diagnostics, and never count it as a valid solution.
7. **Low residual/high error:** assert low residual only passes the configured measurement-fit dimension and does not itself prove accuracy or validity. Truth-based error may be checked in evaluator tests, never in validator decision tests.
8. **Operating-domain variants:** same estimate under terrestrial, aviation, and unrestricted configurations should yield domain outcomes appropriate to each explicitly supplied range. Test inclusive boundary behavior, invalid bound rejection, unrestricted-without-waiver incomplete status, and unrestricted-with-waiver `not_applicable` plus `waived` reporting (never pass).
9. **Thresholds disabled/configured:** verify fit status is `not_evaluated` when no fit limit is configured and the overall result is `incomplete_assessment`; verify deterministic fit pass/fail when RMS and/or maximum-residual thresholds are configured, including that both limits must pass when both are present.
10. **Combination of failures:** a result violating multiple enabled checks must retain every flag while exposing the documented primary classification precedence.
11. **Non-finite/unavailable state:** retain missing diagnostic values and a clear incomplete/not-converged classification; do not fabricate zero residuals or geodetic fields.
12. **Truth-data isolation:** verify validator call signature and runtime behavior do not read `ScenarioTruth` or require scenario access.
13. **Benchmark accounting:** verify added split-specific counts, denominators, conditional/unconditional position-error populations, and missing metrics while asserting existing raw columns and summary outputs are unchanged.

## 10. Trade-offs and limitations

- A physically implausible state can still be a mathematical least-squares solution. The validator reports it; it does not alter the solve or choose another root.
- A physically plausible state with low residual can still be wrong. Without truth, integrity monitoring, priors, temporal history, or independent sensors, the proposed checks cannot prove correctness.
- Hard altitude bounds are application policy. Overly narrow bounds can reject legitimate aircraft, elevated platforms, unusual deployments, or non-terrestrial users; unrestricted mode must remain available.
- The temporary `[-1,000, 100,000] m` synthetic-benchmark screening interval should be reviewed before use even in that experiment. It is not a general definition of “terrestrial” and must not be a universal default for other applications.
- Current geometry PDOP uses equal independent unit-variance observations. It does not model actual measurement uncertainty or multipath/outlier weighting. Geometry and measurement-fit failures should therefore remain separate.
- Condition-number thresholds are implementation-defined engineering choices. They should be documented and evaluated on development data, not described as universal observability constants.
- Local Jacobian rank/condition describes the current estimate, not global uniqueness or correctness. Four equations for four unknowns can admit multiple roots; overdetermined problems can still be ill-conditioned or contaminated.
- Additional validity columns should be versioned/documented to avoid changing the meaning of approved Phase 6 fields. Empty means unavailable; it must not be conflated with pass.

## 11. Recommended implementation sequence

1. Finalize policy semantics and application profiles: terrestrial benchmark bounds, aviation configuration ownership, unrestricted behavior, fit thresholds, and observability threshold provenance.
2. Implement the standalone immutable result/configuration types and validator in `solution_validity.py`, reusing residual, geometry, and coordinate APIs. Add focused unit tests without changing existing numerical code.
3. Add a thin benchmark adapter that calls the validator from observations, final estimate, and existing status only. Keep benchmark truth-evaluation code downstream and separate.
4. Add raw validity columns and split-specific summary outputs while preserving existing columns/files and all missing values.
5. Add regression tests for benchmark truth separation, split-specific denominators, failure accounting, and the pilot’s low-residual/implausible case.
6. Run the complete test suite and a small development-only smoke benchmark; inspect raw and summary outputs. Freeze policy thresholds before any held-out analysis.
7. Only after review, assess whether any later phase should add integrity monitoring, multi-sensor evidence, or solver behavior changes; those are outside this validity-layer proposal.

## 12. Unresolved design decisions

1. **Terrestrial upper/lower bounds:** `-1,000 m` to `100,000 m` is only a temporary experimental screening profile for the current synthetic benchmark, carried forward from the Phase 6.5C diagnostic convention. Confirm whether even this screening profile is appropriate; it must not be a general definition or default for other applications.
2. **Aviation profile:** decide whether to ship a named default profile or require applications to supply bounds. This proposal favors requiring explicit bounds until the supported aviation operating envelope is known.
3. **Measurement-fit limits:** choose application requirements for RMS and maximum-absolute-residual limits before enabling a fit pass/fail decision. Until configured, fit status is `not_evaluated` and the assessment is incomplete. Any covariance-normalized statistic requires an explicit noise/covariance model.
4. **Poor-observability cutoff:** decide whether to reuse the benchmark’s `1e6` condition boundary for validity labeling or define a separately calibrated policy threshold. It must not alter existing geometry categories or solver selection.
5. **Primary-label precedence:** the ordering above is proposed for readable summary CSVs. The multi-flag result is normative; confirm whether consumers need a different headline ordering.
6. **Where to expose results:** the proposed standalone module and benchmark adapter avoid changing `SolverResult`/`HybridResult`. If API consumers later require validity nested in `HybridResult`, that should be a separately reviewed interface change.

## Inspected and created files

Inspected:

- [`models.py`](../src/gps_hybrid/models.py)
- [`gps_model.py`](../src/gps_hybrid/gps_model.py)
- [`coordinates.py`](../src/gps_hybrid/coordinates.py)
- [`geometry.py`](../src/gps_hybrid/geometry.py)
- [`hybrid.py`](../src/gps_hybrid/hybrid.py)
- [`benchmarking.py`](../src/gps_hybrid/benchmarking.py)
- [`solvers/base.py`](../src/gps_hybrid/solvers/base.py) for the existing result/configuration semantics
- [`scenario.py`](../src/gps_hybrid/scenario.py) for the synthetic benchmark’s generated receiver domain
- [`solution_validity_report.md`](../results/pilot_v2/diagnostics/solution_validity_report.md) for the prior pilot findings

Created:

- `docs/solution_validity_policy.md` (this proposal)

No Python source files, tests, or pilot result files were created or modified. No benchmark was executed.
