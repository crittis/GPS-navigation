"""Reproducible benchmarking for the GPS hybrid optimization project.

This module evaluates the approved solvers and hybrid orchestrator on shared,
deterministically generated synthetic scenarios.  Scenario generation is kept
separate from solver execution: ground truth is used only when computing
post-solve metrics and is never passed to a solver or selector.

The core output uses only the Python standard library and NumPy.  Plotting is
optional and loaded lazily so benchmarking remains usable without a plotting
dependency.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from itertools import product
from numbers import Integral, Real
from pathlib import Path
import json
import sys
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .coordinates import enu_to_ecef_rotation, geodetic_to_ecef
from .geometry import GeometryDiagnostics, diagnose_geometry
from .gps_model import residual
from .hybrid import HybridConfig, SolverFunction, _invalid_result_reason, run_hybrid
from .models import GPSObservations, Scenario, ScenarioTruth
from .scenario import ScenarioConfig
from .selection import SolverSelection, select_initial_solver
from .solution_validity import (
    SolutionValidityConfig,
    assess_solution_validity,
    synthetic_benchmark_validity_config,
)
from .solvers import (
    IterationRecord,
    SolverConfig,
    SolverResult,
    solve_bfgs,
    solve_gauss_newton,
    solve_levenberg_marquardt,
)


ALGORITHM_NAMES = (
    "GN", "LM", "BFGS", "initial_selection_only", "fixed_gn_adaptive", "full_hybrid",
)
STANDALONE_SOLVERS: dict[str, Callable[..., SolverResult]] = {
    "GN": solve_gauss_newton,
    "LM": solve_levenberg_marquardt,
    "BFGS": solve_bfgs,
}
GEOMETRY_CATEGORIES = ("good", "moderate", "poor", "rank_deficient")
POSITION_ERROR_METRICS = ("position_error_m", "clock_bias_error_m", "residual_rms_m")
STAGE_FIELDS = (
    "scenario_id", "algorithm", "stage_index", "solver_name", "status", "iterations",
    "function_evaluations", "execution_time_seconds", "rejected_steps", "failure_reason",
)


def _validate_real(name: str, value: object, *, minimum: float | None = None,
                   maximum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return numeric


def _validate_positive_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for deterministic benchmark generation and execution."""

    scenario_count: int = 500
    seed: int = 20261109
    test_fraction: float = 0.2
    satellite_counts: tuple[int, ...] = (4, 6, 8, 10, 12)
    noise_stds_m: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0)
    position_perturbations_m: tuple[float, ...] = (10.0, 100.0, 500.0, 1000.0)
    clock_bias_perturbation_m: float = 100.0
    geometry_categories: tuple[str, ...] = GEOMETRY_CATEGORIES
    outlier_rates: tuple[float, ...] = (0.0,)
    outlier_magnitude_m: float = 1000.0
    solver_config: SolverConfig = field(default_factory=SolverConfig)
    hybrid_config: HybridConfig = field(default_factory=HybridConfig)
    validity_config: SolutionValidityConfig = field(default_factory=synthetic_benchmark_validity_config)
    require_both_splits: bool = False

    def __post_init__(self) -> None:
        _validate_positive_int("scenario_count", self.scenario_count)
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        _validate_real("test_fraction", self.test_fraction, minimum=0.0, maximum=1.0)
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be strictly between zero and one")
        if not self.satellite_counts:
            raise ValueError("satellite_counts cannot be empty")
        for count in self.satellite_counts:
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 4:
                raise ValueError("satellite_counts must contain integers of at least four")
        for noise in self.noise_stds_m:
            _validate_real("noise standard deviation", noise, minimum=0.0)
        if not self.noise_stds_m:
            raise ValueError("noise_stds_m cannot be empty")
        for perturbation in self.position_perturbations_m:
            _validate_real("position perturbation", perturbation, minimum=0.0)
        if not self.position_perturbations_m:
            raise ValueError("position_perturbations_m cannot be empty")
        _validate_real("clock_bias_perturbation_m", self.clock_bias_perturbation_m, minimum=0.0)
        if not self.geometry_categories or any(category not in GEOMETRY_CATEGORIES for category in self.geometry_categories):
            raise ValueError(f"geometry_categories must use {GEOMETRY_CATEGORIES}")
        for rate in self.outlier_rates:
            _validate_real("outlier rate", rate, minimum=0.0, maximum=1.0)
        if not self.outlier_rates:
            raise ValueError("outlier_rates cannot be empty")
        _validate_real("outlier_magnitude_m", self.outlier_magnitude_m, minimum=0.0)
        if not isinstance(self.solver_config, SolverConfig):
            raise ValueError("solver_config must be a SolverConfig")
        if not isinstance(self.hybrid_config, HybridConfig):
            raise ValueError("hybrid_config must be a HybridConfig")
        if not isinstance(self.validity_config, SolutionValidityConfig):
            raise ValueError("validity_config must be a SolutionValidityConfig")
        if not isinstance(self.require_both_splits, (bool, np.bool_)):
            raise ValueError("require_both_splits must be boolean")


@dataclass(frozen=True)
class BenchmarkScenario:
    """One generated scenario and the manifest fields needed to reproduce it."""

    scenario_id: str
    split: str
    seed: int
    scenario_config: ScenarioConfig
    outlier_rate: float
    realized_outlier_count: int
    realized_outlier_fraction: float
    realized_outlier_condition: str
    requested_geometry_category: str
    verified_geometry_category: str
    scenario: Scenario
    initial_geometry: GeometryDiagnostics


@dataclass(frozen=True)
class BenchmarkResults:
    """Raw benchmark rows and metadata returned before or after CSV writing."""

    manifest_rows: tuple[dict[str, Any], ...]
    run_rows: tuple[dict[str, Any], ...]
    stage_rows: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


def _direction_template(category: str, count: int, rng: np.random.Generator) -> np.ndarray:
    if category == "rank_deficient":
        return np.tile(np.array([1.0, 0.0, 0.0]), (count, 1))
    if category == "poor":
        base = np.array([
            [1.0, 0.0, 0.0], [1.0, 1e-4, 1e-4],
            [1.0, -1e-4, 2e-4], [1.0, 2e-4, -1e-4],
        ])
        directions = base[np.arange(count) % len(base)].copy()
        if count > len(base):
            directions[len(base):] += rng.normal(0.0, 1e-5, (count - len(base), 3))
        return directions
    if category == "moderate":
        base = np.array([
            [1.0, 0.0, 0.0], [1.0, 2e-2, 2e-2],
            [1.0, -2e-2, 4e-2], [1.0, 4e-2, -2e-2],
        ])
        directions = base[np.arange(count) % len(base)].copy()
        if count > len(base):
            directions[len(base):] = rng.normal(0.0, 1.0, (count - len(base), 3))
        return directions
    tetrahedron = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [-1.0, -1.0, -1.0],
    ])
    directions = np.empty((count, 3), dtype=float)
    directions[: min(count, 4)] = tetrahedron[: min(count, 4)]
    if count > 4:
        directions[4:] = rng.normal(0.0, 1.0, (count - 4, 3))
    return directions


def _classify_geometry(diagnostics: GeometryDiagnostics) -> str:
    """Map diagnostics to implementation-defined benchmark categories.

    The ``1e3`` and ``1e6`` condition-number boundaries are engineering
    thresholds chosen to separate visibly moderate and severe sensitivity in
    this synthetic study; they are not universal GPS observability constants.
    """
    if diagnostics.insufficient_observations or not diagnostics.full_rank:
        return "rank_deficient"
    if diagnostics.condition_number >= 1e6:
        return "poor"
    if diagnostics.condition_number >= 1e3:
        return "moderate"
    return "good"


def generate_benchmark_scenario(*, scenario_id: str, seed: int, split: str,
                                satellite_count: int, noise_std_m: float,
                                position_perturbation_m: float,
                                geometry_category: str, outlier_rate: float = 0.0,
                                outlier_magnitude_m: float = 1000.0,
                                clock_bias_perturbation_m: float = 100.0) -> BenchmarkScenario:
    """Generate one reproducible synthetic scenario without exposing truth to solvers."""
    if not isinstance(split, str) or split not in {"development", "test"}:
        raise ValueError("split must be 'development' or 'test'")
    if geometry_category not in GEOMETRY_CATEGORIES:
        raise ValueError(f"geometry_category must use {GEOMETRY_CATEGORIES}")
    if isinstance(scenario_id, str) is False or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    _validate_nonnegative_int("seed", seed)
    _validate_positive_int("satellite_count", satellite_count)
    if satellite_count < 4:
        raise ValueError("satellite_count must be at least four")
    _validate_real("noise_std_m", noise_std_m, minimum=0.0)
    _validate_real("position_perturbation_m", position_perturbation_m, minimum=0.0)
    _validate_real("clock_bias_perturbation_m", clock_bias_perturbation_m, minimum=0.0)
    _validate_real("outlier_rate", outlier_rate, minimum=0.0, maximum=1.0)
    _validate_real("outlier_magnitude_m", outlier_magnitude_m, minimum=0.0)
    rng = np.random.default_rng(seed)
    scenario_config = ScenarioConfig(
        seed=seed, satellite_count=satellite_count,
        pseudorange_noise_std_m=noise_std_m,
        initial_position_perturbation_std_m=position_perturbation_m,
        initial_bias_perturbation_std_m=clock_bias_perturbation_m,
    )
    latitude = np.deg2rad(scenario_config.latitude_deg)
    longitude = np.deg2rad(scenario_config.longitude_deg)
    receiver = geodetic_to_ecef(latitude, longitude, scenario_config.altitude_m)
    truth_state = np.array([*receiver, scenario_config.clock_bias_m], dtype=float)
    rotation = enu_to_ecef_rotation(latitude, longitude)
    initial = truth_state + np.concatenate((
        rng.normal(0.0, position_perturbation_m, 3),
        [rng.normal(0.0, clock_bias_perturbation_m)],
    ))
    directions = _direction_template(geometry_category, satellite_count, rng)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    geometry_anchor = initial[:3] if geometry_category == "rank_deficient" else receiver
    satellites = geometry_anchor + scenario_config.satellite_range_m * (directions @ rotation.T)
    ranges = np.linalg.norm(satellites - receiver, axis=1)
    pseudoranges = ranges + scenario_config.clock_bias_m + rng.normal(
        0.0, noise_std_m, satellite_count,
    )
    realized_outlier_count = 0
    if outlier_rate > 0.0 and outlier_magnitude_m > 0.0:
        mask = rng.random(satellite_count) < outlier_rate
        signs = rng.choice(np.array([-1.0, 1.0]), size=satellite_count)
        pseudoranges = pseudoranges + mask * signs * outlier_magnitude_m
        realized_outlier_count = int(np.count_nonzero(mask))
    realized_fraction = realized_outlier_count / satellite_count
    realized_condition = _realized_outlier_condition(realized_outlier_count, satellite_count)
    scenario = Scenario(
        ScenarioTruth(truth_state), GPSObservations(satellites, pseudoranges), initial, seed,
    )
    diagnostics = diagnose_geometry(scenario.observations, scenario.initial_estimate)
    verified = _classify_geometry(diagnostics)
    return BenchmarkScenario(
        scenario_id, split, seed, scenario_config, outlier_rate,
        realized_outlier_count, realized_fraction, realized_condition,
        geometry_category, verified,
        scenario, diagnostics,
    )


def generate_benchmark_scenarios(config: BenchmarkConfig) -> tuple[BenchmarkScenario, ...]:
    """Create a seeded, stratified development/test scenario manifest.

    The experimental parameter tuple is the stratum.  Test allocations use
    the largest-remainder method: each stratum receives the floor of its ideal
    proportional allocation, then remaining slots go to the largest
    fractional remainders, with seeded tie-breaking.  One-scenario runs are
    explicitly development-only; callers requiring a held-out split can set
    ``require_both_splits=True``.
    """
    combinations = list(product(
        config.satellite_counts, config.noise_stds_m,
        config.position_perturbations_m, config.geometry_categories,
        config.outlier_rates,
    ))
    if config.require_both_splits and config.scenario_count < 2:
        raise ValueError("At least two scenarios are required for both development and test splits")
    count = config.scenario_count
    test_count = 0 if count == 1 else max(1, min(count - 1, round(count * config.test_fraction)))
    rng = np.random.default_rng(config.seed)
    assignments = np.array(["development"] * count, dtype=object)
    if test_count:
        strata = {stratum: [] for stratum in range(len(combinations))}
        for index in range(count):
            strata[index % len(combinations)].append(index)
        ideal = {stratum: test_count * len(members) / count
                 for stratum, members in strata.items()}
        allocated = {stratum: min(len(members), int(np.floor(quota)))
                     for stratum, (members, quota) in
                     ((stratum, (strata[stratum], ideal[stratum])) for stratum in strata)}
        remaining_slots = test_count - sum(allocated.values())
        tie_order = list(rng.permutation(len(combinations)))
        tie_rank = {stratum: rank for rank, stratum in enumerate(tie_order)}
        remainder_order = sorted(
            strata,
            key=lambda stratum: (-(ideal[stratum] - np.floor(ideal[stratum])), tie_rank[stratum]),
        )
        for stratum in remainder_order:
            if remaining_slots == 0:
                break
            if allocated[stratum] < len(strata[stratum]):
                allocated[stratum] += 1
                remaining_slots -= 1
        selected: list[int] = []
        for stratum, members in strata.items():
            selected.extend(list(rng.permutation(members))[:allocated[stratum]])
        assignments[selected] = "test"
    scenarios = []
    for index in range(count):
        satellite_count, noise, perturbation, geometry, outlier_rate = combinations[index % len(combinations)]
        seed = int(config.seed) + index
        split = str(assignments[index])
        scenarios.append(generate_benchmark_scenario(
            scenario_id=f"scenario-{index:05d}", split=split, seed=seed,
            satellite_count=int(satellite_count), noise_std_m=float(noise),
            position_perturbation_m=float(perturbation), geometry_category=geometry,
            outlier_rate=float(outlier_rate), outlier_magnitude_m=config.outlier_magnitude_m,
            clock_bias_perturbation_m=config.clock_bias_perturbation_m,
        ))
    return tuple(scenarios)


@dataclass(frozen=True)
class _AlgorithmOutcome:
    status: str
    reason: str | None
    estimated_state: np.ndarray | None
    final_objective: float | None
    residual_rms: float | None
    solver_reported_residual_rms: float | None
    iterations: int
    function_evaluations: int
    execution_time_seconds: float
    rejected_steps: int
    solver_switches: int
    solver_sequence: tuple[str, ...]
    stage_results: tuple[tuple[str, SolverResult], ...]


def _result_outcome(result: SolverResult, elapsed: float, solver_name: str) -> _AlgorithmOutcome:
    return _AlgorithmOutcome(
        status=result.convergence_status,
        reason=result.failure_reason,
        estimated_state=np.asarray(result.estimated_state, dtype=float),
        final_objective=result.final_objective if np.isfinite(result.final_objective) else None,
        residual_rms=result.final_residual_rms if np.isfinite(result.final_residual_rms) else None,
        solver_reported_residual_rms=result.final_residual_rms if np.isfinite(result.final_residual_rms) else None,
        iterations=result.iterations,
        function_evaluations=result.function_evaluations,
        execution_time_seconds=elapsed,
        rejected_steps=sum(not record.accepted for record in result.convergence_history),
        solver_switches=0,
        solver_sequence=(solver_name,),
        stage_results=((solver_name, result),),
    )


def _run_solver(solver: Callable[..., SolverResult], observations: GPSObservations,
                initial_state: np.ndarray, config: SolverConfig,
                solver_name: str) -> _AlgorithmOutcome:
    started = perf_counter()
    try:
        result = solver(observations, initial_state, config)
    except Exception as exc:
        return _AlgorithmOutcome(
            status="numerical_failure", reason=f"{type(exc).__name__}: {exc}",
            estimated_state=None, final_objective=None, residual_rms=None,
            solver_reported_residual_rms=None, iterations=0, function_evaluations=0,
            execution_time_seconds=perf_counter() - started, rejected_steps=0,
            solver_switches=0, solver_sequence=(solver_name,), stage_results=tuple(),
        )
    validation_reason = _invalid_result_reason(result)
    if validation_reason is not None:
        return _AlgorithmOutcome(
            status="numerical_failure", reason=validation_reason,
            estimated_state=None, final_objective=None, residual_rms=None,
            solver_reported_residual_rms=None, iterations=0, function_evaluations=0,
            execution_time_seconds=perf_counter() - started, rejected_steps=0,
            solver_switches=0, solver_sequence=(solver_name,), stage_results=tuple(),
        )
    return _result_outcome(result, perf_counter() - started, solver_name)


def _run_initial_selection_only(scenario: BenchmarkScenario,
                                config: BenchmarkConfig) -> _AlgorithmOutcome:
    started = perf_counter()
    try:
        selection = select_initial_solver(scenario.initial_geometry, config.hybrid_config.selection_config)
        if not selection.solvable or selection.solver is None:
            return _AlgorithmOutcome(
                status=selection.decision, reason=selection.reason, estimated_state=None,
                final_objective=None, residual_rms=None, solver_reported_residual_rms=None,
                iterations=0, function_evaluations=0,
                execution_time_seconds=perf_counter() - started, rejected_steps=0,
                solver_switches=0, solver_sequence=tuple(), stage_results=tuple(),
            )
        outcome = _run_solver(selection.solver, scenario.scenario.observations,
                              scenario.scenario.initial_estimate, config.solver_config,
                              selection.solver_name or "selected_solver")
        return replace(outcome, execution_time_seconds=perf_counter() - started)
    except Exception as exc:
        return _AlgorithmOutcome(
            status="numerical_failure", reason=f"{type(exc).__name__}: {exc}",
            estimated_state=None, final_objective=None, residual_rms=None,
            solver_reported_residual_rms=None, iterations=0, function_evaluations=0,
            execution_time_seconds=perf_counter() - started, rejected_steps=0,
            solver_switches=0, solver_sequence=tuple(), stage_results=tuple(),
        )


def _run_hybrid(scenario: BenchmarkScenario, config: BenchmarkConfig,
                *, fixed_gn: bool) -> _AlgorithmOutcome:
    started = perf_counter()
    hybrid_config = config.hybrid_config
    if fixed_gn:
        hybrid_config = HybridConfig(
            solver_config=hybrid_config.solver_config,
            selection_config=hybrid_config.selection_config,
            max_total_iterations=hybrid_config.max_total_iterations,
            max_transitions=hybrid_config.max_transitions,
            stage_iteration_budget=hybrid_config.stage_iteration_budget,
            consecutive_failure_threshold=hybrid_config.consecutive_failure_threshold,
            consecutive_stagnation_threshold=hybrid_config.consecutive_stagnation_threshold,
            objective_progress_tolerance=hybrid_config.objective_progress_tolerance,
            enable_bfgs_fallback=hybrid_config.enable_bfgs_fallback,
            rank_tolerance=hybrid_config.rank_tolerance,
            initial_solver_override="Gauss-Newton",
        )
    try:
        result = run_hybrid(scenario.scenario.observations, scenario.scenario.initial_estimate, hybrid_config)
    except Exception as exc:
        return _AlgorithmOutcome(
            status="numerical_failure", reason=f"{type(exc).__name__}: {exc}",
            estimated_state=None, final_objective=None, residual_rms=None,
            solver_reported_residual_rms=None, iterations=0, function_evaluations=0,
            execution_time_seconds=perf_counter() - started, rejected_steps=0,
            solver_switches=0, solver_sequence=tuple(), stage_results=tuple(),
        )
    stages = tuple((stage.solver_name, stage.result) for stage in result.stage_results)
    rejected = sum(
        not record.accepted
        for _, stage in stages for record in stage.convergence_history
    )
    return _AlgorithmOutcome(
        result.convergence_status, result.failure_reason, np.asarray(result.estimated_state, dtype=float),
        result.final_objective if np.isfinite(result.final_objective) else None,
        _final_residual_rms(result.estimated_state, scenario.scenario.observations),
        result.final_residual_rms if np.isfinite(result.final_residual_rms) else None,
        result.total_iterations, result.total_function_evaluations,
        perf_counter() - started, rejected, result.solver_switches,
        result.solver_sequence, stages,
    )


def _execute_algorithm(scenario: BenchmarkScenario, algorithm: str,
                       config: BenchmarkConfig) -> _AlgorithmOutcome:
    if algorithm in STANDALONE_SOLVERS:
        return _run_solver(STANDALONE_SOLVERS[algorithm], scenario.scenario.observations,
                           scenario.scenario.initial_estimate, config.solver_config, algorithm)
    if algorithm == "initial_selection_only":
        return _run_initial_selection_only(scenario, config)
    if algorithm == "fixed_gn_adaptive":
        return _run_hybrid(scenario, config, fixed_gn=True)
    if algorithm == "full_hybrid":
        return _run_hybrid(scenario, config, fixed_gn=False)
    raise ValueError(f"Unknown benchmark algorithm: {algorithm}")


def _metric_value(value: float | None) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def _outlier_condition(rate: float) -> str:
    if rate == 0.0:
        return "none"
    if rate == 1.0:
        return "all_observations"
    return "partial_observations"


def _realized_outlier_condition(count: int, satellite_count: int) -> str:
    if count == 0:
        return "none"
    if count == satellite_count:
        return "all_observations"
    return "partial_observations"


def _final_residual_rms(state: object, observations: GPSObservations) -> float | None:
    """Recompute RMS from the final finite state using the approved residual model."""
    estimate = np.asarray(state, dtype=float)
    if estimate.shape != (4,) or not np.all(np.isfinite(estimate)):
        return None
    try:
        values = residual(estimate, observations)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None
    return _metric_value(float(np.sqrt(np.mean(values * values))))


def _run_row(scenario: BenchmarkScenario, algorithm: str,
             outcome: _AlgorithmOutcome,
             validity_config: SolutionValidityConfig | None = None) -> dict[str, Any]:
    # Validate from solver-facing data before evaluator-only truth metrics.
    validity_config = validity_config or synthetic_benchmark_validity_config()
    validity = assess_solution_validity(
        scenario.scenario.observations, outcome.estimated_state,
        outcome.status, validity_config,
    )
    truth = scenario.scenario.truth.receiver_state
    estimated = outcome.estimated_state
    position_error = clock_error = None
    if estimated is not None and estimated.shape == (4,) and np.all(np.isfinite(estimated)):
        position_error = float(np.linalg.norm(estimated[:3] - truth[:3]))
        clock_error = float(abs(estimated[3] - truth[3]))
    final_rms = validity.measurement_fit.residual_rms_m
    unobservable = outcome.status in {
        "rank_deficient", "unobservable", "unobservable_geometry", "insufficient_observations",
    }
    failure = outcome.status not in {
        "converged", "rank_deficient", "unobservable", "unobservable_geometry", "insufficient_observations",
    }
    geometry = validity.geometry
    return {
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "split": scenario.split,
        "algorithm": algorithm,
        "satellite_count": scenario.initial_geometry.satellite_count,
        "noise_std_m": scenario.scenario_config.pseudorange_noise_std_m,
        "initial_perturbation_m": scenario.scenario_config.initial_position_perturbation_std_m,
        "clock_bias_perturbation_m": scenario.scenario_config.initial_bias_perturbation_std_m,
        "requested_geometry_category": scenario.requested_geometry_category,
        "geometry_category": scenario.verified_geometry_category,
        "outlier_rate": scenario.outlier_rate,
        "outlier_condition": _outlier_condition(scenario.outlier_rate),
        "realized_outlier_count": scenario.realized_outlier_count,
        "realized_outlier_fraction": scenario.realized_outlier_fraction,
        "realized_outlier_condition": scenario.realized_outlier_condition,
        "numerical_rank": scenario.initial_geometry.numerical_rank,
        "condition_number": _metric_value(scenario.initial_geometry.condition_number),
        "pdop": _metric_value(scenario.initial_geometry.pdop),
        "convergence_status": outcome.status,
        "failure_reason": outcome.reason,
        "final_objective": _metric_value(outcome.final_objective),
        "position_error_m": position_error,
        "clock_bias_error_m": clock_error,
        "residual_rms_m": final_rms,
        "solver_reported_residual_rms_m": _metric_value(outcome.solver_reported_residual_rms),
        "iterations": outcome.iterations,
        "function_evaluations": outcome.function_evaluations,
        "execution_time_seconds": _metric_value(outcome.execution_time_seconds),
        "rejected_steps": outcome.rejected_steps,
        "solver_switches": outcome.solver_switches,
        "solver_sequence": "->".join(outcome.solver_sequence),
        "is_converged": outcome.status == "converged",
        "is_failure": failure,
        "is_unobservable": unobservable,
        "is_numerical_failure": outcome.status == "numerical_failure",
        "validity_assessment_status": validity.assessment_status,
        "validity_classification": validity.classification,
        "validity_flags": "|".join(sorted(validity.flags)),
        "validity_explanations": "|".join(validity.explanations),
        "validity_fit_status": validity.measurement_fit.status,
        "validity_max_absolute_residual_m": _metric_value(
            validity.measurement_fit.maximum_absolute_residual_m
        ),
        "validity_fit_rms_threshold_m": validity.measurement_fit.rms_threshold_m,
        "validity_fit_max_absolute_threshold_m": validity.measurement_fit.maximum_absolute_threshold_m,
        "validity_fit_rms_threshold_passed": validity.measurement_fit.rms_threshold_passed,
        "validity_fit_max_absolute_threshold_passed": validity.measurement_fit.maximum_absolute_threshold_passed,
        "validity_geometry_status": validity.geometry_status,
        "validity_final_satellite_count": geometry.satellite_count if geometry is not None else None,
        "validity_final_numerical_rank": geometry.numerical_rank if geometry is not None else None,
        "validity_final_condition_number": _metric_value(geometry.condition_number) if geometry is not None else None,
        "validity_final_pdop": _metric_value(geometry.pdop) if geometry is not None else None,
        "validity_final_full_rank": geometry.full_rank if geometry is not None else None,
        "validity_domain": validity.physical.domain,
        "validity_physical_status": validity.physical.status,
        "validity_physical_waived": validity.physical.waived,
        "validity_minimum_altitude_m": validity.physical.minimum_altitude_m,
        "validity_maximum_altitude_m": validity.physical.maximum_altitude_m,
        "validity_estimated_latitude_rad": validity.physical.latitude_rad,
        "validity_estimated_longitude_rad": validity.physical.longitude_rad,
        "validity_estimated_altitude_m": validity.physical.altitude_m,
    }


def _stage_rows(scenario: BenchmarkScenario, algorithm: str,
                outcome: _AlgorithmOutcome) -> list[dict[str, Any]]:
    rows = []
    for index, (solver_name, result) in enumerate(outcome.stage_results, start=1):
        rows.append({
            "scenario_id": scenario.scenario_id,
            "algorithm": algorithm,
            "stage_index": index,
            "solver_name": solver_name,
            "status": result.convergence_status,
            "iterations": result.iterations,
            "function_evaluations": result.function_evaluations,
            "execution_time_seconds": result.execution_time_seconds,
            "rejected_steps": sum(not record.accepted for record in result.convergence_history),
            "failure_reason": result.failure_reason,
        })
    return rows


def _manifest_row(scenario: BenchmarkScenario) -> dict[str, Any]:
    config = scenario.scenario_config
    return {
        "scenario_id": scenario.scenario_id, "seed": scenario.seed, "split": scenario.split,
        "satellite_count": config.satellite_count,
        "noise_std_m": config.pseudorange_noise_std_m,
        "initial_perturbation_m": config.initial_position_perturbation_std_m,
        "clock_bias_perturbation_m": config.initial_bias_perturbation_std_m,
        "outlier_rate": scenario.outlier_rate,
        "outlier_condition": _outlier_condition(scenario.outlier_rate),
        "realized_outlier_count": scenario.realized_outlier_count,
        "realized_outlier_fraction": scenario.realized_outlier_fraction,
        "realized_outlier_condition": scenario.realized_outlier_condition,
        "requested_geometry_category": scenario.requested_geometry_category,
        "verified_geometry_category": scenario.verified_geometry_category,
        "initial_rank": scenario.initial_geometry.numerical_rank,
        "initial_condition_number": _metric_value(scenario.initial_geometry.condition_number),
        "initial_pdop": _metric_value(scenario.initial_geometry.pdop),
    }


def _split_coverage(
    scenarios: Sequence[BenchmarkScenario],
    expected_combinations: Sequence[tuple[Any, ...]] | None = None,
    configured_levels: dict[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Describe configured factor and full-combination coverage separately."""
    fields = (
        "satellite_count", "noise_std_m", "initial_perturbation_m",
        "geometry_category", "outlier_rate",
    )
    def factor_value(item: BenchmarkScenario, field: str) -> Any:
        return {
            "satellite_count": item.scenario_config.satellite_count,
            "noise_std_m": item.scenario_config.pseudorange_noise_std_m,
            "initial_perturbation_m": item.scenario_config.initial_position_perturbation_std_m,
            "geometry_category": item.requested_geometry_category,
            "outlier_rate": item.outlier_rate,
        }[field]

    all_combinations = set(expected_combinations or ()) or {
        tuple(factor_value(item, field) for field in fields) for item in scenarios
    }
    configured_levels = configured_levels or {
        field: sorted({factor_value(item, field) for item in scenarios}, key=str)
        for field in fields
    }
    coverage: dict[str, Any] = {}
    for split in ("development", "test"):
        selected = [item for item in scenarios if item.split == split]
        combinations = {
            tuple(factor_value(item, field) for field in fields)
            for item in selected
        }
        factors = {}
        for field in fields:
            values = {
                factor_value(item, field)
                for item in selected
            }
            configured_values = set(configured_levels[field])
            factors[field] = {
                "configured_values": sorted(configured_values, key=str),
                "observed_values": sorted(values, key=str),
                "missing_configured_values": sorted(configured_values - values, key=str),
                "represented_all_configured_values": values == configured_values,
            }
        coverage[split] = {
            "scenario_count": len(selected),
            "observed_parameter_combinations": len(combinations),
            "total_parameter_combinations": len(all_combinations),
            "missing_parameter_combinations": [
                list(item) for item in sorted(all_combinations - combinations, key=str)
            ],
            "represented_all_combinations": combinations == all_combinations,
            "factor_coverage": factors,
        }
    coverage["limitation"] = (
        "Full-combination coverage is only claimed when every combination is present "
        "in the split; small scenario counts may cover only a subset. Factor coverage "
        "is reported independently."
    )
    return coverage


def run_benchmark(config: BenchmarkConfig = BenchmarkConfig(), *, output_dir: str | Path | None = None) -> BenchmarkResults:
    """Run all configured algorithms on a shared deterministic scenario set."""
    scenarios = generate_benchmark_scenarios(config)
    run_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for algorithm in ALGORITHM_NAMES:
            outcome = _execute_algorithm(scenario, algorithm, config)
            run_rows.append(_run_row(scenario, algorithm, outcome, config.validity_config))
            stage_rows.extend(_stage_rows(scenario, algorithm, outcome))
    metadata = {
        "experiment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "software_version": "0.1.0",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scenario_count": config.scenario_count,
        "algorithm_names": list(ALGORITHM_NAMES),
        "configuration": asdict(config),
        "timing_method": (
            "No warm-up runs. Outer execution_time_seconds uses perf_counter around each "
            "algorithm invocation, including selection/orchestration overhead but excluding "
            "scenario generation and file I/O. Stage CSV times are solver-reported times."
        ),
        "dataset_split": (
            "Largest-remainder proportional test allocation by parameter stratum; "
            "seeded permutation breaks equal-remainder ties and selects members within strata"
        ),
        "split_counts": {
            split: sum(scenario.split == split for scenario in scenarios)
            for split in ("development", "test")
        },
        "split_warning": (
            "Single-scenario benchmark is development-only"
            if config.scenario_count == 1 else None
        ),
        "split_coverage": _split_coverage(
            scenarios,
            list(product(
                config.satellite_counts, config.noise_stds_m,
                config.position_perturbations_m, config.geometry_categories,
                config.outlier_rates,
            )),
            {
                "satellite_count": config.satellite_counts,
                "noise_std_m": config.noise_stds_m,
                "initial_perturbation_m": config.position_perturbations_m,
                "geometry_category": config.geometry_categories,
                "outlier_rate": config.outlier_rates,
            },
        ),
        "synthetic_model_note": "Satellite directions are synthetic local ENU rays, not orbital ephemerides",
    }
    metadata["validity_accounting_by_split"] = {
        split: _summary_rows(
            [row for row in run_rows if row["split"] == split], ("algorithm",),
        )
        for split in ("development", "test")
    }
    results = BenchmarkResults(
        tuple(_manifest_row(scenario) for scenario in scenarios),
        tuple(run_rows), tuple(stage_rows), metadata,
    )
    if output_dir is not None:
        write_benchmark_outputs(results, output_dir)
    return results


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]],
               fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(fieldnames if fieldnames is not None else rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key)) for key in fields} for row in rows)


def _summary_rows(rows: Sequence[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    output = []
    for key, group in grouped.items():
        summary: dict[str, Any] = dict(zip(group_fields, key))
        summary["scenario_algorithm_rows"] = len(group)
        for name, predicate in (
            ("converged", lambda row: row["is_converged"]),
            ("failure", lambda row: row["is_failure"]),
            ("unobservable", lambda row: row["is_unobservable"]),
            ("numerical_failure", lambda row: row["is_numerical_failure"]),
        ):
            count = sum(bool(predicate(row)) for row in group)
            summary[f"{name}_count"] = count
            summary[f"{name}_rate"] = count / len(group)
        for metric in POSITION_ERROR_METRICS + ("iterations", "function_evaluations", "execution_time_seconds", "rejected_steps", "solver_switches"):
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "") and np.isfinite(row[metric])]
            summary[f"{metric}_missing_count"] = len(group) - len(values)
            if values:
                array = np.asarray(values, dtype=float)
                summary[f"{metric}_mean"] = float(np.mean(array))
                summary[f"{metric}_std"] = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
                summary[f"{metric}_median"] = float(np.median(array))
                summary[f"{metric}_q1"] = float(np.percentile(array, 25))
                summary[f"{metric}_q3"] = float(np.percentile(array, 75))
                summary[f"{metric}_min"] = float(np.min(array))
                summary[f"{metric}_max"] = float(np.max(array))
            else:
                for suffix in ("mean", "std", "median", "q1", "q3", "min", "max"):
                    summary[f"{metric}_{suffix}"] = None

        n_all = len(group)
        n_converged = sum(
            row.get("convergence_status") == "converged"
            if "convergence_status" in row else bool(row.get("is_converged", False))
            for row in group
        )
        valid = [row for row in group
                 if row.get("validity_assessment_status") == "valid_under_configured_policy"]
        invalid = [row for row in group
                   if row.get("validity_assessment_status") == "invalid_under_configured_policy"]
        incomplete_count = n_all - len(valid) - len(invalid)
        n_complete = len(valid) + len(invalid)
        summary.update({
            "N_all": n_all,
            "N_converged": n_converged,
            "N_complete": n_complete,
            "valid_count": len(valid),
            "valid_rate_all_runs": _safe_rate(len(valid), n_all),
            "valid_rate_converged_runs": _safe_rate(len(valid), n_converged),
            "invalid_count": len(invalid),
            "invalid_rate_complete_assessments": _safe_rate(len(invalid), n_complete),
            "invalid_rate_all_runs": _safe_rate(len(invalid), n_all),
            "incomplete_count": incomplete_count,
            "incomplete_rate_all_runs": _safe_rate(incomplete_count, n_all),
            "numerical_convergence_rate": _safe_rate(n_converged, n_all),
            "nonconverged_count": n_all - n_converged,
            "nonconverged_rate_all_runs": _safe_rate(n_all - n_converged, n_all),
            "assessment_accounting_invariant": len(valid) + len(invalid) + incomplete_count == n_all,
        })
        status_values = [row.get("convergence_status", "unknown") for row in group]
        known_statuses = {
            "converged", "max_iterations", "max_total_iterations", "max_transitions",
            "numerical_failure", "line_search_failed", "rank_deficient", "unobservable",
            "unobservable_geometry", "insufficient_observations", "solver_failure",
            "budget_integrity_failure",
        }
        for status in sorted(set(status_values) | known_statuses):
            safe_status = "_".join(part for part in str(status).lower().replace("-", "_").split() if part)
            summary[f"solver_status_{safe_status}_count"] = status_values.count(status)

        def diagnostic_counts(prefix: str, statuses: tuple[str, ...], failure_status: str,
                              *, not_applicable_status: str | None = None,
                              excluded_statuses: tuple[str, ...] = ()) -> None:
            values = [row.get(prefix) for row in group]
            evaluated = [value for value in values if value in statuses]
            failed = sum(value == failure_status for value in evaluated)
            unavailable = sum(value in (None, "not_assessable") for value in values)
            not_applicable = (
                sum(value == not_applicable_status for value in values)
                if not_applicable_status is not None else 0
            )
            summary[f"{prefix}_evaluated_count"] = len(evaluated)
            summary[f"{prefix}_failure_count"] = failed
            summary[f"{prefix}_failure_rate"] = _safe_rate(failed, len(evaluated))
            summary[f"{prefix}_unavailable_count"] = unavailable
            summary[f"{prefix}_not_applicable_or_waived_count"] = not_applicable
            for excluded in excluded_statuses:
                summary[f"{prefix}_{excluded}_count"] = sum(value == excluded for value in values)

        diagnostic_counts(
            "validity_physical_status", ("pass", "fail"), "fail",
            not_applicable_status="not_applicable",
        )
        physical_waived_count = sum(
            row.get("validity_physical_status") == "not_applicable"
            and bool(row.get("validity_physical_waived", False)) for row in group
        )
        physical_unavailable_count = sum(
            row.get("validity_physical_status") in (None, "not_assessable")
            or (row.get("validity_physical_status") == "not_applicable"
                and not bool(row.get("validity_physical_waived", False)))
            for row in group
        )
        summary["validity_physical_status_unavailable_count"] = physical_unavailable_count
        summary["validity_physical_status_not_applicable_or_waived_count"] = physical_waived_count
        summary["validity_physical_waived_count"] = physical_waived_count
        geometry_values = [row.get("validity_geometry_status") for row in group]
        insufficient_count = geometry_values.count("insufficient_observations")
        summary["validity_insufficient_observations_count"] = insufficient_count
        summary["validity_insufficient_observations_rate_all_runs"] = _safe_rate(insufficient_count, n_all)
        poor_population = [value for value in geometry_values
                           if value in ("poor_observability", "acceptable_conditioning")]
        poor_failures = sum(value == "poor_observability" for value in poor_population)
        summary.update({
            "validity_poor_observability_evaluated_count": len(poor_population),
            "validity_poor_observability_failure_count": poor_failures,
            "validity_poor_observability_failure_rate": _safe_rate(poor_failures, len(poor_population)),
            "validity_poor_observability_unavailable_count": sum(
                value in (None, "not_assessable", "insufficient_observations")
                for value in geometry_values
            ),
            "validity_poor_observability_excluded_rank_deficient_count": geometry_values.count("rank_deficient"),
            "validity_poor_observability_insufficient_observations_count": geometry_values.count("insufficient_observations"),
        })
        rank_population = [value for value in geometry_values if value in (
            "rank_deficient", "poor_observability", "acceptable_conditioning",
        )]
        rank_failures = rank_population.count("rank_deficient")
        summary.update({
            "validity_rank_deficiency_evaluated_count": len(rank_population),
            "validity_rank_deficiency_failure_count": rank_failures,
            "validity_rank_deficiency_failure_rate": _safe_rate(rank_failures, len(rank_population)),
            "validity_rank_deficiency_unavailable_count": sum(
                value in (None, "not_assessable", "insufficient_observations")
                for value in geometry_values
            ),
            "validity_rank_deficiency_insufficient_observations_count": geometry_values.count("insufficient_observations"),
        })
        diagnostic_counts(
            "validity_fit_status", ("pass", "fail"), "fail",
        )
        summary["validity_fit_not_evaluated_count"] = sum(
            row.get("validity_fit_status") == "not_evaluated" for row in group
        )

        for metric, selected in (
            ("position_error_valid_only_m", valid),
            ("position_error_converged_only_m", [
                row for row in group
                if (row.get("convergence_status") == "converged"
                    if "convergence_status" in row else bool(row.get("is_converged", False)))
            ]),
        ):
            values = [float(row["position_error_m"]) for row in selected
                      if row.get("position_error_m") not in (None, "")
                      and np.isfinite(float(row["position_error_m"]))]
            summary[f"{metric}_available_count"] = len(values)
            summary[f"{metric}_missing_count"] = len(selected) - len(values)
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None)
            summary[f"{metric}_median"] = float(np.median(values)) if values else None
            summary[f"{metric}_q1"] = float(np.percentile(values, 25)) if values else None
            summary[f"{metric}_q3"] = float(np.percentile(values, 75)) if values else None
            summary[f"{metric}_min"] = float(np.min(values)) if values else None
            summary[f"{metric}_max"] = float(np.max(values)) if values else None
        output.append(summary)
    return output


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a missing rate for an empty eligible population."""
    return numerator / denominator if denominator else None


def _paired_summary_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize paired algorithm differences using identical scenario IDs."""
    comparisons = (
        ("GN", "fixed_gn_adaptive"),
        ("initial_selection_only", "full_hybrid"),
        ("fixed_gn_adaptive", "full_hybrid"),
    )
    by_key = {(row["scenario_id"], row["algorithm"]): row for row in rows}
    scenario_ids = sorted({row["scenario_id"] for row in rows})
    output: list[dict[str, Any]] = []
    for left, right in comparisons:
        pairs = [
            (by_key[(scenario_id, left)], by_key[(scenario_id, right)])
            for scenario_id in scenario_ids
            if (scenario_id, left) in by_key and (scenario_id, right) in by_key
        ]
        summary: dict[str, Any] = {
            "left_algorithm": left,
            "right_algorithm": right,
            "available_pairs": len(pairs),
        }
        for name, field in (
            ("position_error_m_difference", "position_error_m"),
            ("execution_time_seconds_difference", "execution_time_seconds"),
        ):
            differences = [
                float(left_row[field]) - float(right_row[field])
                for left_row, right_row in pairs
                if left_row.get(field) not in (None, "")
                and right_row.get(field) not in (None, "")
                and np.isfinite(left_row[field]) and np.isfinite(right_row[field])
            ]
            summary[f"{name}_available_pairs"] = len(differences)
            summary[f"{name}_mean"] = float(np.mean(differences)) if differences else None
            summary[f"{name}_median"] = float(np.median(differences)) if differences else None
        convergence_differences = [
            int(left_row["is_converged"]) - int(right_row["is_converged"])
            for left_row, right_row in pairs
        ]
        summary["convergence_outcome_difference_available_pairs"] = len(convergence_differences)
        summary["convergence_outcome_difference_mean"] = (
            float(np.mean(convergence_differences)) if convergence_differences else None
        )
        output.append(summary)
    return output


def write_benchmark_outputs(results: BenchmarkResults, output_dir: str | Path) -> Path:
    """Write raw rows plus summaries kept strictly separate by data split."""
    root = Path(output_dir)
    raw = root / "raw"
    summaries = root / "summaries"
    (root / "figures").mkdir(parents=True, exist_ok=True)
    _write_csv(raw / "scenario_manifest.csv", results.manifest_rows)
    _write_csv(raw / "per_run_results.csv", results.run_rows)
    _write_csv(raw / "per_stage_results.csv", results.stage_rows, STAGE_FIELDS)
    summary_specs = (
        ("overall", ("algorithm",)),
        ("geometry", ("algorithm", "geometry_category")),
        ("noise", ("algorithm", "noise_std_m")),
        ("satellite_count", ("algorithm", "satellite_count")),
        ("initial_perturbation", ("algorithm", "initial_perturbation_m")),
        ("outlier_condition", ("algorithm", "outlier_condition")),
    )
    for split in ("development", "test"):
        split_rows = [row for row in results.run_rows if row["split"] == split]
        for name, fields in summary_specs:
            _write_csv(summaries / f"{split}_{name}_summary.csv", _summary_rows(split_rows, fields))
        _write_csv(summaries / f"{split}_ablation_summary.csv", _paired_summary_rows(split_rows))
    with (root / "experiment_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(results.metadata, handle, indent=2, default=_csv_value)
    return root


def plot_benchmark_results(output_dir: str | Path) -> None:
    """Create optional PNG figures from generated raw CSV results.

    Matplotlib is intentionally optional.  This function never fabricates
    data: every figure is derived from ``raw/per_run_results.csv``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires optional matplotlib") from exc
    path = Path(output_dir)
    with (path / "raw" / "per_run_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    algorithms = list(ALGORITHM_NAMES)
    figures = path / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def numeric(field: str, algorithm: str) -> list[float]:
        values = []
        for row in rows:
            if row["algorithm"] == algorithm and row.get(field, "") != "":
                value = float(row[field])
                if np.isfinite(value):
                    values.append(value)
        return values

    def boxplot(field: str, title: str, ylabel: str, filename: str) -> None:
        available = [(algorithm, numeric(field, algorithm)) for algorithm in algorithms]
        available = [(algorithm, values) for algorithm, values in available if values]
        if not available:
            return
        labels, data = zip(*available)
        plt.figure(figsize=(11, 6))
        plt.boxplot(data, tick_labels=labels, showfliers=False)
        plt.title(title); plt.ylabel(ylabel); plt.xticks(rotation=25); plt.tight_layout()
        plt.savefig(figures / filename, dpi=200); plt.close()

    boxplot("position_error_m", "Position error by algorithm", "3D ECEF position error (m)", "position_error_comparison.png")
    boxplot("execution_time_seconds", "Execution time by algorithm", "Execution time (s)", "execution_time_comparison.png")
    convergence = {
        algorithm: [row for row in rows if row["algorithm"] == algorithm]
        for algorithm in algorithms
    }
    available_rates = [
        (algorithm, sum(row["is_converged"] == "True" for row in algorithm_rows) / len(algorithm_rows))
        for algorithm, algorithm_rows in convergence.items() if algorithm_rows
    ]
    if available_rates:
        labels, rates = zip(*available_rates)
        plt.clf(); plt.figure(figsize=(11, 6)); plt.bar(labels, rates)
        plt.title("Convergence rate by algorithm"); plt.ylabel("Convergence rate")
        plt.ylim(0, 1); plt.xticks(rotation=25); plt.tight_layout()
    else:
        plt.close()
    if available_rates:
        plt.savefig(figures / "convergence_rate_comparison.png", dpi=200); plt.close()
    for group_field, filename, title in (
        ("noise_std_m", "noise_performance.png", "Position error across noise levels"),
        ("geometry_category", "geometry_performance.png", "Position error across geometry categories"),
        ("initial_perturbation_m", "perturbation_performance.png", "Position error across initial perturbations"),
    ):
        raw_groups = {row[group_field] for row in rows if row[group_field] != ""}
        try:
            groups = sorted(raw_groups, key=float)
        except (TypeError, ValueError):
            groups = sorted(raw_groups)
        plt.figure(figsize=(11, 6))
        for algorithm in algorithms:
            means = []
            for group in groups:
                values = [float(row["position_error_m"]) for row in rows
                          if row["algorithm"] == algorithm and row[group_field] == group
                          and row["position_error_m"] != "" and np.isfinite(float(row["position_error_m"]))]
                means.append(np.mean(values) if values else np.nan)
            if not np.all(np.isnan(means)):
                x_values = [float(value) for value in groups] if group_field != "geometry_category" else groups
                plt.plot(x_values, means, marker="o", label=algorithm)
        plt.title(title); plt.xlabel(group_field); plt.ylabel("Mean position error (m)")
        plt.legend(); plt.tight_layout(); plt.savefig(figures / filename, dpi=200); plt.close()
    hybrid_rows = [row for row in rows if row["algorithm"] in {"fixed_gn_adaptive", "full_hybrid"}]
    if hybrid_rows:
        switch_counts = [sum(float(row["solver_switches"]) for row in hybrid_rows if row["algorithm"] == algorithm)
                         for algorithm in ("fixed_gn_adaptive", "full_hybrid")]
        plt.figure(figsize=(8, 5)); plt.bar(("fixed_gn_adaptive", "full_hybrid"), switch_counts)
        plt.title("Hybrid solver switches"); plt.ylabel("Total switches"); plt.xticks(rotation=20); plt.tight_layout()
        plt.savefig(figures / "hybrid_solver_switch_behavior.png", dpi=200); plt.close()
    ablation = [(algorithm, sum(row["is_converged"] == "True" for row in algorithm_rows) / len(algorithm_rows))
                for algorithm, algorithm_rows in convergence.items() if algorithm_rows]
    if ablation:
        labels, rates = zip(*ablation)
        plt.figure(figsize=(11, 6)); plt.bar(labels, rates); plt.title("Ablation convergence comparison")
        plt.ylabel("Convergence rate"); plt.ylim(0, 1); plt.xticks(rotation=25); plt.tight_layout()
        plt.savefig(figures / "ablation_comparison.png", dpi=200); plt.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run reproducible GPS solver benchmarks")
    parser.add_argument("--output-dir", default="results/benchmark_debug")
    parser.add_argument("--scenario-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--plot", action="store_true", help="create optional matplotlib figures")
    args = parser.parse_args(argv)
    config = BenchmarkConfig(
        scenario_count=args.scenario_count, seed=args.seed, test_fraction=args.test_fraction,
    )
    results = run_benchmark(config, output_dir=args.output_dir)
    if args.plot:
        plot_benchmark_results(args.output_dir)
    print(f"Wrote {len(results.run_rows)} run rows to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALGORITHM_NAMES", "BenchmarkConfig", "BenchmarkResults", "BenchmarkScenario",
    "generate_benchmark_scenario", "generate_benchmark_scenarios", "plot_benchmark_results",
    "run_benchmark", "write_benchmark_outputs",
]
