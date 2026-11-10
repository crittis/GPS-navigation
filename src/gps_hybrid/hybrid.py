"""Hybrid orchestration for the approved GPS numerical solvers.

The orchestrator coordinates complete solver stages because the existing
solver interfaces return a ``SolverResult`` only after a stage finishes; they
do not expose an iteration callback.  It therefore monitors each returned
history, preserves every stage result, and makes at most one acyclic
transition at a time: Gauss-Newton -> Levenberg-Marquardt -> optional BFGS.
The stage iteration budget is passed through ``SolverConfig``.  This enforces
the total iteration budget for the existing solvers, but a future callback
interface would be needed for a hard mid-iteration interruption or a strict
global function-evaluation budget.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real
from time import perf_counter
from typing import Callable

import numpy as np

from .geometry import DEFAULT_RANK_TOLERANCE, GeometryDiagnostics, diagnose_geometry
from .models import GPSObservations
from .selection import SelectionConfig, SolverSelection, select_initial_solver
from .solvers import (
    IterationRecord,
    SolverConfig,
    SolverResult,
    solve_bfgs,
    solve_gauss_newton,
    solve_levenberg_marquardt,
)


SolverFunction = Callable[[GPSObservations, np.ndarray, SolverConfig], SolverResult]
SOLVER_FUNCTIONS: dict[str, SolverFunction] = {
    "Gauss-Newton": solve_gauss_newton,
    "Levenberg-Marquardt": solve_levenberg_marquardt,
    "BFGS": solve_bfgs,
}
SOLVER_ORDER = ("Gauss-Newton", "Levenberg-Marquardt", "BFGS")


def _immutable_state(value: np.ndarray) -> np.ndarray:
    state = np.array(value, dtype=float, copy=True)
    state.setflags(write=False)
    return state


def _valid_state(value: object) -> bool:
    try:
        state = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return state.shape == (4,) and bool(np.all(np.isfinite(state)))


def _valid_real(value: object, *, allow_nan: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return False
    numeric = float(value)
    return bool(np.isfinite(numeric) or (allow_nan and np.isnan(numeric)))


def _valid_nonnegative_real(value: object) -> bool:
    return _valid_real(value) and float(value) >= 0.0


@dataclass(frozen=True)
class HybridConfig:
    """Validated policy and budget for hybrid solver orchestration."""

    solver_config: SolverConfig = SolverConfig()
    selection_config: SelectionConfig = SelectionConfig()
    max_total_iterations: int = 300
    max_transitions: int = 2
    stage_iteration_budget: int = 100
    consecutive_failure_threshold: int = 2
    consecutive_stagnation_threshold: int = 3
    objective_progress_tolerance: float = 1e-12
    enable_bfgs_fallback: bool = True
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE
    initial_solver_override: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.solver_config, SolverConfig):
            raise ValueError("solver_config must be a SolverConfig")
        if not isinstance(self.selection_config, SelectionConfig):
            raise ValueError("selection_config must be a SelectionConfig")
        integer_fields = {
            "max_total_iterations": self.max_total_iterations,
            "max_transitions": self.max_transitions,
            "stage_iteration_budget": self.stage_iteration_budget,
            "consecutive_failure_threshold": self.consecutive_failure_threshold,
            "consecutive_stagnation_threshold": self.consecutive_stagnation_threshold,
        }
        for name, value in integer_fields.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.max_total_iterations < 1:
            raise ValueError("max_total_iterations must be positive")
        if self.max_transitions < 0:
            raise ValueError("max_transitions must be nonnegative")
        if self.stage_iteration_budget < 1:
            raise ValueError("stage_iteration_budget must be positive")
        if self.consecutive_failure_threshold < 1 or self.consecutive_stagnation_threshold < 1:
            raise ValueError("failure and stagnation thresholds must be positive")
        value = self.objective_progress_tolerance
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("objective_progress_tolerance must be a finite real number")
        if value <= 0:
            raise ValueError("objective_progress_tolerance must be positive")
        if not isinstance(self.enable_bfgs_fallback, (bool, np.bool_)):
            raise ValueError("enable_bfgs_fallback must be boolean")
        if isinstance(self.rank_tolerance, (bool, np.bool_)) or not isinstance(self.rank_tolerance, Real):
            raise ValueError("rank_tolerance must be numeric")
        if not np.isfinite(self.rank_tolerance) or not 0 < self.rank_tolerance < 1:
            raise ValueError("rank_tolerance must be greater than zero and less than one")
        if self.initial_solver_override not in (None, "Gauss-Newton"):
            raise ValueError("initial_solver_override must be None or 'Gauss-Newton'")


@dataclass(frozen=True)
class SolverStage:
    """One complete solver invocation and its preserved result."""

    solver_name: str
    result: SolverResult


@dataclass(frozen=True)
class SolverTransition:
    """Auditable record of a warm-started solver transition."""

    previous_solver: str
    next_solver: str
    trigger: str
    rejected_steps: int
    stagnation_steps: int
    condition_number: float
    state: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _immutable_state(self.state))


@dataclass(frozen=True)
class HybridResult:
    """Final hybrid result, including all stages and transition history."""

    estimated_state: np.ndarray
    final_objective: float
    final_residual_rms: float
    convergence_status: str
    failure_reason: str | None
    initial_geometry: GeometryDiagnostics
    final_geometry: GeometryDiagnostics | None
    stage_results: tuple[SolverStage, ...]
    transition_history: tuple[SolverTransition, ...]
    total_iterations: int
    total_function_evaluations: int
    execution_time_seconds: float
    solver_switches: int
    solver_sequence: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimated_state", _immutable_state(self.estimated_state))

    @property
    def converged(self) -> bool:
        return self.convergence_status == "converged"


def _empty_solver_result(state: np.ndarray, status: str, reason: str,
                         elapsed: float = 0.0) -> SolverResult:
    return SolverResult(
        _immutable_state(state), float("nan"), float("nan"), 0, 0, elapsed,
        status, reason, tuple(),
    )


def _tail_rejections(history: tuple[IterationRecord, ...]) -> int:
    count = 0
    for record in reversed(history):
        if record.accepted:
            break
        count += 1
    return count


def _tail_stagnation(history: tuple[IterationRecord, ...], tolerance: float) -> int:
    count = 0
    previous: IterationRecord | None = None
    for record in reversed(history):
        if not record.accepted:
            break
        if previous is not None:
            reduction = previous.objective - record.objective
            scale = max(1.0, abs(previous.objective))
            if reduction <= tolerance * scale:
                count += 1
            else:
                break
        previous = record
    return count


def _stage_trigger(solver_name: str, result: SolverResult, config: HybridConfig) -> tuple[str, int, int] | None:
    rejected = _tail_rejections(result.convergence_history)
    stagnant = _tail_stagnation(result.convergence_history, config.objective_progress_tolerance)
    repeated_rejection = rejected >= config.consecutive_failure_threshold
    repeated_stagnation = stagnant >= config.consecutive_stagnation_threshold
    if solver_name == "Gauss-Newton":
        if result.convergence_status == "numerical_failure":
            return "GN numerical failure; LM is the configured damping fallback", rejected, stagnant
        if repeated_rejection:
            return "GN reached the configured consecutive rejected-step threshold", rejected, stagnant
        if repeated_stagnation:
            return "GN reached the configured consecutive stagnation threshold", rejected, stagnant
    elif solver_name == "Levenberg-Marquardt":
        if repeated_rejection:
            return "LM reached the configured consecutive rejected-step threshold; BFGS fallback enabled", rejected, stagnant
        if repeated_stagnation:
            return "LM reached the configured consecutive stagnation threshold; BFGS fallback enabled", rejected, stagnant
    return None


def _invalid_result_reason(result: object) -> str | None:
    if not isinstance(result, SolverResult):
        return "Solver returned an invalid result object"
    if not _valid_state(result.estimated_state):
        return "Solver returned a non-finite state or a state with the wrong shape"
    for name, value in (("iterations", result.iterations),
                        ("function_evaluations", result.function_evaluations)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
            return f"Solver returned an invalid {name} count"
    if not _valid_real(result.execution_time_seconds) or result.execution_time_seconds < 0:
        return "Solver returned an invalid execution time"
    if not isinstance(result.convergence_status, str) or not result.convergence_status:
        return "Solver returned an invalid convergence status"
    if result.failure_reason is not None and not isinstance(result.failure_reason, str):
        return "Solver returned an invalid failure reason"
    for name, value in (("final_objective", result.final_objective),
                        ("final_residual_rms", result.final_residual_rms)):
        if not _valid_real(value, allow_nan=True):
            return f"Solver returned an invalid {name}"
        if np.isfinite(value) and value < 0:
            return f"Solver returned a negative {name}"
    if result.converged:
        if not np.isfinite(result.final_objective) or not np.isfinite(result.final_residual_rms):
            return "A converged solver result must have finite objective and residual RMS"
    history = result.convergence_history
    if not isinstance(history, tuple):
        return "Solver returned a convergence history that is not a tuple"
    previous_iteration = 0
    for record in history:
        if not isinstance(record, IterationRecord):
            return "Solver returned a convergence history with an invalid record"
        if (isinstance(record.iteration, (bool, np.bool_))
                or not isinstance(record.iteration, Integral)
                or record.iteration < 1 or record.iteration <= previous_iteration):
            return "Solver returned invalid or non-increasing history iteration numbers"
        previous_iteration = record.iteration
        if not isinstance(record.accepted, (bool, np.bool_)):
            return "Solver returned an invalid history accepted flag"
        for name, value in (
            ("objective", record.objective),
            ("residual_rms", record.residual_rms),
            ("gradient_inf_norm", record.gradient_inf_norm),
            ("step_norm", record.step_norm),
            ("step_scale", record.step_scale),
        ):
            if not _valid_nonnegative_real(value):
                return f"Solver returned an invalid history {name}"
        for name, value in (("damping", record.damping), ("gain_ratio", record.gain_ratio)):
            if value is not None and not _valid_real(value):
                return f"Solver returned an invalid history {name}"
    return None


def run_hybrid(observations: GPSObservations, initial_state: np.ndarray,
               config: HybridConfig = HybridConfig()) -> HybridResult:
    """Run the acyclic GN -> LM -> optional BFGS orchestration policy.

    Geometry is diagnosed before selection and before every transition.  A
    transition uses the latest finite stage estimate and never the scenario
    truth or the original estimate.  A solver's ``converged`` status is the
    only route to overall convergence.
    """
    started = perf_counter()
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must be a finite vector of shape (4,)")
    initial_geometry = diagnose_geometry(observations, state, config.rank_tolerance)
    if initial_geometry.insufficient_observations:
        return HybridResult(
            state, float("nan"), float("nan"), "insufficient_observations",
            initial_geometry.explanation, initial_geometry, initial_geometry,
            tuple(), tuple(), 0, 0, perf_counter() - started, 0, tuple(),
        )
    if not initial_geometry.full_rank:
        return HybridResult(
            state, float("nan"), float("nan"), "unobservable_geometry",
            initial_geometry.explanation, initial_geometry, initial_geometry,
            tuple(), tuple(), 0, 0, perf_counter() - started, 0, tuple(),
        )

    if config.initial_solver_override is None:
        selection: SolverSelection = select_initial_solver(initial_geometry, config.selection_config)
    else:
        selection = SolverSelection(
            "gauss_newton", "Gauss-Newton", SOLVER_FUNCTIONS["Gauss-Newton"],
            "Initial solver overridden to Gauss-Newton for the fixed-GN adaptive ablation",
            initial_geometry,
        )
    if not selection.solvable or selection.solver_name is None or selection.solver is None:
        status = "insufficient_observations" if selection.decision == "insufficient_observations" else "unobservable_geometry"
        return HybridResult(
            state, float("nan"), float("nan"), status, selection.reason,
            initial_geometry, initial_geometry, tuple(), tuple(), 0, 0,
            perf_counter() - started, 0, tuple(),
        )

    solver_name = selection.solver_name
    solver_function: SolverFunction = selection.solver  # type: ignore[assignment]
    stages: list[SolverStage] = []
    transitions: list[SolverTransition] = []
    total_iterations = 0
    total_evaluations = 0
    final_geometry: GeometryDiagnostics | None = initial_geometry
    last_result: SolverResult | None = None

    while True:
        remaining = config.max_total_iterations - total_iterations
        if remaining <= 0:
            return _make_result(state, last_result, initial_geometry, final_geometry, stages,
                                transitions, total_iterations, total_evaluations, started,
                                "max_total_iterations", "Maximum total iteration budget reached")
        stage_limit = min(config.stage_iteration_budget, remaining)
        stage_config = replace(config.solver_config, max_iterations=stage_limit)
        stage_started = perf_counter()
        try:
            result = solver_function(observations, state, stage_config)
        except Exception as exc:  # solver failures become explicit hybrid results
            result = _empty_solver_result(
                state, "numerical_failure", f"{solver_name} raised {type(exc).__name__}: {exc}",
                perf_counter() - stage_started,
            )
        reason = _invalid_result_reason(result)
        if reason is not None:
            return _make_result(state, last_result,
                                initial_geometry, final_geometry, stages, transitions,
                                total_iterations, total_evaluations, started,
                                "numerical_failure", reason)
        if result.iterations > stage_limit:
            return _make_result(
                state, last_result, initial_geometry, final_geometry, stages, transitions,
                total_iterations, total_evaluations, started, "budget_integrity_failure",
                f"Solver reported {result.iterations} iterations, exceeding its allocated stage budget of {stage_limit}",
            )
        last_result = result
        stages.append(SolverStage(solver_name, result))
        total_iterations += result.iterations
        total_evaluations += result.function_evaluations
        state = np.asarray(result.estimated_state, dtype=float).copy()
        try:
            final_geometry = diagnose_geometry(observations, state, config.rank_tolerance)
        except ValueError as exc:
            return _make_result(state, result, initial_geometry, None, stages, transitions,
                                total_iterations, total_evaluations, started,
                                "numerical_failure", f"Final geometry could not be diagnosed: {exc}")
        if final_geometry.insufficient_observations or not final_geometry.full_rank:
            return _make_result(state, result, initial_geometry, final_geometry, stages,
                                transitions, total_iterations, total_evaluations, started,
                                "unobservable_geometry", final_geometry.explanation)
        if result.converged:
            return _make_result(state, result, initial_geometry, final_geometry, stages,
                                transitions, total_iterations, total_evaluations, started,
                                "converged", None)
        trigger = _stage_trigger(solver_name, result, config)
        next_solver = {
            "Gauss-Newton": "Levenberg-Marquardt",
            "Levenberg-Marquardt": "BFGS",
        }.get(solver_name)
        if trigger is None or next_solver is None:
            return _make_result(state, result, initial_geometry, final_geometry, stages,
                                transitions, total_iterations, total_evaluations, started,
                                result.convergence_status,
                                result.failure_reason or "Solver did not converge")
        reason_text, rejected, stagnant = trigger
        if next_solver == "BFGS" and not config.enable_bfgs_fallback:
            return _make_result(state, result, initial_geometry, final_geometry, stages,
                                transitions, total_iterations, total_evaluations, started,
                                "solver_failure", f"LM fallback to BFGS is disabled; trigger: {reason_text}")
        if len(transitions) >= config.max_transitions:
            return _make_result(state, result, initial_geometry, final_geometry, stages,
                                transitions, total_iterations, total_evaluations, started,
                                "max_transitions",
                                f"Maximum solver transition count reached; trigger: {reason_text}")
        if total_iterations >= config.max_total_iterations:
            return _make_result(state, result, initial_geometry, final_geometry, stages,
                                transitions, total_iterations, total_evaluations, started,
                                "max_total_iterations",
                                f"Maximum total iteration budget reached; trigger: {reason_text}")
        transitions.append(SolverTransition(
            solver_name, next_solver, reason_text, rejected, stagnant,
            final_geometry.condition_number, state,
        ))
        solver_name = next_solver
        solver_function = SOLVER_FUNCTIONS[solver_name]


def _make_result(state: np.ndarray, result: SolverResult | None,
                 initial_geometry: GeometryDiagnostics,
                 final_geometry: GeometryDiagnostics | None,
                 stages: list[SolverStage], transitions: list[SolverTransition],
                 total_iterations: int, total_evaluations: int, started: float,
                 status: str, reason: str | None) -> HybridResult:
    stage_time = float(sum(stage.result.execution_time_seconds for stage in stages))
    return HybridResult(
        state,
        result.final_objective if result is not None else float("nan"),
        result.final_residual_rms if result is not None else float("nan"),
        status, reason, initial_geometry, final_geometry,
        tuple(stages), tuple(transitions), total_iterations, total_evaluations,
        stage_time if stages else perf_counter() - started,
        len(transitions), tuple(stage.solver_name for stage in stages),
    )


__all__ = [
    "HybridConfig", "HybridResult", "SolverStage", "SolverTransition", "run_hybrid",
]
