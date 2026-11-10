"""Shared solver configuration, results, rank checks and evaluation accounting."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter
import numpy as np

from ..gps_model import jacobian, residual
from ..models import GPSObservations


@dataclass(frozen=True)
class SolverConfig:
    max_iterations: int = 100
    gradient_tolerance: float = 1e-8
    step_tolerance: float = 1e-8
    objective_tolerance: float = 1e-12
    rank_rcond: float = 1e-12
    lm_initial_damping: float = 1e-3
    lm_acceptance_threshold: float = 1e-4
    lm_damping_increase: float = 2.0
    lm_damping_decrease_factor: float = 1.0 / 3.0
    line_search_contraction: float = 0.5
    line_search_armijo: float = 1e-4
    line_search_min_step: float = 1e-12

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, (bool, np.bool_)) or not isinstance(self.max_iterations, Integral) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        numeric = {
            "gradient_tolerance": self.gradient_tolerance,
            "step_tolerance": self.step_tolerance,
            "objective_tolerance": self.objective_tolerance,
            "rank_rcond": self.rank_rcond,
            "lm_initial_damping": self.lm_initial_damping,
            "lm_acceptance_threshold": self.lm_acceptance_threshold,
            "lm_damping_increase": self.lm_damping_increase,
            "lm_damping_decrease_factor": self.lm_damping_decrease_factor,
            "line_search_contraction": self.line_search_contraction,
            "line_search_armijo": self.line_search_armijo,
            "line_search_min_step": self.line_search_min_step,
        }
        for name, value in numeric.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        positive = (self.gradient_tolerance, self.step_tolerance, self.objective_tolerance,
                    self.rank_rcond, self.lm_initial_damping, self.lm_acceptance_threshold,
                    self.lm_damping_increase, self.line_search_min_step)
        if any(value <= 0 for value in positive):
            raise ValueError("Solver tolerances and damping constants must be positive")
        if self.lm_damping_increase < 1.0:
            raise ValueError("lm_damping_increase must be at least one")
        if not 0 < self.rank_rcond <= 1:
            raise ValueError("rank_rcond must be greater than zero and at most one")
        if not 0 < self.lm_damping_decrease_factor < 1:
            raise ValueError("lm_damping_decrease_factor must be between zero and one")
        if not 0 < self.line_search_contraction < 1:
            raise ValueError("line_search_contraction must be between zero and one")
        if not 0 < self.line_search_armijo < 1:
            raise ValueError("line_search_armijo must be between zero and one")
        if self.line_search_min_step > 1:
            raise ValueError("line_search_min_step must be at most one")


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    objective: float
    residual_rms: float
    gradient_inf_norm: float
    step_norm: float
    accepted: bool
    step_scale: float = 1.0
    damping: float | None = None
    gain_ratio: float | None = None


@dataclass(frozen=True)
class SolverResult:
    estimated_state: np.ndarray
    final_objective: float
    final_residual_rms: float
    iterations: int
    function_evaluations: int
    execution_time_seconds: float
    convergence_status: str
    failure_reason: str | None
    convergence_history: tuple[IterationRecord, ...]

    @property
    def converged(self) -> bool:
        return self.convergence_status == "converged"


class Evaluator:
    """Count residual/objective evaluations in one solve."""

    def __init__(self, observations: GPSObservations):
        self.observations = observations
        self.count = 0

    def residual(self, state: np.ndarray) -> np.ndarray:
        self.count += 1
        return residual(state, self.observations)

    def objective(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        r = self.residual(state)
        return r, float(0.5 * np.dot(r, r))

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        return jacobian(state, self.observations)


def geometry_rank(j: np.ndarray, rcond: float) -> tuple[int, float]:
    singular_values = np.linalg.svd(j, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0, float("inf")
    rank = int(np.count_nonzero(singular_values > rcond * singular_values[0]))
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    return rank, condition


def build_result(state: np.ndarray, r: np.ndarray, iterations: int, evaluator: Evaluator,
                 start_time: float, status: str, reason: str | None,
                 history: list[IterationRecord]) -> SolverResult:
    state_copy = np.array(state, dtype=float, copy=True)
    state_copy.setflags(write=False)
    rms = float(np.sqrt(np.mean(np.square(r)))) if r.size else float("nan")
    return SolverResult(state_copy, float(0.5 * np.dot(r, r)), rms, iterations,
                        evaluator.count, perf_counter() - start_time, status, reason, tuple(history))


def initialize(observations: GPSObservations, initial_state: np.ndarray, config: SolverConfig):
    start = perf_counter()
    evaluator = Evaluator(observations)
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError("Initial receiver state must be a finite vector of shape (4,)")
    state = state.copy()
    r, f = evaluator.objective(state)
    j = evaluator.jacobian(state)
    rank, condition = geometry_rank(j, config.rank_rcond)
    return start, evaluator, state, r, f, j, rank, condition


def record(iteration: int, r: np.ndarray, gradient: np.ndarray, step_norm: float,
           accepted: bool, step_scale: float = 1.0, damping: float | None = None,
           gain_ratio: float | None = None) -> IterationRecord:
    return IterationRecord(iteration, float(0.5 * r @ r), float(np.sqrt(np.mean(r * r)),),
                           float(np.linalg.norm(gradient, ord=np.inf)), step_norm,
                           accepted, step_scale, damping, gain_ratio)


def has_converged(config: SolverConfig, gradient: np.ndarray, step: np.ndarray,
                  state: np.ndarray, objective_reduction: float,
                  previous_objective: float, j: np.ndarray, r: np.ndarray) -> bool:
    """Return whether a solver has met a first-order or joint step/objective test.

    Position and clock-bias coordinates are measured in metres but have very
    different useful scales.  A raw ``||step||`` therefore lets the ECEF
    coordinates dominate the clock component.  We use a dimensionless norm
    with a 6,400 km ECEF scale and a 1 km clock-bias scale, then apply the
    usual relative-to-state factor.  With the default ``step_tolerance`` this
    corresponds to roughly 6.4 cm in position or 0.01 mm in bias near the
    origin of the scaled state.  A small step is never sufficient by itself:
    it must also have a small objective reduction; the gradient test remains
    an independent first-order stopping criterion.
    """
    gradient_scale = max(1.0, float(np.linalg.norm(j, ord=2) * np.linalg.norm(r)))
    grad_ok = np.linalg.norm(gradient, ord=np.inf) / gradient_scale <= config.gradient_tolerance
    scales = np.array([6_400_000.0, 6_400_000.0, 6_400_000.0, 1_000.0])
    scaled_step = float(np.linalg.norm(np.asarray(step, dtype=float) / scales))
    scaled_state = float(np.linalg.norm(np.asarray(state, dtype=float) / scales))
    step_ok = scaled_step <= config.step_tolerance * max(1.0, scaled_state)
    f_ok = abs(objective_reduction) <= config.objective_tolerance * max(1.0, abs(previous_objective))
    return bool(grad_ok or (step_ok and f_ok))


def invalid_geometry_result(start, evaluator, state, r, iterations, history, status, reason):
    return build_result(state, r, iterations, evaluator, start, status, reason, history)
