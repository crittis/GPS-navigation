"""Levenberg-Marquardt with augmented least squares and gain-ratio control."""

from __future__ import annotations

from math import log
from time import perf_counter

import numpy as np

from ..models import GPSObservations
from .base import (Evaluator, IterationRecord, SolverConfig, SolverResult,
                   build_result, geometry_rank, has_converged, record)


def _failure(state: np.ndarray, residual: np.ndarray, iterations: int,
             evaluator: Evaluator, start: float, reason: str,
             history: list[IterationRecord]) -> SolverResult:
    return build_result(state, residual, iterations, evaluator, start,
                        "numerical_failure", reason, history)


def _finite_array(value: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(value)))


def _safe_step_norm(step: np.ndarray) -> float | None:
    """Return the Euclidean norm only when it is representable as a float."""
    absolute = np.abs(step)
    scale = float(np.max(absolute))
    if not np.isfinite(scale):
        return None
    if scale == 0.0:
        return 0.0
    normalized_norm = float(np.sqrt(np.dot(absolute / scale, absolute / scale)))
    if not np.isfinite(normalized_norm) or scale > np.finfo(float).max / normalized_norm:
        return None
    return scale * normalized_norm


def _append_finite_record(history: list[IterationRecord], iteration: int,
                          residual: np.ndarray, gradient: np.ndarray,
                          step_norm: float, accepted: bool, damping: float,
                          gain_ratio: float) -> bool:
    """Append a history item only if every recorded numeric field is finite."""
    item = record(iteration, residual, gradient, step_norm, accepted,
                  damping=damping, gain_ratio=gain_ratio)
    values = (item.objective, item.residual_rms, item.gradient_inf_norm,
              item.step_norm, item.step_scale, item.damping, item.gain_ratio)
    if not all(value is not None and np.isfinite(value) for value in values):
        return False
    history.append(item)
    return True


def _accepted_damping(damping: float, rho: float, config: SolverConfig) -> float | None:
    """Apply the ordinary Nielsen reduction and validate its finite result."""
    factor = (
        max(config.lm_damping_decrease_factor, 1.0 - (2.0 * rho - 1.0) ** 3)
        if rho < 1.0 else config.lm_damping_decrease_factor
    )
    if not np.isfinite(factor) or factor <= 0.0:
        return None
    with np.errstate(over="raise", invalid="raise"):
        try:
            updated = damping * factor
        except FloatingPointError:
            return None
    if not np.isfinite(updated) or updated < 0.0:
        return None
    return max(updated, np.finfo(float).tiny)


def _compute_reductions(current_objective: float, candidate_objective: float,
                        linearized_residual: np.ndarray) -> tuple[float, float]:
    predicted = current_objective - 0.5 * float(np.dot(linearized_residual, linearized_residual))
    actual = current_objective - candidate_objective
    return predicted, actual


def _gain_ratio(actual: float, predicted: float) -> float:
    if actual == 0.0:
        return 0.0
    if log(abs(actual)) - log(predicted) > log(np.finfo(float).max):
        return float("inf") if actual > 0.0 else float("-inf")
    return float(actual / predicted)


def _initial_damping_scales(jacobian: np.ndarray) -> np.ndarray | None:
    """Compute diag(J.T J) safely; return None if it is not representable."""
    if not _finite_array(jacobian):
        return None
    column_max = np.max(np.abs(jacobian), axis=0)
    scales = np.ones(jacobian.shape[1], dtype=float)
    max_float = np.finfo(float).max
    for column, maximum in enumerate(column_max):
        if maximum == 0.0:
            continue
        normalized = jacobian[:, column] / maximum
        normalized_sum = float(np.dot(normalized, normalized))
        # Check before squaring the scale, avoiding an overflow intermediate.
        if maximum > np.sqrt(max_float / normalized_sum):
            return None
        scales[column] = max(float(maximum * maximum * normalized_sum), 1.0)
    return scales if _finite_array(scales) else None


def solve_levenberg_marquardt(observations: GPSObservations, initial_state,
                              config: SolverConfig = SolverConfig()) -> SolverResult:
    """Damp with diag(J.T J), solved as an augmented least-squares problem.

    Initial lambda is ``lm_initial_damping`` times the largest diagonal
    curvature. Accepted steps retain the Nielsen gain-ratio damping update;
    rejected steps multiply lambda by a growing ``nu`` (2, 4, 8, ...).
    Damping is not arbitrarily capped: if a required update would make the
    augmented diagonal unrepresentable, the solver returns numerical failure
    before multiplication. The representability bound is derived from the
    largest finite float and the current diagonal scaling.
    """
    start = perf_counter()
    evaluator = Evaluator(observations)
    state = np.asarray(initial_state, dtype=float)
    if state.shape != (4,) or not _finite_array(state):
        raise ValueError("Initial receiver state must be a finite vector of shape (4,)")
    state = state.copy()
    history: list[IterationRecord] = []

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            r, f = evaluator.objective(state)
            j = evaluator.jacobian(state)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError, OverflowError) as exc:
        return _failure(state, np.zeros(observations.pseudoranges.size), 0,
                        evaluator, start, f"Initial LM evaluation failed: {exc}", history)
    if not _finite_array(r) or not np.isfinite(f) or not _finite_array(j):
        return _failure(state, r, 0, evaluator, start,
                        "Initial LM residual, objective, or Jacobian is non-finite", history)
    try:
        rank, _ = geometry_rank(j, config.rank_rcond)
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        return _failure(state, r, 0, evaluator, start,
                        f"Initial LM Jacobian rank calculation failed: {exc}", history)
    if rank < 4:
        return build_result(state, r, 0, evaluator, start, "rank_deficient",
                            f"Jacobian rank is {rank}; four independent state directions are required", history)

    scales = _initial_damping_scales(j)
    if scales is None or not _finite_array(scales) or np.any(scales <= 0.0):
        return _failure(state, r, 0, evaluator, start,
                        "LM Jacobian scaling is non-finite or not representable", history)
    try:
        with np.errstate(over="raise", invalid="raise"):
            damping = config.lm_initial_damping * max(float(np.max(scales)), 1.0)
    except FloatingPointError:
        return _failure(state, r, 0, evaluator, start,
                        "Initial LM damping is not representable", history)
    if not np.isfinite(damping) or damping <= 0.0:
        return _failure(state, r, 0, evaluator, start,
                        "Initial LM damping is non-finite or not representable", history)

    nu = 2.0
    max_float = np.finfo(float).max
    for iteration in range(1, config.max_iterations + 1):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                gradient = j.T @ r
                if not _finite_array(gradient):
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    "LM gradient is non-finite", history)
                max_scale = float(np.max(scales))
                # The augmented diagonal is sqrt(lambda * scale). This bound
                # keeps its squared argument finite without a heuristic cap.
                if damping > max_float / max_scale:
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    "LM damping and scaling exceed floating-point range", history)
                damping_diagonal = np.sqrt(damping * scales)
                augmented_j = np.vstack((j, np.diag(damping_diagonal)))
                augmented_rhs = np.concatenate((-r, np.zeros(4)))
                if not _finite_array(augmented_j) or not _finite_array(augmented_rhs):
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    "Damped LM system contains non-finite values", history)
                # A common positive scaling leaves the least-squares minimizer
                # unchanged and keeps SVD inputs away from extreme magnitudes.
                system_scale = max(float(np.max(np.abs(augmented_j))),
                                   float(np.max(np.abs(augmented_rhs))), 1.0)
                delta, _, aug_rank, _ = np.linalg.lstsq(
                    augmented_j / system_scale,
                    augmented_rhs / system_scale,
                    rcond=config.rank_rcond,
                )
                if aug_rank < 4 or not _finite_array(delta):
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    "Damped least-squares system produced an invalid step", history)
                step_norm = _safe_step_norm(delta)
                if step_norm is None:
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    "LM step norm exceeds floating-point range", history)
                try:
                    candidate = state + delta
                except FloatingPointError as exc:
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    f"LM candidate state is not representable: {exc}", history)
                if not _finite_array(candidate):
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    "LM candidate state is non-finite", history)
                candidate_r, candidate_f = evaluator.objective(candidate)
                if not _finite_array(candidate_r) or not np.isfinite(candidate_f):
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    "LM candidate residual or objective is non-finite", history)
                linearized_residual = r + j @ delta
                if not _finite_array(linearized_residual):
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    "LM predicted residual is non-finite", history)
                predicted, actual = _compute_reductions(f, candidate_f, linearized_residual)
                if not np.isfinite(predicted) or not np.isfinite(actual):
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    "LM predicted or actual reduction is non-finite", history)
                # A finite non-positive prediction is a rejected trial under
                # the established policy; encode its undefined ratio as zero.
                rho = _gain_ratio(actual, predicted) if predicted > 0.0 else 0.0
                if not np.isfinite(rho):
                    return _failure(state, r, iteration - 1, evaluator, start,
                                    "LM gain ratio is non-finite", history)
                accepted = rho > config.lm_acceptance_threshold and actual > 0.0
        except (np.linalg.LinAlgError, FloatingPointError, ValueError, OverflowError) as exc:
            return _failure(state, r, iteration - 1, evaluator, start,
                            f"LM numerical calculation failed: {exc}", history)

        if accepted:
            step = delta
            previous_f = f
            state, r, f = candidate, candidate_r, candidate_f
            try:
                j = evaluator.jacobian(state)
                if not _finite_array(j):
                    return _failure(state, r, iteration, evaluator, start,
                                    "LM Jacobian at candidate state is non-finite", history)
                rank, _ = geometry_rank(j, config.rank_rcond)
                scales = _initial_damping_scales(j)
                if scales is None:
                    return _failure(state, r, iteration, evaluator, start,
                                    "LM scaling at candidate state is non-finite or not representable", history)
                updated_damping = _accepted_damping(damping, rho, config)
                if updated_damping is None or not np.isfinite(updated_damping) or updated_damping <= 0.0:
                    return _failure(state, r, iteration, evaluator, start,
                                    "LM damping reduction produced an invalid value", history)
                damping = updated_damping
                nu = 2.0
                new_gradient = j.T @ r
                if not _finite_array(new_gradient):
                    return _failure(state, r, iteration, evaluator, start,
                                    "LM gradient at candidate state is non-finite", history)
                if not _append_finite_record(history, iteration, r, new_gradient, step_norm, True,
                                             damping, float(rho)):
                    return _failure(state, r, iteration, evaluator, start,
                                    "LM accepted iteration history would contain a non-finite field", history)
                if rank < 4:
                    return build_result(state, r, iteration, evaluator, start, "rank_deficient",
                                        "Jacobian became rank deficient", history)
                if has_converged(config, new_gradient, step, state, actual, previous_f, j, r):
                    return build_result(state, r, iteration, evaluator, start, "converged", None, history)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError, OverflowError) as exc:
                return _failure(state, r, iteration, evaluator, start,
                                f"LM accepted-step update failed: {exc}", history)
        else:
            if not _append_finite_record(history, iteration, r, gradient, step_norm, False,
                                         damping, float(rho)):
                return _failure(state, r, iteration, evaluator, start,
                                "LM rejected iteration history would contain a non-finite field", history)
            # Don't mistake a tiny damping-induced step for stationarity: test
            # the undamped local least-squares step as before.
            try:
                gn_delta, _, gn_rank, _ = np.linalg.lstsq(j, -r, rcond=config.rank_rcond)
                if not _finite_array(gn_delta):
                    return _failure(state, r, iteration, evaluator, start,
                                    "Undamped LM diagnostic step is non-finite", history)
                gn_linearized = r + j @ gn_delta
                if not _finite_array(gn_linearized):
                    return _failure(state, r, iteration, evaluator, start,
                                    "Undamped LM predicted residual is non-finite", history)
                gn_predicted = f - 0.5 * float(np.dot(gn_linearized, gn_linearized))
                if not np.isfinite(gn_predicted):
                    return _failure(state, r, iteration, evaluator, start,
                                    "Undamped LM predicted reduction is non-finite", history)
                if gn_rank == 4 and has_converged(
                    config, gradient, gn_delta, state, gn_predicted, f, j, r
                ):
                    return build_result(state, r, iteration, evaluator, start, "converged", None, history)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError, OverflowError) as exc:
                return _failure(state, r, iteration, evaluator, start,
                                f"Undamped LM stationarity check failed: {exc}", history)
            update_factor = nu
            if (not np.isfinite(update_factor) or update_factor < 1.0
                    or damping > max_float / update_factor):
                return _failure(state, r, iteration, evaluator, start,
                                "LM damping update would exceed floating-point range", history)
            damping *= update_factor
            if not np.isfinite(damping):
                return _failure(state, r, iteration, evaluator, start,
                                "LM damping update became non-finite", history)
            if nu > max_float / config.lm_damping_increase:
                return _failure(state, r, iteration, evaluator, start,
                                "LM rejection multiplier would exceed floating-point range", history)
            nu *= config.lm_damping_increase

    return build_result(state, r, config.max_iterations, evaluator, start, "max_iterations",
                        "Maximum iteration count reached", history)
