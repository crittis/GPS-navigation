"""Gauss-Newton using direct SVD least squares and safeguarded backtracking."""

from __future__ import annotations

import numpy as np

from ..models import GPSObservations
from .base import (IterationRecord, SolverConfig, SolverResult, build_result,
                   geometry_rank, has_converged, initialize, record)


def solve_gauss_newton(observations: GPSObservations, initial_state,
                       config: SolverConfig = SolverConfig()) -> SolverResult:
    """Minimize 0.5||r||²; solve min||J delta+r|| by SVD, never normal equations.

    Backtracking is applied only when the full Gauss-Newton step is not a
    descent step in the nonlinear objective. The step's linear least-squares
    problem is solved directly, avoiding the squared condition number of J.T J.
    """
    start, evaluator, state, r, f, j, rank, _ = initialize(observations, initial_state, config)
    history: list[IterationRecord] = []
    if rank < 4:
        return build_result(state, r, 0, evaluator, start, "rank_deficient",
                            f"Jacobian rank is {rank}; four independent state directions are required", history)
    for iteration in range(1, config.max_iterations + 1):
        try:
            delta, _, rank, _ = np.linalg.lstsq(j, -r, rcond=config.rank_rcond)
            if rank < 4 or not np.all(np.isfinite(delta)):
                return build_result(state, r, iteration - 1, evaluator, start, "rank_deficient",
                                    "Jacobian became rank deficient", history)
            gradient = j.T @ r
            alpha = 1.0
            accepted = False
            candidate_r = r
            candidate_f = f
            while alpha >= config.line_search_min_step:
                candidate = state + alpha * delta
                candidate_r, candidate_f = evaluator.objective(candidate)
                if np.isfinite(candidate_f) and candidate_f < f:
                    accepted = True
                    break
                alpha *= config.line_search_contraction
            if not accepted:
                history.append(record(iteration, r, gradient, float(np.linalg.norm(delta)), False,
                                      alpha,))
                predicted_reduction = -float(gradient @ delta) - 0.5 * float(np.linalg.norm(j @ delta) ** 2)
                if has_converged(config, gradient, delta, state, predicted_reduction, f, j, r):
                    return build_result(state, r, iteration, evaluator, start, "converged", None, history)
                return build_result(state, r, iteration - 1, evaluator, start, "line_search_failed",
                                    "No decreasing Gauss-Newton trial step was found", history)
            step = alpha * delta
            reduction = f - candidate_f
            state, r, f = candidate, candidate_r, candidate_f
            j = evaluator.jacobian(state)
            rank, _ = geometry_rank(j, config.rank_rcond)
            history.append(record(iteration, r, j.T @ r, float(np.linalg.norm(step)), True, alpha))
            if rank < 4:
                return build_result(state, r, iteration, evaluator, start, "rank_deficient",
                                    "Jacobian became rank deficient", history)
            if has_converged(config, j.T @ r, step, state, reduction, f + reduction, j, r):
                return build_result(state, r, iteration, evaluator, start, "converged", None, history)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            return build_result(state, r, iteration - 1, evaluator, start, "numerical_failure", str(exc), history)
    return build_result(state, r, config.max_iterations, evaluator, start, "max_iterations",
                        "Maximum iteration count reached", history)
