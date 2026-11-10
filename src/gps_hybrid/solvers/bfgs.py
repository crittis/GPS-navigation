"""BFGS with Armijo backtracking and curvature-safe Hessian updates."""

from __future__ import annotations

import numpy as np

from ..models import GPSObservations
from .base import (IterationRecord, SolverConfig, SolverResult, build_result,
                   geometry_rank, has_converged, initialize, record)


def solve_bfgs(observations: GPSObservations, initial_state,
               config: SolverConfig = SolverConfig()) -> SolverResult:
    """Minimize the same least-squares objective with direct-Hessian BFGS.

    The search uses Armijo backtracking, requires a descent direction, and
    resets its Hessian approximation when curvature becomes unreliable.
    """
    start, evaluator, state, r, f, j, rank, _ = initialize(observations, initial_state, config)
    history: list[IterationRecord] = []
    if rank < 4:
        return build_result(state, r, 0, evaluator, start, "rank_deficient",
                            f"Jacobian rank is {rank}; four independent state directions are required", history)
    hessian = np.eye(4)
    for iteration in range(1, config.max_iterations + 1):
        try:
            gradient = j.T @ r
            direction = np.linalg.solve(hessian, -gradient)
            if not np.all(np.isfinite(direction)) or float(gradient @ direction) >= 0:
                hessian = np.eye(4)
                direction = -gradient
            directional = float(gradient @ direction)
            alpha = 1.0
            accepted = False
            while alpha >= config.line_search_min_step:
                candidate = state + alpha * direction
                candidate_r, candidate_f = evaluator.objective(candidate)
                if np.isfinite(candidate_f) and candidate_f <= f + config.line_search_armijo * alpha * directional:
                    accepted = True
                    break
                alpha *= config.line_search_contraction
            if not accepted:
                history.append(record(iteration, r, gradient, float(np.linalg.norm(direction)), False, alpha))
                return build_result(state, r, iteration - 1, evaluator, start, "line_search_failed",
                                    "Armijo backtracking did not find an acceptable step", history)
            new_j = evaluator.jacobian(candidate)
            new_gradient = new_j.T @ candidate_r
            s = alpha * direction
            y = new_gradient - gradient
            sy = float(s @ y)
            if sy > np.finfo(float).eps * np.linalg.norm(s) * np.linalg.norm(y):
                hy = hessian @ s
                s_h_s = float(s @ hy)
                if s_h_s > np.finfo(float).tiny:
                    hessian += np.outer(y, y) / sy - np.outer(hy, hy) / s_h_s
                    hessian = 0.5 * (hessian + hessian.T)
                else:
                    hessian = np.eye(4)
            else:
                hessian = np.eye(4)
            old_f = f
            state, r, f, j = candidate, candidate_r, candidate_f, new_j
            rank, _ = geometry_rank(j, config.rank_rcond)
            history.append(record(iteration, r, new_gradient, float(np.linalg.norm(s)), True, alpha))
            if rank < 4:
                return build_result(state, r, iteration, evaluator, start, "rank_deficient",
                                    "Jacobian became rank deficient", history)
            if has_converged(config, new_gradient, s, state, old_f - f, old_f, j, r):
                return build_result(state, r, iteration, evaluator, start, "converged", None, history)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
            return build_result(state, r, iteration - 1, evaluator, start, "numerical_failure", str(exc), history)
    return build_result(state, r, config.max_iterations, evaluator, start, "max_iterations",
                        "Maximum iteration count reached", history)
