"""Initial solver selection from explicit geometry diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable

import numpy as np

from .geometry import GeometryDiagnostics
from .solvers import SolverResult, solve_gauss_newton, solve_levenberg_marquardt


Solver = Callable[..., SolverResult]


@dataclass(frozen=True)
class SelectionConfig:
    """Engineering thresholds for initial solver selection.

    ``condition_number_threshold`` is deliberately configurable.  It is an
    engineering policy value to calibrate with later benchmarking, not a
    universal mathematical constant.
    """

    condition_number_threshold: float = 1e8

    def __post_init__(self) -> None:
        value = self.condition_number_threshold
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("condition_number_threshold must be a finite real number")
        if value < 1.0:
            raise ValueError("condition_number_threshold must be at least one")


@dataclass(frozen=True)
class SolverSelection:
    """A safe selection decision and the reason supporting it."""

    decision: str
    solver_name: str | None
    solver: Solver | None
    reason: str
    diagnostics: GeometryDiagnostics

    @property
    def solvable(self) -> bool:
        return self.solver is not None


def select_initial_solver(diagnostics: GeometryDiagnostics,
                          config: SelectionConfig = SelectionConfig()) -> SolverSelection:
    """Choose an initial solver without performing runtime solver switching."""
    if not isinstance(diagnostics, GeometryDiagnostics):
        raise ValueError("diagnostics must be a GeometryDiagnostics instance")
    if diagnostics.insufficient_observations:
        return SolverSelection(
            "insufficient_observations", None, None,
            "Insufficient observations: at least four satellites are required for four unknowns",
            diagnostics,
        )
    if not diagnostics.full_rank:
        return SolverSelection(
            "unobservable", None, None,
            "Unobservable geometry: the GPS Jacobian is numerically rank deficient",
            diagnostics,
        )
    if diagnostics.condition_number <= config.condition_number_threshold:
        return SolverSelection(
            "gauss_newton", "Gauss-Newton", solve_gauss_newton,
            "Full-rank geometry is within the configured condition-number threshold; select Gauss-Newton",
            diagnostics,
        )
    return SolverSelection(
        "levenberg_marquardt", "Levenberg-Marquardt", solve_levenberg_marquardt,
        "Full-rank geometry exceeds the configured condition-number threshold; select Levenberg-Marquardt",
        diagnostics,
    )


__all__ = ["SelectionConfig", "SolverSelection", "select_initial_solver"]
