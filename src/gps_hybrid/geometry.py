"""Satellite-geometry diagnostics for the GPS pseudorange model.

The diagnostics use the existing residual Jacobian with columns ordered as
``[x, y, z, b]``.  Rank is numerical rank: a singular value is retained when
it is greater than ``rank_tolerance * s_max``.  This is a relative SVD
tolerance, so it is independent of the number of observations except through
the singular values of the actual ``N x 4`` matrix.

PDOP is computed in the receiver-local ENU frame (East, North, Up).  The
clock-bias column is retained while forming the covariance, and PDOP is the
square root of the trace of the 3-by-3 position block.  This calculation
assumes that all pseudorange observations have equal variance and that their
measurement errors are independent.  Consequently, the reported PDOP is
geometry-only: it does not include an observation weighting matrix.  The
resulting covariance is a unit-variance geometry covariance, not a prediction
of actual receiver-position error in metres.  Estimating actual position
uncertainty requires an appropriate measurement covariance model.  PDOP is
reported only when all four state directions are numerically observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from .coordinates import ecef_to_geodetic, enu_to_ecef_rotation
from .gps_model import jacobian
from .models import GPSObservations


FloatArray = NDArray[np.float64]
STATE_DIMENSION = 4
DEFAULT_RANK_TOLERANCE = 1e-12


def _validate_rank_tolerance(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError("rank_tolerance must be a finite real number")
    if value <= 0 or value >= 1:
        raise ValueError("rank_tolerance must be greater than zero and less than one")
    return float(value)


def _validate_observations(observations: GPSObservations) -> int:
    if not isinstance(observations, GPSObservations):
        raise ValueError("observations must be a GPSObservations instance")
    satellites = np.asarray(observations.satellite_positions_ecef, dtype=float)
    pseudoranges = np.asarray(observations.pseudoranges, dtype=float)
    if satellites.ndim != 2 or satellites.shape[1:] != (3,):
        raise ValueError("Satellite positions must have shape (N, 3)")
    if pseudoranges.shape != (satellites.shape[0],):
        raise ValueError("There must be one pseudorange per satellite")
    if not np.all(np.isfinite(satellites)) or not np.all(np.isfinite(pseudoranges)):
        raise ValueError("Observations must contain only finite values")
    return int(satellites.shape[0])


@dataclass(frozen=True)
class GeometryDiagnostics:
    """Immutable summary of observability and satellite geometry."""

    satellite_count: int
    numerical_rank: int
    singular_values: FloatArray
    condition_number: float
    pdop: float | None
    full_rank: bool
    insufficient_observations: bool
    rank_tolerance: float
    explanation: str

    def __post_init__(self) -> None:
        values = np.array(self.singular_values, dtype=float, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "singular_values", values)


def _pdop_from_jacobian(j: np.ndarray, receiver_position: np.ndarray,
                        rank_tolerance: float, numerical_rank: int) -> float | None:
    if numerical_rank < STATE_DIMENSION:
        return None
    latitude, longitude, _ = ecef_to_geodetic(receiver_position)
    rotation = enu_to_ecef_rotation(latitude, longitude)
    # J's position columns are ECEF line-of-sight row vectors.  Multiplying by
    # R converts those rows to ENU while preserving the clock column.
    enu_jacobian = np.column_stack((j[:, :3] @ rotation, j[:, 3]))
    _, singular_values, vh = np.linalg.svd(enu_jacobian, full_matrices=False)
    if singular_values.size != STATE_DIMENSION or singular_values[-1] <= rank_tolerance * singular_values[0]:
        return None
    covariance = (vh.T / np.square(singular_values)) @ vh
    pdop_squared = float(np.trace(covariance[:3, :3]))
    if not np.isfinite(pdop_squared) or pdop_squared < 0:
        return None
    return float(np.sqrt(pdop_squared))


def diagnose_geometry(observations: GPSObservations, receiver_state: np.ndarray,
                      rank_tolerance: float = DEFAULT_RANK_TOLERANCE) -> GeometryDiagnostics:
    """Evaluate satellite count, Jacobian rank, conditioning, and ENU PDOP.

    ``receiver_state`` is the supplied current estimate ``[x, y, z, b]`` in
    metres.  A state with fewer than four observations is reported as
    insufficient without attempting a four-column Jacobian rank test.
    Invalid or non-finite inputs raise ``ValueError`` rather than producing a
    plausible-looking diagnostic.
    """
    tolerance = _validate_rank_tolerance(rank_tolerance)
    satellite_count = _validate_observations(observations)
    state = np.asarray(receiver_state, dtype=float)
    if state.shape != (STATE_DIMENSION,) or not np.all(np.isfinite(state)):
        raise ValueError("receiver_state must be a finite vector of shape (4,)")
    if satellite_count < STATE_DIMENSION:
        return GeometryDiagnostics(
            satellite_count, 0, np.empty(0), float("inf"), None, False, True,
            tolerance, "Fewer than four observations are available for four unknowns",
        )

    try:
        matrix = np.asarray(jacobian(state, observations), dtype=float)
    except (FloatingPointError, ValueError) as exc:
        raise ValueError(f"Cannot evaluate geometry Jacobian: {exc}") from exc
    if matrix.shape != (satellite_count, STATE_DIMENSION) or not np.all(np.isfinite(matrix)):
        raise ValueError("Geometry Jacobian must be finite with four state columns")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    cutoff = tolerance * largest
    numerical_rank = int(np.count_nonzero(singular_values > cutoff)) if largest > 0 else 0
    full_rank = numerical_rank == STATE_DIMENSION
    condition = float(singular_values[0] / singular_values[-1]) if full_rank else float("inf")
    pdop = _pdop_from_jacobian(matrix, state[:3], tolerance, numerical_rank)
    if full_rank:
        explanation = "Four state directions are numerically observable"
    else:
        explanation = f"Jacobian is rank deficient: numerical rank is {numerical_rank} of 4"
    return GeometryDiagnostics(
        satellite_count, numerical_rank, singular_values, condition, pdop,
        full_rank, False, tolerance, explanation,
    )


def analyze_geometry(observations: GPSObservations, receiver_state: np.ndarray,
                     rank_tolerance: float = DEFAULT_RANK_TOLERANCE) -> GeometryDiagnostics:
    """Alias for :func:`diagnose_geometry` with a descriptive public name."""
    return diagnose_geometry(observations, receiver_state, rank_tolerance)


__all__ = ["DEFAULT_RANK_TOLERANCE", "GeometryDiagnostics", "analyze_geometry", "diagnose_geometry"]
