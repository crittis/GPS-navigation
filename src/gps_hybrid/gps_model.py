"""GPS pseudorange residual model and analytical derivatives (ECEF metres)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .models import GPSObservations


def _validated_inputs(state: ArrayLike, observations: GPSObservations) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(state, dtype=float)
    satellites = np.asarray(observations.satellite_positions_ecef, dtype=float)
    measured = np.asarray(observations.pseudoranges, dtype=float)
    if u.shape != (4,) or not np.all(np.isfinite(u)):
        raise ValueError("Receiver state must be a finite vector of shape (4,)")
    if satellites.ndim != 2 or satellites.shape[1:] != (3,) or satellites.shape[0] < 4:
        raise ValueError("Satellite positions must have shape (N, 3) with N >= 4")
    if measured.shape != (satellites.shape[0],):
        raise ValueError("There must be one pseudorange per satellite")
    if not np.all(np.isfinite(satellites)) or not np.all(np.isfinite(measured)):
        raise ValueError("Observations must contain only finite values")
    differences = satellites - u[:3]
    distances = np.linalg.norm(differences, axis=1)
    if np.any(distances == 0.0):
        raise ValueError("Receiver estimate coincides with a satellite position")
    if not np.all(np.isfinite(distances)):
        raise FloatingPointError("Satellite ranges are non-finite")
    return u, differences, distances


def residual(state: ArrayLike, observations: GPSObservations) -> np.ndarray:
    """Return r_i = rho_i - ||s_i-p|| - b, in metres."""
    u, _, distances = _validated_inputs(state, observations)
    return np.asarray(observations.pseudoranges, dtype=float) - distances - u[3]


def jacobian(state: ArrayLike, observations: GPSObservations) -> np.ndarray:
    """Analytical derivative of the residual vector with respect to [x,y,z,b]."""
    u, differences, distances = _validated_inputs(state, observations)
    del u
    return np.column_stack((differences / distances[:, None], -np.ones(distances.size)))


def objective(state: ArrayLike, observations: GPSObservations) -> float:
    """Return F(u) = 0.5 * r.T @ r."""
    r = residual(state, observations)
    return float(0.5 * np.dot(r, r))
