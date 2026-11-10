"""Data models that keep generation truth apart from solver observations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _immutable_array(value: object, *, name: str) -> FloatArray:
    """Return an owned, read-only float64 copy for frozen data records."""
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a numeric array") from exc
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ScenarioTruth:
    """Generation-only truth; coordinates and range bias are in metres (ECEF)."""

    receiver_state: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "receiver_state", _immutable_array(self.receiver_state, name="receiver_state"))


@dataclass(frozen=True)
class GPSObservations:
    """Solver-facing satellite ECEF positions and pseudoranges, all in metres."""

    satellite_positions_ecef: FloatArray
    pseudoranges: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "satellite_positions_ecef", _immutable_array(
            self.satellite_positions_ecef, name="satellite_positions_ecef"
        ))
        object.__setattr__(self, "pseudoranges", _immutable_array(self.pseudoranges, name="pseudoranges"))


@dataclass(frozen=True)
class Scenario:
    """Synthetic experiment bundle. Solvers should receive observations only."""

    truth: ScenarioTruth
    observations: GPSObservations
    initial_estimate: FloatArray
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_estimate", _immutable_array(self.initial_estimate, name="initial_estimate"))
