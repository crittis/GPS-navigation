"""Controlled synthetic GPS scenario generation and input validation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .coordinates import (
    enu_direction_from_azimuth_elevation,
    enu_to_ecef_rotation,
    ecef_to_geodetic,
    geodetic_to_ecef,
)
from .models import GPSObservations, Scenario, ScenarioTruth


@dataclass(frozen=True)
class ScenarioConfig:
    """Controls a synthetic local-geometry scenario; angles are degrees."""

    seed: int = 0
    satellite_count: int = 8
    latitude_deg: float = 37.0
    longitude_deg: float = -122.0
    altitude_m: float = 20.0
    clock_bias_m: float = 75.0
    satellite_range_m: float = 22_000_000.0
    min_elevation_deg: float = 10.0
    max_elevation_deg: float = 80.0
    pseudorange_noise_std_m: float = 2.0
    initial_position_perturbation_std_m: float = 100.0
    initial_bias_perturbation_std_m: float = 100.0


def validate_scenario_config(config: ScenarioConfig) -> None:
    """Raise ValueError for invalid generation settings."""
    if not isinstance(config, ScenarioConfig):
        raise ValueError("config must be a ScenarioConfig")
    if isinstance(config.seed, (bool, np.bool_)) or not isinstance(config.seed, Integral):
        raise ValueError("seed must be a nonnegative integer")
    if config.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(config.satellite_count, (bool, np.bool_)) or not isinstance(config.satellite_count, Integral):
        raise ValueError("satellite_count must be an integer of at least four")
    if config.satellite_count < 4:
        raise ValueError("At least four satellites are required for four unknowns")
    finite_values = [
        config.latitude_deg, config.longitude_deg, config.altitude_m,
        config.clock_bias_m, config.satellite_range_m, config.min_elevation_deg,
        config.max_elevation_deg, config.pseudorange_noise_std_m,
        config.initial_position_perturbation_std_m, config.initial_bias_perturbation_std_m,
    ]
    numeric_names = [
        "latitude_deg", "longitude_deg", "altitude_m", "clock_bias_m",
        "satellite_range_m", "min_elevation_deg", "max_elevation_deg",
        "pseudorange_noise_std_m", "initial_position_perturbation_std_m",
        "initial_bias_perturbation_std_m",
    ]
    for name, value in zip(numeric_names, finite_values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not -90.0 <= config.latitude_deg <= 90.0:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not -180.0 <= config.longitude_deg <= 180.0:
        raise ValueError("Longitude must be between -180 and 180 degrees")
    if config.satellite_range_m <= 0:
        raise ValueError("Satellite range must be positive")
    if not 0 <= config.min_elevation_deg < config.max_elevation_deg <= 90:
        raise ValueError("Require 0 <= min elevation < max elevation <= 90 degrees")
    if min(config.pseudorange_noise_std_m, config.initial_position_perturbation_std_m,
           config.initial_bias_perturbation_std_m) < 0:
        raise ValueError("Noise and perturbation standard deviations must be nonnegative")


def generate_scenario(config: ScenarioConfig = ScenarioConfig()) -> Scenario:
    """Generate observations, truth, and a separate controlled initial estimate.

    Satellites are placed on controlled azimuth/elevation rays in the receiver's
    local ENU frame at a fixed slant range, then transformed to ECEF. This is a
    geometry simulator, not an orbital ephemeris or visibility simulator.
    """
    validate_scenario_config(config)
    rng = np.random.default_rng(config.seed)
    lat, lon = np.deg2rad([config.latitude_deg, config.longitude_deg])
    receiver = geodetic_to_ecef(float(lat), float(lon), config.altitude_m)
    truth_state = np.array([*receiver, config.clock_bias_m], dtype=float)

    azimuths = rng.uniform(0.0, 2.0 * np.pi, config.satellite_count)
    elevations = np.deg2rad(
        rng.uniform(config.min_elevation_deg, config.max_elevation_deg, config.satellite_count)
    )
    rotation = enu_to_ecef_rotation(float(lat), float(lon))
    directions_enu = np.stack(
        [enu_direction_from_azimuth_elevation(a, e) for a, e in zip(azimuths, elevations)]
    )
    directions_ecef = directions_enu @ rotation.T
    satellites = receiver + config.satellite_range_m * directions_ecef
    ranges = np.linalg.norm(satellites - receiver, axis=1)
    pseudoranges = ranges + config.clock_bias_m + rng.normal(
        0.0, config.pseudorange_noise_std_m, config.satellite_count
    )
    observations = GPSObservations(satellites, pseudoranges)

    perturbation = rng.normal(0.0, config.initial_position_perturbation_std_m, 3)
    bias_perturbation = rng.normal(0.0, config.initial_bias_perturbation_std_m)
    initial = truth_state + np.array([*perturbation, bias_perturbation])
    scenario = Scenario(
        truth=ScenarioTruth(truth_state.copy()),
        observations=GPSObservations(satellites.copy(), pseudoranges.copy()),
        initial_estimate=initial,
        seed=config.seed,
    )
    validate_scenario(scenario)
    return scenario


def validate_scenario(scenario: Scenario) -> None:
    """Validate shapes, finiteness, ranges, and separation-ready scenario fields."""
    truth = np.asarray(scenario.truth.receiver_state, dtype=float)
    sats = np.asarray(scenario.observations.satellite_positions_ecef, dtype=float)
    ranges = np.asarray(scenario.observations.pseudoranges, dtype=float)
    initial = np.asarray(scenario.initial_estimate, dtype=float)
    if truth.shape != (4,) or initial.shape != (4,):
        raise ValueError("Truth and initial estimate must have shape (4,)")
    if sats.ndim != 2 or sats.shape[1:] != (3,):
        raise ValueError("Satellite ECEF positions must have shape (N, 3)")
    if ranges.shape != (sats.shape[0],):
        raise ValueError("There must be one pseudorange per satellite")
    if sats.shape[0] < 4:
        raise ValueError("At least four satellite observations are required")
    if not all(np.all(np.isfinite(a)) for a in (truth, sats, ranges, initial)):
        raise ValueError("Scenario arrays must contain only finite values")
    if np.any(ranges <= 0):
        raise ValueError("Pseudoranges must be positive")
    if np.linalg.norm(truth[:3]) < 1.0:
        raise ValueError("Truth receiver ECEF position is invalid")
    if np.linalg.norm(initial[:3]) < 1.0:
        raise ValueError("Initial estimate ECEF position is invalid")
    distances = np.linalg.norm(sats - truth[:3], axis=1)
    if not np.all(np.isfinite(distances)):
        raise ValueError("Satellite geometry relative to truth must be finite")
    if np.any(distances == 0.0):
        raise ValueError("Satellite positions must not coincide with the true receiver position")
