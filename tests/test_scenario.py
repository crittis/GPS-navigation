import numpy as np
import pytest

from gps_hybrid.coordinates import (
    ecef_to_geodetic,
    enu_direction_from_azimuth_elevation,
    enu_to_ecef_rotation,
    geodetic_to_ecef,
    WGS84_A,
    WGS84_E2,
)
from gps_hybrid.models import GPSObservations, Scenario, ScenarioTruth
from gps_hybrid.scenario import ScenarioConfig, generate_scenario, validate_scenario, validate_scenario_config


def test_geodetic_ecef_round_trip():
    expected = (np.deg2rad(37.25), np.deg2rad(-122.1), 42.0)
    actual = ecef_to_geodetic(geodetic_to_ecef(*expected))
    assert actual[0] == pytest.approx(expected[0], abs=1e-10)
    assert actual[1] == pytest.approx(expected[1], abs=1e-10)
    assert actual[2] == pytest.approx(expected[2], abs=1e-5)


@pytest.mark.parametrize("sign", [1, -1])
def test_exact_poles_use_polar_altitude_formula(sign):
    xyz = np.array([0.0, 0.0, sign * 6_400_000.0])
    latitude, longitude, altitude = ecef_to_geodetic(xyz)
    assert latitude == pytest.approx(sign * np.pi / 2)
    assert longitude == 0.0
    assert altitude == pytest.approx(6_400_000.0 - WGS84_A * np.sqrt(1 - WGS84_E2))


@pytest.mark.parametrize("latitude_deg", [89.99999, -89.99999])
def test_near_polar_geodetic_round_trip(latitude_deg):
    expected = (np.deg2rad(latitude_deg), np.deg2rad(73.0), 125.0)
    actual = ecef_to_geodetic(geodetic_to_ecef(*expected))
    assert actual[0] == pytest.approx(expected[0], abs=1e-10)
    assert actual[1] == pytest.approx(expected[1], abs=1e-10)
    assert actual[2] == pytest.approx(expected[2], abs=1e-5)


def test_enu_frame_is_orthonormal_and_direction_convention():
    rotation = enu_to_ecef_rotation(0.4, -1.2)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(enu_direction_from_azimuth_elevation(0, 0), [0, 1, 0])
    np.testing.assert_allclose(
        enu_direction_from_azimuth_elevation(np.pi / 2, 0), [1, 0, 0], atol=1e-15
    )


def test_scenario_reproducibility_and_truth_separation():
    a = generate_scenario(ScenarioConfig(seed=123))
    b = generate_scenario(ScenarioConfig(seed=123))
    np.testing.assert_array_equal(a.observations.satellite_positions_ecef,
                                  b.observations.satellite_positions_ecef)
    np.testing.assert_array_equal(a.observations.pseudoranges, b.observations.pseudoranges)
    np.testing.assert_array_equal(a.initial_estimate, b.initial_estimate)
    assert a.truth is not a.observations
    assert not hasattr(a.observations, "receiver_state")
    assert not np.array_equal(a.initial_estimate, a.truth.receiver_state)


def test_generated_ranges_follow_model_with_zero_noise():
    scenario = generate_scenario(ScenarioConfig(seed=8, pseudorange_noise_std_m=0))
    truth = scenario.truth.receiver_state
    expected = np.linalg.norm(scenario.observations.satellite_positions_ecef - truth[:3], axis=1) + truth[3]
    np.testing.assert_allclose(scenario.observations.pseudoranges, expected, atol=1e-8)


def test_validation_rejects_too_few_satellites_and_bad_data():
    with pytest.raises(ValueError, match="four satellites"):
        generate_scenario(ScenarioConfig(satellite_count=3))
    scenario = generate_scenario()
    bad = scenario.observations.pseudoranges.copy()
    bad[0] = np.nan
    from dataclasses import replace
    invalid = replace(scenario, observations=replace(scenario.observations, pseudoranges=bad))
    with pytest.raises(ValueError, match="finite"):
        validate_scenario(invalid)


def test_array_models_copy_inputs_and_expose_read_only_arrays():
    truth_input = np.array([1.0, 2.0, 3.0, 4.0])
    truth = ScenarioTruth(truth_input)
    truth_input[0] = 99.0
    assert truth.receiver_state[0] == 1.0
    with pytest.raises(ValueError):
        truth.receiver_state[0] = 0.0

    satellite_input = np.ones((4, 3))
    range_input = np.full(4, 100.0)
    observations = GPSObservations(satellite_input, range_input)
    satellite_input[0, 0] = 42.0
    range_input[0] = 42.0
    assert observations.satellite_positions_ecef[0, 0] == 1.0
    assert observations.pseudoranges[0] == 100.0
    with pytest.raises(ValueError):
        observations.pseudoranges[0] = 0.0

    initial_input = np.array([1.0, 2.0, 3.0, 4.0])
    scenario = Scenario(truth, observations, initial_input, 2)
    initial_input[0] = 55.0
    assert scenario.initial_estimate[0] == 1.0
    with pytest.raises(ValueError):
        scenario.initial_estimate[0] = 0.0


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"seed": -1}, {"seed": 1.5},
    {"satellite_count": True}, {"satellite_count": 4.5}, {"satellite_count": "8"},
    {"latitude_deg": float("nan")}, {"longitude_deg": "zero"},
    {"satellite_range_m": 0.0}, {"pseudorange_noise_std_m": -1.0},
])
def test_config_validation_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        validate_scenario_config(ScenarioConfig(**kwargs))


def test_scenario_rejects_satellite_at_true_receiver():
    scenario = generate_scenario()
    sats = scenario.observations.satellite_positions_ecef.copy()
    sats[0] = scenario.truth.receiver_state[:3]
    invalid = Scenario(
        scenario.truth,
        GPSObservations(sats, scenario.observations.pseudoranges),
        scenario.initial_estimate,
        scenario.seed,
    )
    with pytest.raises(ValueError, match="coincide"):
        validate_scenario(invalid)


def test_validation_preserves_rank_deficient_geometry():
    scenario = generate_scenario()
    truth_position = scenario.truth.receiver_state[:3]
    # Four distinct satellites on one ray: geometrically degenerate, but valid
    # observation data that must reach a future solver/selector unchanged.
    offsets = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1)) * np.arange(1.0, 5.0)[:, None]
    sats = truth_position + offsets
    invalid_for_solver = Scenario(
        scenario.truth,
        GPSObservations(sats, np.full(4, 22_000_000.0)),
        scenario.initial_estimate,
        scenario.seed,
    )
    validate_scenario(invalid_for_solver)
