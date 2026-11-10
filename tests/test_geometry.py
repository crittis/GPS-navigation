import numpy as np
import pytest

from gps_hybrid.coordinates import enu_to_ecef_rotation
from gps_hybrid.geometry import diagnose_geometry
from gps_hybrid.gps_model import jacobian
from gps_hybrid.models import GPSObservations


RECEIVER = np.array([6_378_137.0, 0.0, 0.0, 75.0])


def observations_from_enu(directions):
    directions = np.asarray(directions, dtype=float)
    rotation = enu_to_ecef_rotation(0.0, 0.0)
    satellites = RECEIVER[:3] + 22_000_000.0 * (directions @ rotation.T)
    ranges = np.linalg.norm(satellites - RECEIVER[:3], axis=1) + RECEIVER[3]
    return GPSObservations(satellites, ranges)


def well_conditioned_observations():
    directions = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0],
    ])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return observations_from_enu(directions)


def test_well_conditioned_geometry_is_full_rank_with_finite_pdop():
    diagnostics = diagnose_geometry(well_conditioned_observations(), RECEIVER)
    assert diagnostics.satellite_count == 4
    assert diagnostics.full_rank
    assert diagnostics.numerical_rank == 4
    assert np.isfinite(diagnostics.condition_number)
    assert diagnostics.pdop is not None and np.isfinite(diagnostics.pdop)


def test_poorly_conditioned_geometry_remains_full_rank():
    directions = np.array([
        [1.0, 0.0, 0.0], [1.0, 1e-4, 1e-4],
        [1.0, -1e-4, 2e-4], [1.0, 2e-4, -1e-4],
    ])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    diagnostics = diagnose_geometry(observations_from_enu(directions), RECEIVER)
    assert diagnostics.full_rank
    assert diagnostics.condition_number > 1e4
    assert diagnostics.pdop is not None


def test_rank_deficient_geometry_has_infinite_condition_and_no_pdop():
    directions = np.tile([1.0, 0.0, 0.0], (4, 1))
    diagnostics = diagnose_geometry(observations_from_enu(directions), RECEIVER)
    assert not diagnostics.full_rank
    assert diagnostics.numerical_rank < 4
    assert diagnostics.condition_number == np.inf
    assert diagnostics.pdop is None


def test_fewer_than_four_observations_is_distinct_from_rank_deficiency():
    observations = observations_from_enu(np.eye(3))
    diagnostics = diagnose_geometry(observations, RECEIVER)
    assert diagnostics.insufficient_observations
    assert diagnostics.numerical_rank == 0
    assert diagnostics.singular_values.size == 0
    assert diagnostics.pdop is None


def test_singular_values_condition_number_and_jacobian_are_consistent():
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)
    expected_singular_values = np.linalg.svd(jacobian(RECEIVER, observations), compute_uv=False)
    np.testing.assert_allclose(diagnostics.singular_values, expected_singular_values)
    assert diagnostics.condition_number == pytest.approx(
        expected_singular_values[0] / expected_singular_values[-1]
    )


def test_pdop_uses_enu_position_block_and_clock_column():
    observations = well_conditioned_observations()
    diagnostics = diagnose_geometry(observations, RECEIVER)
    matrix = jacobian(RECEIVER, observations)
    rotation = enu_to_ecef_rotation(0.0, 0.0)
    enu_matrix = np.column_stack((matrix[:, :3] @ rotation, matrix[:, 3]))
    _, singular_values, vh = np.linalg.svd(enu_matrix, full_matrices=False)
    covariance = (vh.T / np.square(singular_values)) @ vh
    expected_pdop = np.sqrt(np.trace(covariance[:3, :3]))
    assert diagnostics.pdop == pytest.approx(expected_pdop)


def test_rank_tolerance_can_expose_or_discard_a_small_singular_direction():
    directions = np.array([
        [1.0, 0.0, 0.0], [1.0, 1e-3, 0.0],
        [1.0, 0.0, 1e-3], [1.0, -1e-3, -1e-3],
        [1.0, 2e-3, -2e-3],
    ])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    observations = observations_from_enu(directions)
    strict = diagnose_geometry(observations, RECEIVER, rank_tolerance=1e-5)
    permissive = diagnose_geometry(observations, RECEIVER, rank_tolerance=1e-14)
    assert not strict.full_rank
    assert permissive.full_rank
    assert np.isinf(strict.condition_number)
    assert np.isfinite(permissive.condition_number)


@pytest.mark.parametrize("state", [np.array([np.nan, 0, 0, 0]), np.zeros(3)])
def test_invalid_receiver_state_is_rejected(state):
    with pytest.raises(ValueError):
        diagnose_geometry(well_conditioned_observations(), state)


def test_nonfinite_observations_are_rejected():
    observations = well_conditioned_observations()
    bad_ranges = observations.pseudoranges.copy()
    bad_ranges[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        diagnose_geometry(GPSObservations(observations.satellite_positions_ecef, bad_ranges), RECEIVER)


def test_invalid_rank_tolerance_is_rejected():
    with pytest.raises(ValueError):
        diagnose_geometry(well_conditioned_observations(), RECEIVER, rank_tolerance=0.0)
