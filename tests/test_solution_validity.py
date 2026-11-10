import inspect

import numpy as np
import pytest

import gps_hybrid.solution_validity as validity
from gps_hybrid.coordinates import geodetic_to_ecef
from gps_hybrid.benchmarking import generate_benchmark_scenario
from gps_hybrid.models import GPSObservations
from gps_hybrid.scenario import ScenarioConfig, generate_scenario
from gps_hybrid.solvers import SolverConfig


def _policy(*, domain=None, rms=0.1, maximum=0.2, condition=1e6):
    if domain is None:
        domain = validity.OperatingDomainConfig("terrestrial", -1_000.0, 100_000.0)
    return validity.SolutionValidityConfig(
        operating_domain=domain,
        max_residual_rms_m=rms,
        max_absolute_residual_m=maximum,
        condition_number_threshold=condition,
    )


def _good_case(noise=0.0):
    scenario = generate_scenario(ScenarioConfig(seed=71, pseudorange_noise_std_m=noise))
    return scenario.observations, scenario.truth.receiver_state.copy()


def _observations_for_state(satellites, state, offsets=None):
    ranges = np.linalg.norm(satellites - state[:3], axis=1)
    pseudoranges = ranges + state[3]
    if offsets is not None:
        pseudoranges = pseudoranges + np.asarray(offsets, dtype=float)
    return GPSObservations(satellites, pseudoranges)


def _pilot_wrong_root_case():
    scenario = generate_benchmark_scenario(
        scenario_id="scenario-00007", split="development", seed=20261117,
        satellite_count=4, noise_std_m=0.3, position_perturbation_m=10.0,
        geometry_category="poor", outlier_rate=0.5,
    )
    estimate = np.array([
        15946681.007513316, -15978667.632112216,
        3818943.988147863, 21990898.60175015,
    ])
    return scenario.scenario.observations, scenario.scenario.truth.receiver_state.copy(), estimate


def test_good_geometry_plausible_estimate_passes_configured_policy():
    observations, state = _good_case()
    result = validity.assess_solution_validity(observations, state, "converged", _policy())
    assert result.numerical_status == "converged"
    assert result.assessment_status == "valid_under_configured_policy"
    assert result.classification == "valid_under_configured_policy"
    assert result.measurement_fit.status == "pass"
    assert result.geometry_status == "acceptable_conditioning"
    assert result.physical.status == "pass"
    assert result.geometry is not None and result.geometry.full_rank


def test_exact_fit_wrong_root_is_physically_implausible_and_not_valid():
    observations, test_only_truth, estimate = _pilot_wrong_root_case()
    result = validity.assess_solution_validity(observations, estimate, "converged", _policy())
    assert np.linalg.norm(estimate[:3] - test_only_truth[:3]) > 20_000_000.0
    assert result.measurement_fit.residual_rms_m == pytest.approx(0.0, abs=1e-8)
    assert result.measurement_fit.maximum_absolute_residual_m == pytest.approx(0.0, abs=1e-8)
    assert result.physical.status == "fail"
    assert result.assessment_status == "invalid_under_configured_policy"
    assert result.classification == "physically_implausible"
    assert "physically_implausible" in result.flags


def test_low_residual_does_not_establish_accuracy():
    observations, test_only_reference, wrong_estimate = _pilot_wrong_root_case()
    result = validity.assess_solution_validity(observations, wrong_estimate, "converged", _policy())
    test_only_error = np.linalg.norm(wrong_estimate[:3] - test_only_reference[:3])
    assert test_only_error > 1_000_000.0
    assert result.measurement_fit.status == "pass"
    assert result.assessment_status == "invalid_under_configured_policy"
    assert result.physical.status == "fail"


def test_full_rank_poor_conditioning_is_separate_failure():
    center = np.array([6_378_137.0, 0.0, 0.0])
    directions = np.array([
        [1.0, 0.0, 0.0], [1.0, 1e-4, 1e-4],
        [1.0, -1e-4, 2e-4], [1.0, 2e-4, -1e-4],
        [1.0, -2e-4, -1e-4], [1.0, 3e-4, 2e-4],
    ])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    satellites = center + 22_000_000.0 * directions
    state = np.array([*center, 75.0])
    observations = _observations_for_state(satellites, state)
    result = validity.assess_solution_validity(observations, state, "converged", _policy())
    assert result.geometry is not None and result.geometry.full_rank
    assert result.geometry.condition_number > 1e6
    assert result.geometry_status == "poor_observability"
    assert "poor_observability" in result.flags
    assert result.assessment_status == "invalid_under_configured_policy"
    assert result.classification == "poor_observability"


def test_rank_deficiency_is_not_poor_conditioning():
    center = np.array([6_378_137.0, 0.0, 0.0])
    satellites = center + np.arange(1.0, 7.0)[:, None] * np.array([22_000_000.0, 0.0, 0.0])
    state = np.array([*center, 75.0])
    observations = _observations_for_state(satellites, state)
    result = validity.assess_solution_validity(observations, state, "converged", _policy())
    assert result.geometry_status == "rank_deficient"
    assert result.geometry is not None and result.geometry.numerical_rank < 4
    assert result.geometry.pdop is None
    assert "geometry_unobservable" in result.flags
    assert "poor_observability" not in result.flags
    assert result.classification == "geometry_unobservable"


def test_fewer_than_four_observations_is_insufficient_and_incomplete():
    observations, state = _good_case()
    short = GPSObservations(
        observations.satellite_positions_ecef[:3], observations.pseudoranges[:3],
    )
    result = validity.assess_solution_validity(short, state, "converged", _policy())
    assert result.geometry_status == "insufficient_observations"
    assert result.geometry is not None and result.geometry.insufficient_observations
    assert result.measurement_fit.status == "not_assessable"
    assert result.assessment_status == "incomplete_assessment"
    assert "insufficient_observations" in result.flags


def test_non_converged_status_is_preserved_and_assessment_incomplete():
    observations, state = _good_case()
    result = validity.assess_solution_validity(observations, state, "max_iterations", _policy())
    assert result.numerical_status == "max_iterations"
    assert result.measurement_fit.status == "pass"
    assert result.geometry_status == "acceptable_conditioning"
    assert result.assessment_status == "incomplete_assessment"
    assert result.classification == "incomplete_assessment"
    assert "not_numerically_converged" in result.flags


def test_both_configured_fit_thresholds_must_pass():
    observations, state = _good_case()
    result = validity.assess_solution_validity(observations, state, "converged", _policy(rms=0.0, maximum=0.0))
    assert result.measurement_fit.rms_threshold_passed is True
    assert result.measurement_fit.maximum_absolute_threshold_passed is True
    assert result.measurement_fit.status == "pass"


def test_rms_threshold_failure_is_reported():
    observations, state = _good_case()
    observations = _observations_for_state(
        observations.satellite_positions_ecef, state, np.full(8, 2.0),
    )
    result = validity.assess_solution_validity(observations, state, "converged", _policy(rms=1.0, maximum=3.0))
    assert result.measurement_fit.rms_threshold_passed is False
    assert result.measurement_fit.maximum_absolute_threshold_passed is True
    assert result.measurement_fit.status == "fail"
    assert "measurement_fit_failed" in result.flags


def test_maximum_absolute_residual_threshold_failure_is_reported():
    observations, state = _good_case()
    observations = _observations_for_state(
        observations.satellite_positions_ecef, state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
    )
    result = validity.assess_solution_validity(observations, state, "converged", _policy(rms=1.0, maximum=1.0))
    assert result.measurement_fit.rms_threshold_passed is True
    assert result.measurement_fit.maximum_absolute_threshold_passed is False
    assert result.measurement_fit.status == "fail"


def test_no_fit_threshold_is_not_evaluated_and_overall_is_incomplete():
    observations, state = _good_case()
    result = validity.assess_solution_validity(observations, state, "converged", _policy(rms=None, maximum=None))
    assert result.measurement_fit.status == "not_evaluated"
    assert result.measurement_fit.residual_rms_m is not None
    assert result.assessment_status == "incomplete_assessment"


def test_unrestricted_domain_requires_explicit_waiver():
    observations, state = _good_case()
    no_waiver = validity.assess_solution_validity(
        observations, state, "converged",
        _policy(domain=validity.OperatingDomainConfig("unrestricted_ecef")),
    )
    waived = validity.assess_solution_validity(
        observations, state, "converged",
        _policy(domain=validity.OperatingDomainConfig(
            "unrestricted_ecef", allow_unrestricted_domain_waiver=True,
        )),
    )
    assert no_waiver.physical.status == "not_applicable"
    assert not no_waiver.physical.waived
    assert no_waiver.assessment_status == "incomplete_assessment"
    assert waived.physical.status == "not_applicable"
    assert waived.physical.waived
    assert "physical_domain_waived" in waived.flags
    assert "physical_domain_passed" not in waived.flags
    assert waived.assessment_status == "valid_under_configured_policy"


@pytest.mark.parametrize("estimate", [
    None,
    np.array([np.nan, 0.0, 0.0, 0.0]),
    np.array([1.0, 2.0, 3.0]),
])
def test_unrestricted_unavailable_or_invalid_estimate_is_not_waived(estimate):
    domain = validity.OperatingDomainConfig(
        "unrestricted_ecef", allow_unrestricted_domain_waiver=True,
    )
    physical = validity._assess_physical(estimate, domain)
    assert physical.status == "not_assessable"
    assert physical.waived is False
    assert physical.domain == "unrestricted_ecef"
    assert physical.minimum_altitude_m is None
    assert physical.maximum_altitude_m is None


@pytest.mark.parametrize("converted", [
    (np.nan, 0.0, 0.0),
    (0.0, np.inf, 0.0),
])
def test_unrestricted_failed_or_nonfinite_conversion_is_unavailable(monkeypatch, converted):
    domain = validity.OperatingDomainConfig(
        "unrestricted_ecef", allow_unrestricted_domain_waiver=True,
    )
    monkeypatch.setattr(validity, "ecef_to_geodetic", lambda _position: converted)
    physical = validity._assess_physical(np.array([1.0, 2.0, 3.0, 4.0]), domain)
    assert physical.status == "not_assessable"
    assert physical.waived is False
    assert physical.domain == "unrestricted_ecef"


def test_unrestricted_conversion_exception_is_not_waived(monkeypatch):
    domain = validity.OperatingDomainConfig(
        "unrestricted_ecef", allow_unrestricted_domain_waiver=True,
    )
    monkeypatch.setattr(
        validity, "ecef_to_geodetic",
        lambda _position: (_ for _ in ()).throw(RuntimeError("conversion error")),
    )
    physical = validity._assess_physical(np.array([1.0, 2.0, 3.0, 4.0]), domain)
    assert physical.status == "not_assessable"
    assert physical.waived is False


@pytest.mark.parametrize("allow_waiver, expected_waived", [(False, False), (True, True)])
def test_unrestricted_successful_conversion_is_not_applicable_with_explicit_waiver(
    allow_waiver, expected_waived,
):
    domain = validity.OperatingDomainConfig(
        "unrestricted_ecef", allow_unrestricted_domain_waiver=allow_waiver,
    )
    physical = validity._assess_physical(np.array([6_378_137.0, 0.0, 0.0, 0.0]), domain)
    assert physical.status == "not_applicable"
    assert physical.waived is expected_waived
    assert physical.altitude_m is not None


def test_invalid_or_nonfinite_estimate_is_incomplete_with_missing_diagnostics():
    observations, _state = _good_case()
    result = validity.assess_solution_validity(
        observations, np.array([np.nan, 0.0, 0.0, 0.0]), "converged", _policy(),
    )
    assert result.assessment_status == "incomplete_assessment"
    assert result.measurement_fit.residual_rms_m is None
    assert result.geometry is None
    assert result.physical.altitude_m is None


def test_ecef_conversion_failure_is_not_treated_as_pass(monkeypatch):
    observations, state = _good_case()
    monkeypatch.setattr(validity, "ecef_to_geodetic", lambda _position: (_ for _ in ()).throw(ValueError("bad conversion")))
    result = validity.assess_solution_validity(observations, state, "converged", _policy())
    assert result.physical.status == "not_assessable"
    assert result.assessment_status == "incomplete_assessment"


def test_simultaneous_failures_are_preserved_and_conditioning_has_precedence():
    center = np.array([6_378_137.0, 0.0, 0.0])
    directions = np.array([
        [1.0, 0.0, 0.0], [1.0, 1e-4, 1e-4],
        [1.0, -1e-4, 2e-4], [1.0, 2e-4, -1e-4],
        [1.0, -2e-4, -1e-4], [1.0, 3e-4, 2e-4],
    ])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    satellites = center + 22_000_000.0 * directions
    state = np.array([*geodetic_to_ecef(0.7, -1.1, 250_000.0), 75.0])
    observations = _observations_for_state(satellites, state, np.full(6, 100.0))
    result = validity.assess_solution_validity(observations, state, "converged", _policy(rms=1.0, maximum=1.0))
    assert "poor_observability" in result.flags
    assert "measurement_fit_failed" in result.flags
    assert "physically_implausible" in result.flags
    assert result.classification == "poor_observability"
    assert result.assessment_status == "invalid_under_configured_policy"


@pytest.mark.parametrize("kwargs", [
    {"name": "invalid"},
    {"name": "terrestrial", "minimum_altitude_m": None, "maximum_altitude_m": 10.0},
    {"name": "terrestrial", "minimum_altitude_m": 10.0, "maximum_altitude_m": -10.0},
    {"name": "terrestrial", "minimum_altitude_m": True, "maximum_altitude_m": 10.0},
])
def test_invalid_operating_domain_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        validity.OperatingDomainConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"max_residual_rms_m": -1.0},
    {"max_absolute_residual_m": np.inf},
    {"max_residual_rms_m": True},
    {"condition_number_threshold": 0.5},
    {"condition_number_threshold": np.nan},
    {"rank_tolerance": 1.0},
])
def test_invalid_validity_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        validity.SolutionValidityConfig(
            operating_domain=validity.OperatingDomainConfig("terrestrial", -1_000.0, 100_000.0),
            **kwargs,
        )


def test_validator_signature_and_imports_exclude_scenario_truth():
    signature = inspect.signature(validity.assess_solution_validity)
    assert tuple(signature.parameters) == (
        "observations", "estimated_state", "numerical_status", "config",
    )
    assert "Scenario" not in vars(validity)
    assert "ScenarioTruth" not in vars(validity)
    assert "truth" not in signature.parameters


def test_benchmark_profile_is_explicit_temporary_terrestrial_policy():
    config = validity.synthetic_benchmark_validity_config()
    assert config.operating_domain.name == "terrestrial"
    assert config.operating_domain.minimum_altitude_m == -1_000.0
    assert config.operating_domain.maximum_altitude_m == 100_000.0
    assert config.max_residual_rms_m is None
    assert config.max_absolute_residual_m is None
