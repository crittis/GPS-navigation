"""Independent GPS solution-fit, geometry, and operating-domain assessment.

This module deliberately accepts observations and an estimate only. It never
imports scenario or truth data and does not change solver status or behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .coordinates import ecef_to_geodetic
from .geometry import DEFAULT_RANK_TOLERANCE, GeometryDiagnostics, diagnose_geometry
from .gps_model import residual
from .models import GPSObservations


OperatingDomainName = Literal["terrestrial", "aviation", "unrestricted_ecef"]
AssessmentStatus = Literal[
    "valid_under_configured_policy",
    "invalid_under_configured_policy",
    "incomplete_assessment",
]


def _finite_real(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return numeric


@dataclass(frozen=True)
class OperatingDomainConfig:
    """Explicit physical operating domain; bounds are WGS-84 altitude metres.

    ``unrestricted_ecef`` has no altitude bounds. Its physical dimension is
    ``not_applicable`` and can be waived only with the explicit flag.
    """

    name: OperatingDomainName
    minimum_altitude_m: float | None = None
    maximum_altitude_m: float | None = None
    allow_unrestricted_domain_waiver: bool = False

    def __post_init__(self) -> None:
        if self.name not in ("terrestrial", "aviation", "unrestricted_ecef"):
            raise ValueError("name must be terrestrial, aviation, or unrestricted_ecef")
        if not isinstance(self.allow_unrestricted_domain_waiver, (bool, np.bool_)):
            raise ValueError("allow_unrestricted_domain_waiver must be boolean")
        if self.name == "unrestricted_ecef":
            if self.minimum_altitude_m is not None or self.maximum_altitude_m is not None:
                raise ValueError("unrestricted_ecef cannot have altitude bounds")
        else:
            if self.minimum_altitude_m is None or self.maximum_altitude_m is None:
                raise ValueError("bounded domains require minimum and maximum altitude")
            minimum = _finite_real("minimum_altitude_m", self.minimum_altitude_m)
            maximum = _finite_real("maximum_altitude_m", self.maximum_altitude_m)
            if minimum > maximum:
                raise ValueError("minimum_altitude_m cannot exceed maximum_altitude_m")
            object.__setattr__(self, "minimum_altitude_m", minimum)
            object.__setattr__(self, "maximum_altitude_m", maximum)
            if self.allow_unrestricted_domain_waiver:
                raise ValueError("domain waiver is only valid for unrestricted_ecef")


@dataclass(frozen=True)
class SolutionValidityConfig:
    """Required validity policies; no measurement-fit limit is invented.

    The default condition threshold of ``1e6`` is an engineering policy choice
    aligned with the existing benchmark's ``poor`` category, not a universal
    mathematical constant. Applications should set their own policy.
    """

    operating_domain: OperatingDomainConfig
    max_residual_rms_m: float | None = None
    max_absolute_residual_m: float | None = None
    condition_number_threshold: float = 1e6
    rank_tolerance: float = DEFAULT_RANK_TOLERANCE

    def __post_init__(self) -> None:
        if not isinstance(self.operating_domain, OperatingDomainConfig):
            raise ValueError("operating_domain must be an OperatingDomainConfig")
        for field_name in ("max_residual_rms_m", "max_absolute_residual_m"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _finite_real(field_name, value, minimum=0.0))
        threshold = _finite_real("condition_number_threshold", self.condition_number_threshold, minimum=1.0)
        object.__setattr__(self, "condition_number_threshold", threshold)
        rank_tolerance = _finite_real("rank_tolerance", self.rank_tolerance, minimum=np.finfo(float).tiny)
        if rank_tolerance >= 1.0:
            raise ValueError("rank_tolerance must be less than one")
        object.__setattr__(self, "rank_tolerance", rank_tolerance)


@dataclass(frozen=True)
class MeasurementFitDiagnostics:
    """Residual statistics and per-limit outcomes, all in metres."""

    residual_rms_m: float | None
    maximum_absolute_residual_m: float | None
    rms_threshold_m: float | None
    maximum_absolute_threshold_m: float | None
    rms_threshold_passed: bool | None
    maximum_absolute_threshold_passed: bool | None
    status: Literal["pass", "fail", "not_evaluated", "not_assessable"]
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = {"pass", "fail", "not_evaluated", "not_assessable"}
        if self.status not in allowed:
            raise ValueError("Invalid measurement-fit status")
        for name in ("residual_rms_m", "maximum_absolute_residual_m", "rms_threshold_m",
                     "maximum_absolute_threshold_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_real(name, value, minimum=0.0))
        for name in ("rms_threshold_passed", "maximum_absolute_threshold_passed"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean or None")
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))


@dataclass(frozen=True)
class PhysicalPlausibilityDiagnostics:
    """Geodetic position and explicit operating-domain outcome."""

    domain: OperatingDomainName
    latitude_rad: float | None
    longitude_rad: float | None
    altitude_m: float | None
    minimum_altitude_m: float | None
    maximum_altitude_m: float | None
    status: Literal["pass", "fail", "not_applicable", "not_assessable"]
    waived: bool
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.domain not in ("terrestrial", "aviation", "unrestricted_ecef"):
            raise ValueError("Invalid physical-domain name")
        if self.status not in ("pass", "fail", "not_applicable", "not_assessable"):
            raise ValueError("Invalid physical-domain status")
        if not isinstance(self.waived, (bool, np.bool_)):
            raise ValueError("waived must be boolean")
        if self.waived and (self.domain != "unrestricted_ecef" or self.status != "not_applicable"):
            raise ValueError("Only unrestricted_ecef may be waived, and it remains not_applicable")
        for name in ("latitude_rad", "longitude_rad", "altitude_m", "minimum_altitude_m", "maximum_altitude_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_real(name, value))
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))


@dataclass(frozen=True)
class SolutionValidityResult:
    """Immutable multidimensional assessment; solver status stays unchanged."""

    numerical_status: str
    measurement_fit: MeasurementFitDiagnostics
    geometry: GeometryDiagnostics | None
    geometry_status: Literal[
        "insufficient_observations", "rank_deficient", "poor_observability",
        "acceptable_conditioning", "not_assessable",
    ]
    physical: PhysicalPlausibilityDiagnostics
    assessment_status: AssessmentStatus
    classification: str
    flags: frozenset[str]
    explanations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.numerical_status, str) or not self.numerical_status:
            raise ValueError("numerical_status must be a non-empty string")
        if self.assessment_status not in (
            "valid_under_configured_policy", "invalid_under_configured_policy", "incomplete_assessment",
        ):
            raise ValueError("Invalid solution assessment status")
        allowed_classifications = {
            "insufficient_observations", "not_numerically_converged", "geometry_unobservable",
            "poor_observability", "measurement_fit_unacceptable", "physically_implausible",
            "valid_under_configured_policy", "invalid_under_configured_policy", "incomplete_assessment",
        }
        if self.classification not in allowed_classifications:
            raise ValueError("Invalid primary solution classification")
        if self.assessment_status == "incomplete_assessment" and self.classification != "incomplete_assessment":
            raise ValueError("Incomplete assessment must use incomplete_assessment classification")
        if self.assessment_status == "valid_under_configured_policy" and self.classification != "valid_under_configured_policy":
            raise ValueError("Valid assessment must use valid_under_configured_policy classification")
        if self.geometry_status not in (
            "insufficient_observations", "rank_deficient", "poor_observability",
            "acceptable_conditioning", "not_assessable",
        ):
            raise ValueError("Invalid geometry assessment status")
        if not isinstance(self.measurement_fit, MeasurementFitDiagnostics):
            raise ValueError("measurement_fit must be MeasurementFitDiagnostics")
        if self.geometry is not None and not isinstance(self.geometry, GeometryDiagnostics):
            raise ValueError("geometry must be GeometryDiagnostics or None")
        if not isinstance(self.physical, PhysicalPlausibilityDiagnostics):
            raise ValueError("physical must be PhysicalPlausibilityDiagnostics")
        object.__setattr__(self, "flags", frozenset(self.flags))
        object.__setattr__(self, "explanations", tuple(self.explanations))


def _state_array(estimated_state: object) -> np.ndarray | None:
    if estimated_state is None:
        return None
    try:
        state = np.asarray(estimated_state, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        return None
    return state


def _assess_fit(observations: GPSObservations, state: np.ndarray | None,
                config: SolutionValidityConfig) -> MeasurementFitDiagnostics:
    rms_limit = config.max_residual_rms_m
    max_limit = config.max_absolute_residual_m
    if state is None:
        return MeasurementFitDiagnostics(
            None, None, rms_limit, max_limit, None, None, "not_assessable",
            ("A finite estimated state is unavailable",),
        )
    try:
        values = residual(state, observations)
        rms = float(np.sqrt(np.mean(values * values)))
        maximum = float(np.max(np.abs(values))) if values.size else None
        if not np.isfinite(rms) or maximum is None or not np.isfinite(maximum):
            raise FloatingPointError("Residual statistics are non-finite")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, TypeError) as exc:
        return MeasurementFitDiagnostics(
            None, None, rms_limit, max_limit, None, None, "not_assessable",
            (f"Residual evaluation failed: {exc}",),
        )
    if rms_limit is None and max_limit is None:
        return MeasurementFitDiagnostics(
            rms, maximum, None, None, None, None, "not_evaluated",
            ("No measurement-fit threshold is configured",),
        )
    rms_pass = None if rms_limit is None else rms <= rms_limit
    maximum_pass = None if max_limit is None else maximum <= max_limit
    failures = []
    if rms_pass is False:
        failures.append(f"Residual RMS {rms:g} m exceeds configured limit {rms_limit:g} m")
    if maximum_pass is False:
        failures.append(
            f"Maximum absolute residual {maximum:g} m exceeds configured limit {max_limit:g} m"
        )
    return MeasurementFitDiagnostics(
        rms, maximum, rms_limit, max_limit, rms_pass, maximum_pass,
        "fail" if failures else "pass", tuple(failures),
    )


def _assess_geometry(observations: GPSObservations, state: np.ndarray | None,
                     config: SolutionValidityConfig) -> tuple[GeometryDiagnostics | None, str, tuple[str, ...]]:
    if state is None:
        return None, "not_assessable", ("A finite estimated state is unavailable",)
    try:
        geometry = diagnose_geometry(observations, state, config.rank_tolerance)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, TypeError) as exc:
        return None, "not_assessable", (f"Geometry evaluation failed: {exc}",)
    if geometry.insufficient_observations:
        return geometry, "insufficient_observations", (geometry.explanation,)
    if not geometry.full_rank:
        return geometry, "rank_deficient", (geometry.explanation,)
    if geometry.condition_number > config.condition_number_threshold:
        return geometry, "poor_observability", (
            f"Condition number {geometry.condition_number:g} exceeds configured threshold "
            f"{config.condition_number_threshold:g}",
        )
    return geometry, "acceptable_conditioning", ()


def _assess_physical(state: np.ndarray | None,
                     domain: OperatingDomainConfig) -> PhysicalPlausibilityDiagnostics:
    # Validate the estimate here as well as at the public entry point: this
    # keeps a waived unrestricted domain from masking unavailable input.
    finite_state = _state_array(state)
    if finite_state is None:
        return PhysicalPlausibilityDiagnostics(
            domain.name, None, None, None, domain.minimum_altitude_m,
            domain.maximum_altitude_m,
            "not_assessable", False,
            ("A finite estimated state is unavailable",),
        )
    try:
        latitude, longitude, altitude = ecef_to_geodetic(finite_state[:3])
    except Exception as exc:
        return PhysicalPlausibilityDiagnostics(
            domain.name, None, None, None, domain.minimum_altitude_m,
            domain.maximum_altitude_m,
            "not_assessable", False,
            (f"ECEF-to-geodetic conversion failed: {exc}",),
        )
    if not np.all(np.isfinite((latitude, longitude, altitude))):
        return PhysicalPlausibilityDiagnostics(
            domain.name, None, None, None, domain.minimum_altitude_m,
            domain.maximum_altitude_m,
            "not_assessable", False,
            ("ECEF-to-geodetic conversion returned non-finite coordinates",),
        )
    if domain.name == "unrestricted_ecef":
        return PhysicalPlausibilityDiagnostics(
            domain.name, latitude, longitude, altitude, None, None, "not_applicable",
            domain.allow_unrestricted_domain_waiver,
            ("Unrestricted ECEF domain is not a physical-plausibility pass",),
        )
    assert domain.minimum_altitude_m is not None and domain.maximum_altitude_m is not None
    accepted = domain.minimum_altitude_m <= altitude <= domain.maximum_altitude_m
    reasons = () if accepted else (
        f"Ellipsoidal altitude {altitude:g} m is outside configured bounds "
        f"[{domain.minimum_altitude_m:g}, {domain.maximum_altitude_m:g}] m",
    )
    return PhysicalPlausibilityDiagnostics(
        domain.name, latitude, longitude, altitude, domain.minimum_altitude_m,
        domain.maximum_altitude_m, "pass" if accepted else "fail", False, reasons,
    )


def assess_solution_validity(
    observations: GPSObservations,
    estimated_state: ArrayLike | None,
    numerical_status: str,
    config: SolutionValidityConfig,
) -> SolutionValidityResult:
    """Assess fit, final geometry and physical domain without truth data.

    The supplied ``numerical_status`` is retained verbatim. Diagnostics are
    computed where possible even for non-converged results, but a result is
    valid only when the numerical status is ``converged`` and all required
    configured checks are complete. Truth and scenario objects are not part
    of this API.
    """
    if not isinstance(observations, GPSObservations):
        raise ValueError("observations must be a GPSObservations instance")
    if not isinstance(numerical_status, str) or not numerical_status:
        raise ValueError("numerical_status must be a non-empty string")
    if not isinstance(config, SolutionValidityConfig):
        raise ValueError("config must be a SolutionValidityConfig")

    state = _state_array(estimated_state)
    fit = _assess_fit(observations, state, config)
    geometry, geometry_status, geometry_reasons = _assess_geometry(observations, state, config)
    physical = _assess_physical(state, config.operating_domain)

    flags: set[str] = set()
    explanations: list[str] = []
    if numerical_status != "converged":
        flags.add("not_numerically_converged")
        explanations.append(f"Numerical status is {numerical_status!r}, not 'converged'")
    if geometry_status == "insufficient_observations":
        flags.add("insufficient_observations")
    elif geometry_status == "rank_deficient":
        flags.add("geometry_unobservable")
        flags.add("rank_deficient")
    elif geometry_status == "poor_observability":
        flags.add("poor_observability")
    elif geometry_status == "acceptable_conditioning":
        flags.add("acceptable_conditioning")
    if fit.status == "fail":
        flags.add("measurement_fit_failed")
    elif fit.status == "pass":
        flags.add("measurement_fit_passed")
    elif fit.status == "not_evaluated":
        flags.add("measurement_fit_not_evaluated")
    if physical.status == "fail":
        flags.add("physically_implausible")
    elif physical.status == "pass":
        flags.add("physical_domain_passed")
    elif physical.status == "not_applicable":
        flags.add("physical_domain_not_applicable")
        if physical.waived:
            flags.add("physical_domain_waived")
    for reasons in (fit.failure_reasons, geometry_reasons, physical.failure_reasons):
        explanations.extend(reasons)

    incomplete_reasons = []
    if numerical_status != "converged":
        incomplete_reasons.append("Numerical result did not converge")
    if state is None:
        incomplete_reasons.append("Estimated state is invalid or unavailable")
    if fit.status in ("not_assessable", "not_evaluated"):
        incomplete_reasons.append("Measurement-fit check is not complete")
    if geometry_status in ("not_assessable", "insufficient_observations"):
        incomplete_reasons.append("Required geometry assessment is not complete")
    if physical.status == "not_assessable":
        incomplete_reasons.append("Required physical-domain assessment is not complete")
    if physical.status == "not_applicable" and not physical.waived:
        incomplete_reasons.append("Unrestricted physical domain was not explicitly waived")

    if incomplete_reasons:
        assessment_status: AssessmentStatus = "incomplete_assessment"
        classification = "incomplete_assessment"
        explanations.extend(incomplete_reasons)
    else:
        failed = fit.status == "fail" or geometry_status in ("rank_deficient", "poor_observability") or physical.status == "fail"
        assessment_status = "invalid_under_configured_policy" if failed else "valid_under_configured_policy"
        if geometry_status == "rank_deficient":
            classification = "geometry_unobservable"
        elif geometry_status == "poor_observability":
            classification = "poor_observability"
        elif fit.status == "fail":
            classification = "measurement_fit_unacceptable"
        elif physical.status == "fail":
            classification = "physically_implausible"
        else:
            classification = assessment_status

    return SolutionValidityResult(
        numerical_status=numerical_status,
        measurement_fit=fit,
        geometry=geometry,
        geometry_status=geometry_status,  # type: ignore[arg-type]
        physical=physical,
        assessment_status=assessment_status,
        classification=classification,
        flags=frozenset(flags),
        explanations=tuple(dict.fromkeys(explanations)),
    )


def synthetic_benchmark_validity_config() -> SolutionValidityConfig:
    """Return the explicitly temporary domain profile for this synthetic study.

    The altitude interval is an experimental screening choice, not a general
    terrestrial receiver limit. Measurement fit remains unevaluated until the
    benchmark caller supplies application-specific thresholds.
    """
    return SolutionValidityConfig(
        operating_domain=OperatingDomainConfig("terrestrial", -1_000.0, 100_000.0),
    )


__all__ = [
    "AssessmentStatus",
    "MeasurementFitDiagnostics",
    "OperatingDomainConfig",
    "OperatingDomainName",
    "PhysicalPlausibilityDiagnostics",
    "SolutionValidityConfig",
    "SolutionValidityResult",
    "assess_solution_validity",
    "synthetic_benchmark_validity_config",
]
