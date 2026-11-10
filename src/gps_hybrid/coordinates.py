"""WGS-84 geodetic, ECEF and local ENU coordinate transformations.

Angles are radians. ECEF/ENU coordinates are metres. ENU is right-handed:
east, north, up. Azimuth is clockwise from north; elevation is above the
local horizon. The local frame is anchored at the receiver's geodetic location.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def geodetic_to_ecef(latitude_rad: float, longitude_rad: float, altitude_m: float) -> FloatArray:
    """Convert WGS-84 geodetic latitude/longitude/altitude to ECEF metres."""
    if not np.all(np.isfinite([latitude_rad, longitude_rad, altitude_m])):
        raise ValueError("Geodetic coordinates must be finite")
    if not -np.pi / 2 <= latitude_rad <= np.pi / 2:
        raise ValueError("Latitude must be in [-pi/2, pi/2]")
    sin_lat = np.sin(latitude_rad)
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    return np.array(
        [
            (prime_vertical + altitude_m) * np.cos(latitude_rad) * np.cos(longitude_rad),
            (prime_vertical + altitude_m) * np.cos(latitude_rad) * np.sin(longitude_rad),
            (prime_vertical * (1.0 - WGS84_E2) + altitude_m) * sin_lat,
        ],
        dtype=float,
    )


def ecef_to_geodetic(position_ecef: ArrayLike) -> tuple[float, float, float]:
    """Convert one ECEF position in metres to WGS-84 lat, lon, altitude."""
    xyz = np.asarray(position_ecef, dtype=float)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("ECEF position must be a finite vector of shape (3,)")
    x, y, z = xyz
    p = np.hypot(x, y)
    if np.linalg.norm(xyz) < 1.0:
        raise ValueError("ECEF position is too close to the Earth's centre")
    # Longitude is undefined on the rotation axis; use zero by convention.
    # At this scale the horizontal component is below one micrometre and the
    # polar ellipsoid formula avoids an ill-conditioned latitude iteration.
    if p <= 1e-6:
        latitude = np.copysign(np.pi / 2.0, z)
        longitude = 0.0
        altitude = abs(z) - WGS84_A * np.sqrt(1.0 - WGS84_E2)
        return float(latitude), longitude, float(altitude)
    longitude = np.arctan2(y, x)
    latitude = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        sin_lat = np.sin(latitude)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        altitude = p / max(abs(np.cos(latitude)), 1e-15) - n
        updated = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + altitude)))
        if abs(updated - latitude) < 1e-13:
            latitude = updated
            break
        latitude = updated
    sin_lat = np.sin(latitude)
    # Projection onto the ellipsoid normal avoids dividing by cos(latitude),
    # which amplifies roundoff near the poles.
    altitude = (
        p * np.cos(latitude)
        + z * sin_lat
        - WGS84_A * np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    )
    return float(latitude), float(longitude), float(altitude)


def enu_to_ecef_rotation(latitude_rad: float, longitude_rad: float) -> FloatArray:
    """Return R such that an ENU column vector maps to ECEF via R @ v."""
    slat, clat = np.sin(latitude_rad), np.cos(latitude_rad)
    slon, clon = np.sin(longitude_rad), np.cos(longitude_rad)
    return np.array(
        [
            [-slon, -slat * clon, clat * clon],
            [clon, -slat * slon, clat * slon],
            [0.0, clat, slat],
        ],
        dtype=float,
    )


def enu_direction_from_azimuth_elevation(azimuth_rad: float, elevation_rad: float) -> FloatArray:
    """Unit ENU direction; azimuth clockwise from north, elevation above horizon."""
    if not np.all(np.isfinite([azimuth_rad, elevation_rad])):
        raise ValueError("Azimuth and elevation must be finite")
    if not 0.0 <= elevation_rad <= np.pi / 2:
        raise ValueError("Elevation must be in [0, pi/2]")
    return np.array(
        [np.cos(elevation_rad) * np.sin(azimuth_rad),
         np.cos(elevation_rad) * np.cos(azimuth_rad),
         np.sin(elevation_rad)],
        dtype=float,
    )
