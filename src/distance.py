from math import asin, cos, radians, sin, sqrt
from typing import Protocol


class HasCoordinates(Protocol):
    latitude: float
    longitude: float


def haversine_km(
    origin: HasCoordinates,
    destination: HasCoordinates,
) -> float:
    earth_radius_km = 6371.0088

    origin_latitude = radians(origin.latitude)
    origin_longitude = radians(origin.longitude)

    destination_latitude = radians(destination.latitude)
    destination_longitude = radians(destination.longitude)

    latitude_difference = destination_latitude - origin_latitude
    longitude_difference = destination_longitude - origin_longitude

    haversine_value = (
        sin(latitude_difference / 2) ** 2
        + cos(origin_latitude)
        * cos(destination_latitude)
        * sin(longitude_difference / 2) ** 2
    )

    central_angle = 2 * asin(sqrt(haversine_value))

    return earth_radius_km * central_angle


def route_distance_km(
    depot: HasCoordinates,
    stops: list[HasCoordinates],
) -> float:
    if not stops:
        return 0.0

    route_points = [depot, *stops, depot]

    total_distance = 0.0

    for current_point, next_point in zip(
        route_points,
        route_points[1:],
    ):
        total_distance += haversine_km(
            current_point,
            next_point,
        )

    return total_distance
