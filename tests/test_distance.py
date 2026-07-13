import pytest

from src.distance import haversine_km, route_distance_km
from src.models import Delivery, Depot


def create_depot() -> Depot:
    return Depot(
        name="Centro de Distribuição Hospitalar",
        latitude=-27.595377,
        longitude=-48.548050,
    )


def test_distance_between_same_point_is_zero() -> None:
    depot = create_depot()

    distance = haversine_km(
        depot,
        depot,
    )

    assert distance == pytest.approx(0.0)


def test_distance_is_symmetric() -> None:
    depot = create_depot()

    delivery = Delivery(
        id=1,
        name="Unidade Centro",
        latitude=-27.596900,
        longitude=-48.549500,
        demand_kg=12.0,
        priority="critical",
        item="Insulina",
    )

    outbound_distance = haversine_km(
        depot,
        delivery,
    )

    return_distance = haversine_km(
        delivery,
        depot,
    )

    assert outbound_distance == pytest.approx(return_distance)


def test_empty_route_has_zero_distance() -> None:
    depot = create_depot()

    distance = route_distance_km(
        depot,
        [],
    )

    assert distance == pytest.approx(0.0)


def test_route_distance_includes_return_to_depot() -> None:
    depot = create_depot()

    center_delivery = Delivery(
        id=1,
        name="Unidade Centro",
        latitude=-27.596900,
        longitude=-48.549500,
        demand_kg=12.0,
        priority="critical",
        item="Insulina",
    )

    trindade_delivery = Delivery(
        id=2,
        name="Unidade Trindade",
        latitude=-27.586000,
        longitude=-48.519800,
        demand_kg=18.0,
        priority="high",
        item="Antibióticos",
    )

    distance = route_distance_km(
        depot,
        [
            center_delivery,
            trindade_delivery,
        ],
    )

    assert distance == pytest.approx(
        6.362,
        abs=0.001,
    )
