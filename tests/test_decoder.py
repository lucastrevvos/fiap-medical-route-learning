import pytest

from src.genetic.decoder import decode_chromosome
from src.models import Delivery, Depot, Vehicle


def create_depot() -> Depot:
    return Depot(
        name="Hospital",
        latitude=-27.595,
        longitude=-48.548,
    )


def create_deliveries() -> list[Delivery]:
    demands = {
        1: 12.0,
        2: 8.0,
        3: 10.0,
        4: 5.0,
    }

    return [
        Delivery(
            id=delivery_id,
            name=f"Unidade {delivery_id}",
            latitude=-27.5 - delivery_id / 100,
            longitude=-48.5 - delivery_id / 100,
            demand_kg=demand,
            priority="medium",
            item="Medicamento",
        )
        for delivery_id, demand in demands.items()
    ]


def test_decoder_preserves_every_delivery() -> None:
    deliveries = create_deliveries()

    vehicles = [
        Vehicle(
            id=1,
            name="Van 1",
            capacity_kg=20.0,
            max_distance_km=200.0,
        ),
        Vehicle(
            id=2,
            name="Van 2",
            capacity_kg=20.0,
            max_distance_km=200.0,
        ),
    ]

    routes = decode_chromosome(
        chromosome=[1, 2, 3, 4],
        deliveries=deliveries,
        vehicles=vehicles,
        depot=create_depot(),
    )

    decoded_ids = [
        delivery.id
        for route in routes
        for delivery in route.deliveries
    ]

    assert sorted(decoded_ids) == [
        1,
        2,
        3,
        4,
    ]


def test_decoder_moves_to_next_vehicle_when_capacity_is_reached() -> None:
    routes = decode_chromosome(
        chromosome=[1, 2, 3, 4],
        deliveries=create_deliveries(),
        vehicles=[
            Vehicle(
                id=1,
                name="Van 1",
                capacity_kg=20.0,
                max_distance_km=200.0,
            ),
            Vehicle(
                id=2,
                name="Van 2",
                capacity_kg=20.0,
                max_distance_km=200.0,
            ),
        ],
        depot=create_depot(),
    )

    first_route_ids = [
        delivery.id
        for delivery in routes[0].deliveries
    ]

    second_route_ids = [
        delivery.id
        for delivery in routes[1].deliveries
    ]

    assert first_route_ids == [1, 2]
    assert second_route_ids == [3, 4]

    assert routes[0].load_kg == pytest.approx(20.0)
    assert routes[1].load_kg == pytest.approx(15.0)


def test_decoder_records_capacity_excess_on_last_vehicle() -> None:
    routes = decode_chromosome(
        chromosome=[1, 2, 3, 4],
        deliveries=create_deliveries(),
        vehicles=[
            Vehicle(
                id=1,
                name="Van 1",
                capacity_kg=15.0,
                max_distance_km=200.0,
            ),
            Vehicle(
                id=2,
                name="Van 2",
                capacity_kg=15.0,
                max_distance_km=200.0,
            ),
        ],
        depot=create_depot(),
    )

    assert routes[0].load_kg == pytest.approx(12.0)
    assert routes[0].capacity_excess_kg == pytest.approx(0.0)

    assert routes[1].load_kg == pytest.approx(23.0)
    assert routes[1].capacity_excess_kg == pytest.approx(8.0)


def test_decoder_without_vehicles_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="sem veículos",
    ):
        decode_chromosome(
            chromosome=[1, 2, 3, 4],
            deliveries=create_deliveries(),
            vehicles=[],
            depot=create_depot(),
        )
