import pytest

from src.io_utils import validate_scenario
from src.models import Delivery, Depot, Vehicle


def create_depot() -> Depot:
    return Depot(
        name="Hospital",
        latitude=-27.595377,
        longitude=-48.54805,
    )


def create_vehicle(
    vehicle_id: int = 1,
    capacity_kg: float = 80.0,
) -> Vehicle:
    return Vehicle(
        id=vehicle_id,
        name=f"Van {vehicle_id}",
        capacity_kg=capacity_kg,
        max_distance_km=130.0,
    )


def create_delivery(
    delivery_id: int = 1,
    demand_kg: float = 10.0,
    priority: str = "high",
) -> Delivery:
    return Delivery(
        id=delivery_id,
        name=f"Unidade {delivery_id}",
        latitude=-27.6,
        longitude=-48.5,
        demand_kg=demand_kg,
        priority=priority,
        item="Medicamento",
    )


def test_valid_scenario_does_not_raise_error() -> None:
    validate_scenario(
        deliveries=[create_delivery()],
        vehicles=[create_vehicle()],
        depot=create_depot(),
    )


def test_duplicate_delivery_ids_raise_error() -> None:
    deliveries = [
        create_delivery(delivery_id=1),
        create_delivery(delivery_id=1),
    ]

    with pytest.raises(
        ValueError,
        match="IDs de entrega duplicados",
    ):
        validate_scenario(
            deliveries=deliveries,
            vehicles=[create_vehicle()],
            depot=create_depot(),
        )


def test_invalid_priority_raises_error() -> None:
    delivery = create_delivery(
        priority="banana",
    )

    with pytest.raises(
        ValueError,
        match="Prioridade inválida",
    ):
        validate_scenario(
            deliveries=[delivery],
            vehicles=[create_vehicle()],
            depot=create_depot(),
        )


def test_total_demand_above_capacity_raises_error() -> None:
    deliveries = [
        create_delivery(
            delivery_id=1,
            demand_kg=60.0,
        ),
        create_delivery(
            delivery_id=2,
            demand_kg=40.0,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="carga total",
    ):
        validate_scenario(
            deliveries=deliveries,
            vehicles=[
                create_vehicle(
                    capacity_kg=80.0,
                )
            ],
            depot=create_depot(),
        )
