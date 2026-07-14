import pytest

from src.baseline import (
    evaluate_nearest_neighbor,
    nearest_neighbor_chromosome,
)
from src.models import Delivery, Depot, Vehicle


def create_delivery(
    delivery_id: int,
    longitude: float,
) -> Delivery:
    return Delivery(
        id=delivery_id,
        name=f"Unidade {delivery_id}",
        latitude=0.0,
        longitude=longitude,
        demand_kg=5.0,
        priority="medium",
        item="Medicamento",
    )


def test_nearest_neighbor_selects_closest_deliveries() -> None:
    depot = Depot(
        name="Hospital",
        latitude=0.0,
        longitude=0.0,
    )

    deliveries = [
        create_delivery(
            delivery_id=1,
            longitude=0.01,
        ),
        create_delivery(
            delivery_id=2,
            longitude=0.03,
        ),
        create_delivery(
            delivery_id=3,
            longitude=0.02,
        ),
    ]

    chromosome = nearest_neighbor_chromosome(
        deliveries=deliveries,
        depot=depot,
    )

    assert chromosome == [1, 3, 2]


def test_nearest_neighbor_preserves_every_delivery() -> None:
    depot = Depot(
        name="Hospital",
        latitude=0.0,
        longitude=0.0,
    )

    deliveries = [
        create_delivery(
            delivery_id=1,
            longitude=0.01,
        ),
        create_delivery(
            delivery_id=2,
            longitude=0.03,
        ),
        create_delivery(
            delivery_id=3,
            longitude=0.02,
        ),
    ]

    chromosome = nearest_neighbor_chromosome(
        deliveries=deliveries,
        depot=depot,
    )

    assert sorted(chromosome) == [1, 2, 3]


def test_baseline_returns_fitness_result() -> None:
    depot = Depot(
        name="Hospital",
        latitude=0.0,
        longitude=0.0,
    )

    deliveries = [
        create_delivery(
            delivery_id=1,
            longitude=0.01,
        ),
        create_delivery(
            delivery_id=2,
            longitude=0.02,
        ),
    ]

    result = evaluate_nearest_neighbor(
        deliveries=deliveries,
        vehicles=[
            Vehicle(
                id=1,
                name="Van",
                capacity_kg=20.0,
                max_distance_km=1000.0,
            )
        ],
        depot=depot,
    )

    assert result.chromosome == (1, 2)
    assert result.fitness.total_distance_km > 0.0
    assert result.fitness.total_cost > 0.0


def test_empty_deliveries_raise_error() -> None:
    with pytest.raises(
        ValueError,
        match="pelo menos uma entrega",
    ):
        nearest_neighbor_chromosome(
            deliveries=[],
            depot=Depot(
                name="Hospital",
                latitude=0.0,
                longitude=0.0,
            ),
        )
