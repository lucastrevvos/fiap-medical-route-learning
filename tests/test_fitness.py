import pytest

from src.genetic.fitness import evaluate_chromosome
from src.models import Delivery, Depot, Vehicle


def create_depot() -> Depot:
    return Depot(
        name="Hospital",
        latitude=0.0,
        longitude=0.0,
    )


def create_delivery(
    delivery_id: int,
    longitude: float,
    demand_kg: float,
    priority: str,
) -> Delivery:
    return Delivery(
        id=delivery_id,
        name=f"Unidade {delivery_id}",
        latitude=0.0,
        longitude=longitude,
        demand_kg=demand_kg,
        priority=priority,
        item="Medicamento",
    )


def test_critical_delivery_later_in_route_increases_cost() -> None:
    deliveries = [
        create_delivery(
            delivery_id=1,
            longitude=0.01,
            demand_kg=5.0,
            priority="critical",
        ),
        create_delivery(
            delivery_id=2,
            longitude=0.02,
            demand_kg=5.0,
            priority="low",
        ),
    ]

    vehicles = [
        Vehicle(
            id=1,
            name="Van",
            capacity_kg=20.0,
            max_distance_km=1000.0,
        )
    ]

    critical_first = evaluate_chromosome(
        chromosome=[1, 2],
        deliveries=deliveries,
        vehicles=vehicles,
        depot=create_depot(),
    )

    critical_last = evaluate_chromosome(
        chromosome=[2, 1],
        deliveries=deliveries,
        vehicles=vehicles,
        depot=create_depot(),
    )

    assert (
        critical_first.total_distance_km
        == pytest.approx(
            critical_last.total_distance_km
        )
    )

    assert (
        critical_first.priority_penalty
        < critical_last.priority_penalty
    )

    assert (
        critical_first.total_cost
        < critical_last.total_cost
    )


def test_capacity_excess_receives_high_penalty() -> None:
    deliveries = [
        create_delivery(
            delivery_id=1,
            longitude=0.01,
            demand_kg=8.0,
            priority="low",
        ),
        create_delivery(
            delivery_id=2,
            longitude=0.02,
            demand_kg=8.0,
            priority="low",
        ),
    ]

    feasible_result = evaluate_chromosome(
        chromosome=[1, 2],
        deliveries=deliveries,
        vehicles=[
            Vehicle(
                id=1,
                name="Van",
                capacity_kg=20.0,
                max_distance_km=1000.0,
            )
        ],
        depot=create_depot(),
    )

    overloaded_result = evaluate_chromosome(
        chromosome=[1, 2],
        deliveries=deliveries,
        vehicles=[
            Vehicle(
                id=1,
                name="Van",
                capacity_kg=10.0,
                max_distance_km=1000.0,
            )
        ],
        depot=create_depot(),
    )

    assert (
        overloaded_result.total_capacity_excess_kg
        == pytest.approx(6.0)
    )

    assert (
        overloaded_result.total_cost
        - feasible_result.total_cost
        == pytest.approx(12000.0)
    )


def test_autonomy_excess_increases_total_cost() -> None:
    deliveries = [
        create_delivery(
            delivery_id=1,
            longitude=0.02,
            demand_kg=5.0,
            priority="low",
        )
    ]

    normal_result = evaluate_chromosome(
        chromosome=[1],
        deliveries=deliveries,
        vehicles=[
            Vehicle(
                id=1,
                name="Van",
                capacity_kg=20.0,
                max_distance_km=1000.0,
            )
        ],
        depot=create_depot(),
    )

    insufficient_autonomy_result = evaluate_chromosome(
        chromosome=[1],
        deliveries=deliveries,
        vehicles=[
            Vehicle(
                id=1,
                name="Van",
                capacity_kg=20.0,
                max_distance_km=1.0,
            )
        ],
        depot=create_depot(),
    )

    assert (
        normal_result.total_autonomy_excess_km
        == pytest.approx(0.0)
    )

    assert (
        insufficient_autonomy_result
        .total_autonomy_excess_km
        > 0.0
    )

    assert (
        insufficient_autonomy_result.total_cost
        > normal_result.total_cost
    )
