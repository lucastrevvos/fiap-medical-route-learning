import pytest

from src.genetic.optimizer import (
    GeneticConfig,
    optimize_routes,
)
from src.models import Delivery, Depot, Vehicle


def create_scenario() -> tuple[
    list[Delivery],
    list[Vehicle],
    Depot,
]:
    deliveries = [
        Delivery(
            id=delivery_id,
            name=f"Unidade {delivery_id}",
            latitude=-27.50 - delivery_id / 100,
            longitude=-48.50 - delivery_id / 100,
            demand_kg=8.0,
            priority=priority,
            item="Medicamento",
        )
        for delivery_id, priority in [
            (1, "critical"),
            (2, "high"),
            (3, "medium"),
            (4, "low"),
            (5, "high"),
        ]
    ]

    vehicles = [
        Vehicle(
            id=1,
            name="Van 1",
            capacity_kg=24.0,
            max_distance_km=300.0,
        ),
        Vehicle(
            id=2,
            name="Van 2",
            capacity_kg=24.0,
            max_distance_km=300.0,
        ),
    ]

    depot = Depot(
        name="Hospital",
        latitude=-27.50,
        longitude=-48.50,
    )

    return deliveries, vehicles, depot


def create_config() -> GeneticConfig:
    return GeneticConfig(
        population_size=20,
        generations=30,
        crossover_rate=0.85,
        mutation_rate=0.20,
        tournament_size=3,
        elite_size=2,
        random_seed=42,
    )


def test_optimizer_returns_valid_solution() -> None:
    deliveries, vehicles, depot = create_scenario()

    result = optimize_routes(
        deliveries=deliveries,
        vehicles=vehicles,
        depot=depot,
        config=create_config(),
    )

    assert sorted(result.best_chromosome) == [
        1,
        2,
        3,
        4,
        5,
    ]

    assert len(result.history) == 30
    assert result.best_fitness.total_cost > 0.0

    decoded_ids = [
        delivery.id
        for route in result.best_fitness.routes
        for delivery in route.deliveries
    ]

    assert sorted(decoded_ids) == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_elitism_prevents_best_cost_from_increasing() -> None:
    deliveries, vehicles, depot = create_scenario()

    result = optimize_routes(
        deliveries=deliveries,
        vehicles=vehicles,
        depot=depot,
        config=create_config(),
    )

    best_costs = [
        summary.best_cost
        for summary in result.history
    ]

    for previous_cost, current_cost in zip(
        best_costs,
        best_costs[1:],
    ):
        assert current_cost <= previous_cost + 0.000001


def test_same_seed_produces_same_result() -> None:
    deliveries, vehicles, depot = create_scenario()

    first_result = optimize_routes(
        deliveries=deliveries,
        vehicles=vehicles,
        depot=depot,
        config=create_config(),
    )

    second_result = optimize_routes(
        deliveries=deliveries,
        vehicles=vehicles,
        depot=depot,
        config=create_config(),
    )

    assert (
        first_result.best_chromosome
        == second_result.best_chromosome
    )

    assert (
        first_result.best_fitness.total_cost
        == pytest.approx(
            second_result.best_fitness.total_cost
        )
    )


@pytest.mark.parametrize(
    "config",
    [
        GeneticConfig(population_size=1),
        GeneticConfig(generations=0),
        GeneticConfig(crossover_rate=1.5),
        GeneticConfig(mutation_rate=-0.1),
        GeneticConfig(
            population_size=10,
            tournament_size=11,
        ),
        GeneticConfig(
            population_size=10,
            elite_size=10,
        ),
    ],
)
def test_invalid_config_raises_error(
    config: GeneticConfig,
) -> None:
    deliveries, vehicles, depot = create_scenario()

    with pytest.raises(ValueError):
        optimize_routes(
            deliveries=deliveries,
            vehicles=vehicles,
            depot=depot,
            config=config,
        )
