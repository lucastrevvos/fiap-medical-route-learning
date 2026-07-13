import pytest

from src.genetic.population import (
    create_initial_population,
    create_random_chromosome,
    validate_chromosome,
)
from src.models import Delivery


def create_deliveries() -> list[Delivery]:
    return [
        Delivery(
            id=delivery_id,
            name=f"Unidade {delivery_id}",
            latitude=-27.5 - delivery_id / 100,
            longitude=-48.5 - delivery_id / 100,
            demand_kg=10.0,
            priority="medium",
            item="Medicamento",
        )
        for delivery_id in range(1, 6)
    ]


def test_random_chromosome_contains_every_delivery_once() -> None:
    deliveries = create_deliveries()

    chromosome = create_random_chromosome(
        deliveries=deliveries,
    )

    assert sorted(chromosome) == [1, 2, 3, 4, 5]
    assert len(chromosome) == 5


def test_initial_population_has_requested_size() -> None:
    deliveries = create_deliveries()

    population = create_initial_population(
        deliveries=deliveries,
        population_size=10,
        random_seed=42,
    )

    assert len(population) == 10

    for chromosome in population:
        validate_chromosome(
            chromosome=chromosome,
            deliveries=deliveries,
        )


def test_same_seed_produces_same_population() -> None:
    deliveries = create_deliveries()

    first_population = create_initial_population(
        deliveries=deliveries,
        population_size=5,
        random_seed=42,
    )

    second_population = create_initial_population(
        deliveries=deliveries,
        population_size=5,
        random_seed=42,
    )

    assert first_population == second_population


def test_invalid_chromosome_raises_error() -> None:
    deliveries = create_deliveries()

    invalid_chromosome = [
        1,
        2,
        2,
        4,
        5,
    ]

    with pytest.raises(
        ValueError,
        match="cada entrega exatamente uma vez",
    ):
        validate_chromosome(
            chromosome=invalid_chromosome,
            deliveries=deliveries,
        )


def test_population_with_less_than_two_individuals_raises_error() -> None:
    deliveries = create_deliveries()

    with pytest.raises(
        ValueError,
        match="pelo menos dois indivíduos",
    ):
        create_initial_population(
            deliveries=deliveries,
            population_size=1,
            random_seed=42,
        )
