from __future__ import annotations

import random
from typing import TypeAlias

from src.models import Delivery


Chromosome: TypeAlias = list[int]
Population: TypeAlias = list[Chromosome]


def get_delivery_ids(
    deliveries: list[Delivery],
) -> list[int]:
    delivery_ids = [
        delivery.id
        for delivery in deliveries
    ]

    if len(delivery_ids) != len(set(delivery_ids)):
        raise ValueError(
            "Não é possível criar cromossomos com IDs duplicados."
        )

    return delivery_ids


def validate_chromosome(
    chromosome: Chromosome,
    deliveries: list[Delivery],
) -> None:
    expected_delivery_ids = sorted(
        get_delivery_ids(deliveries)
    )

    chromosome_delivery_ids = sorted(chromosome)

    if chromosome_delivery_ids != expected_delivery_ids:
        raise ValueError(
            "O cromossomo deve conter cada entrega exatamente uma vez."
        )


def create_random_chromosome(
    deliveries: list[Delivery],
    random_generator: random.Random | None = None,
) -> Chromosome:
    if not deliveries:
        raise ValueError(
            "Não é possível criar um cromossomo sem entregas."
        )

    generator = random_generator or random.Random()

    chromosome = get_delivery_ids(deliveries)

    generator.shuffle(chromosome)

    return chromosome


def create_initial_population(
    deliveries: list[Delivery],
    population_size: int,
    random_seed: int | None = None,
) -> Population:
    if population_size < 2:
        raise ValueError(
            "A população precisa conter pelo menos dois indivíduos."
        )

    random_generator = random.Random(random_seed)

    return [
        create_random_chromosome(
            deliveries=deliveries,
            random_generator=random_generator,
        )
        for _ in range(population_size)
    ]
