from __future__ import annotations

from dataclasses import dataclass

from src.distance import haversine_km
from src.genetic.fitness import (
    FitnessResult,
    FitnessWeights,
    evaluate_chromosome,
)
from src.genetic.population import Chromosome
from src.models import Delivery, Depot, Vehicle


@dataclass(frozen=True)
class BaselineResult:
    chromosome: tuple[int, ...]
    fitness: FitnessResult


def nearest_neighbor_chromosome(
    deliveries: list[Delivery],
    depot: Depot,
) -> Chromosome:
    if not deliveries:
        raise ValueError(
            "O baseline precisa conter pelo menos uma entrega."
        )

    remaining_deliveries = deliveries.copy()
    chromosome: Chromosome = []

    current_location = depot

    while remaining_deliveries:
        nearest_delivery = min(
            remaining_deliveries,
            key=lambda delivery: haversine_km(
                current_location,
                delivery,
            ),
        )

        chromosome.append(nearest_delivery.id)

        remaining_deliveries.remove(
            nearest_delivery
        )

        current_location = nearest_delivery

    return chromosome


def evaluate_nearest_neighbor(
    deliveries: list[Delivery],
    vehicles: list[Vehicle],
    depot: Depot,
    weights: FitnessWeights | None = None,
) -> BaselineResult:
    chromosome = nearest_neighbor_chromosome(
        deliveries=deliveries,
        depot=depot,
    )

    fitness = evaluate_chromosome(
        chromosome=chromosome,
        deliveries=deliveries,
        vehicles=vehicles,
        depot=depot,
        weights=weights,
    )

    return BaselineResult(
        chromosome=tuple(chromosome),
        fitness=fitness,
    )
