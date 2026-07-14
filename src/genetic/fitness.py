from __future__ import annotations

from dataclasses import dataclass

from src.genetic.decoder import decode_chromosome
from src.genetic.population import Chromosome
from src.models import (
    Delivery,
    Depot,
    Priority,
    Vehicle,
    VehicleRoute,
)


@dataclass(frozen=True)
class FitnessWeights:
    capacity_excess_per_kg: float = 2000.0
    autonomy_excess_per_km: float = 1000.0

    critical_position_penalty: float = 30.0
    high_position_penalty: float = 15.0
    medium_position_penalty: float = 5.0
    low_position_penalty: float = 0.0

    def priority_weight(
        self,
        priority: Priority,
    ) -> float:
        weights = {
            "critical": self.critical_position_penalty,
            "high": self.high_position_penalty,
            "medium": self.medium_position_penalty,
            "low": self.low_position_penalty,
        }

        return weights[priority]


@dataclass(frozen=True)
class FitnessResult:
    total_cost: float
    total_distance_km: float
    total_capacity_excess_kg: float
    total_autonomy_excess_km: float
    priority_penalty: float
    routes: tuple[VehicleRoute, ...]


def calculate_priority_penalty(
    routes: list[VehicleRoute],
    weights: FitnessWeights,
) -> float:
    total_penalty = 0.0

    for route in routes:
        for position, delivery in enumerate(
            route.deliveries,
        ):
            total_penalty += (
                position
                * weights.priority_weight(
                    delivery.priority
                )
            )

    return total_penalty


def evaluate_chromosome(
    chromosome: Chromosome,
    deliveries: list[Delivery],
    vehicles: list[Vehicle],
    depot: Depot,
    weights: FitnessWeights | None = None,
) -> FitnessResult:
    effective_weights = weights or FitnessWeights()

    routes = decode_chromosome(
        chromosome=chromosome,
        deliveries=deliveries,
        vehicles=vehicles,
        depot=depot,
    )

    total_distance_km = sum(
        route.distance_km
        for route in routes
    )

    total_capacity_excess_kg = sum(
        route.capacity_excess_kg
        for route in routes
    )

    total_autonomy_excess_km = sum(
        route.autonomy_excess_km
        for route in routes
    )

    priority_penalty = calculate_priority_penalty(
        routes=routes,
        weights=effective_weights,
    )

    total_cost = (
        total_distance_km
        + total_capacity_excess_kg
        * effective_weights.capacity_excess_per_kg
        + total_autonomy_excess_km
        * effective_weights.autonomy_excess_per_km
        + priority_penalty
    )

    return FitnessResult(
        total_cost=total_cost,
        total_distance_km=total_distance_km,
        total_capacity_excess_kg=total_capacity_excess_kg,
        total_autonomy_excess_km=total_autonomy_excess_km,
        priority_penalty=priority_penalty,
        routes=tuple(routes),
    )
