from __future__ import annotations

from dataclasses import dataclass
import random

from src.genetic.crossover import ordered_crossover
from src.genetic.fitness import (
    FitnessResult,
    FitnessWeights,
    evaluate_chromosome,
)
from src.genetic.mutation import swap_mutation
from src.genetic.population import (
    Chromosome,
    Population,
    create_initial_population,
)
from src.genetic.selection import (
    select_parent_by_tournament,
)
from src.models import Delivery, Depot, Vehicle


@dataclass(frozen=True)
class GeneticConfig:
    population_size: int = 100
    generations: int = 200
    crossover_rate: float = 0.85
    mutation_rate: float = 0.10
    tournament_size: int = 3
    elite_size: int = 2
    random_seed: int = 42


@dataclass(frozen=True)
class GenerationSummary:
    generation: int
    best_cost: float
    average_cost: float
    worst_cost: float


@dataclass(frozen=True)
class OptimizationResult:
    best_chromosome: tuple[int, ...]
    best_fitness: FitnessResult
    best_generation: int
    history: tuple[GenerationSummary, ...]


def validate_config(
    config: GeneticConfig,
) -> None:
    if config.population_size < 2:
        raise ValueError(
            "A população precisa conter pelo menos dois indivíduos."
        )

    if config.generations < 1:
        raise ValueError(
            "A quantidade de gerações precisa ser positiva."
        )

    if not 0.0 <= config.crossover_rate <= 1.0:
        raise ValueError(
            "A taxa de crossover precisa estar entre 0 e 1."
        )

    if not 0.0 <= config.mutation_rate <= 1.0:
        raise ValueError(
            "A taxa de mutação precisa estar entre 0 e 1."
        )

    if not 2 <= config.tournament_size <= config.population_size:
        raise ValueError(
            "O tamanho do torneio precisa estar entre 2 "
            "e o tamanho da população."
        )

    if not 1 <= config.elite_size < config.population_size:
        raise ValueError(
            "O elitismo precisa preservar pelo menos um indivíduo, "
            "mas não pode preservar a população inteira."
        )


def evaluate_population(
    population: Population,
    deliveries: list[Delivery],
    vehicles: list[Vehicle],
    depot: Depot,
    weights: FitnessWeights | None = None,
) -> list[FitnessResult]:
    return [
        evaluate_chromosome(
            chromosome=chromosome,
            deliveries=deliveries,
            vehicles=vehicles,
            depot=depot,
            weights=weights,
        )
        for chromosome in population
    ]


def optimize_routes(
    deliveries: list[Delivery],
    vehicles: list[Vehicle],
    depot: Depot,
    config: GeneticConfig | None = None,
    weights: FitnessWeights | None = None,
) -> OptimizationResult:
    effective_config = config or GeneticConfig()

    validate_config(effective_config)

    population = create_initial_population(
        deliveries=deliveries,
        population_size=effective_config.population_size,
        random_seed=effective_config.random_seed,
    )

    random_generator = random.Random(
        effective_config.random_seed + 1
    )

    best_chromosome: Chromosome | None = None
    best_fitness: FitnessResult | None = None
    best_generation = 0

    history: list[GenerationSummary] = []

    for generation in range(
        1,
        effective_config.generations + 1,
    ):
        fitness_results = evaluate_population(
            population=population,
            deliveries=deliveries,
            vehicles=vehicles,
            depot=depot,
            weights=weights,
        )

        costs = [
            result.total_cost
            for result in fitness_results
        ]

        ranked_indices = sorted(
            range(len(population)),
            key=lambda index: costs[index],
        )

        generation_best_index = ranked_indices[0]
        generation_best_fitness = fitness_results[
            generation_best_index
        ]

        if (
            best_fitness is None
            or generation_best_fitness.total_cost
            < best_fitness.total_cost
        ):
            best_chromosome = population[
                generation_best_index
            ].copy()

            best_fitness = generation_best_fitness
            best_generation = generation

        history.append(
            GenerationSummary(
                generation=generation,
                best_cost=min(costs),
                average_cost=sum(costs) / len(costs),
                worst_cost=max(costs),
            )
        )

        if generation == effective_config.generations:
            break

        new_population: Population = [
            population[index].copy()
            for index in ranked_indices[
                :effective_config.elite_size
            ]
        ]

        while (
            len(new_population)
            < effective_config.population_size
        ):
            parent_a = select_parent_by_tournament(
                population=population,
                costs=costs,
                tournament_size=(
                    effective_config.tournament_size
                ),
                random_generator=random_generator,
            )

            parent_b = select_parent_by_tournament(
                population=population,
                costs=costs,
                tournament_size=(
                    effective_config.tournament_size
                ),
                random_generator=random_generator,
            )

            should_crossover = (
                random_generator.random()
                < effective_config.crossover_rate
            )

            if should_crossover:
                child_a, child_b = ordered_crossover(
                    parent_a=parent_a,
                    parent_b=parent_b,
                    random_generator=random_generator,
                )
            else:
                child_a = parent_a.copy()
                child_b = parent_b.copy()

            child_a = swap_mutation(
                chromosome=child_a,
                mutation_rate=effective_config.mutation_rate,
                random_generator=random_generator,
            )

            child_b = swap_mutation(
                chromosome=child_b,
                mutation_rate=effective_config.mutation_rate,
                random_generator=random_generator,
            )

            new_population.append(child_a)

            if (
                len(new_population)
                < effective_config.population_size
            ):
                new_population.append(child_b)

        population = new_population

    if best_chromosome is None or best_fitness is None:
        raise RuntimeError(
            "O algoritmo terminou sem produzir uma solução."
        )

    return OptimizationResult(
        best_chromosome=tuple(best_chromosome),
        best_fitness=best_fitness,
        best_generation=best_generation,
        history=tuple(history),
    )
