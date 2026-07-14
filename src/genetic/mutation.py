from __future__ import annotations

import random

from src.genetic.population import Chromosome


def swap_mutation(
    chromosome: Chromosome,
    mutation_rate: float = 0.10,
    random_generator: random.Random | None = None,
) -> Chromosome:
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            "A taxa de mutação precisa estar entre 0 e 1."
        )

    mutated_chromosome = chromosome.copy()

    if len(mutated_chromosome) < 2:
        return mutated_chromosome

    generator = random_generator or random.Random()

    should_mutate = (
        generator.random() < mutation_rate
    )

    if not should_mutate:
        return mutated_chromosome

    first_index, second_index = generator.sample(
        population=range(len(mutated_chromosome)),
        k=2,
    )

    (
        mutated_chromosome[first_index],
        mutated_chromosome[second_index],
    ) = (
        mutated_chromosome[second_index],
        mutated_chromosome[first_index],
    )

    return mutated_chromosome
