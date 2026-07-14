from __future__ import annotations

import random

from src.genetic.population import (
    Chromosome,
    Population,
)


def select_parent_by_tournament(
    population: Population,
    costs: list[float],
    tournament_size: int = 3,
    random_generator: random.Random | None = None,
) -> Chromosome:
    if not population:
        raise ValueError(
            "Não é possível realizar seleção em uma população vazia."
        )

    if len(population) != len(costs):
        raise ValueError(
            "Cada indivíduo da população precisa possuir um custo."
        )

    if tournament_size < 2:
        raise ValueError(
            "O torneio precisa conter pelo menos dois indivíduos."
        )

    if tournament_size > len(population):
        raise ValueError(
            "O torneio não pode ser maior que a população."
        )

    generator = random_generator or random.Random()

    candidate_indices = generator.sample(
        population=range(len(population)),
        k=tournament_size,
    )

    winner_index = min(
        candidate_indices,
        key=lambda index: costs[index],
    )

    return population[winner_index].copy()
