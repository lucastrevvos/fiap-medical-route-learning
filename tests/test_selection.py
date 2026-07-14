import random

import pytest

from src.genetic.selection import (
    select_parent_by_tournament,
)


def create_population() -> list[list[int]]:
    return [
        [1, 2, 3, 4],
        [2, 1, 4, 3],
        [3, 4, 1, 2],
        [4, 3, 2, 1],
    ]


def test_tournament_selects_lowest_cost_when_all_compete() -> None:
    population = create_population()

    costs = [
        400.0,
        150.0,
        300.0,
        220.0,
    ]

    selected_parent = select_parent_by_tournament(
        population=population,
        costs=costs,
        tournament_size=4,
        random_generator=random.Random(42),
    )

    assert selected_parent == population[1]


def test_selected_parent_is_a_copy() -> None:
    population = create_population()

    selected_parent = select_parent_by_tournament(
        population=population,
        costs=[100.0, 200.0, 300.0, 400.0],
        tournament_size=4,
        random_generator=random.Random(42),
    )

    selected_parent[0] = 99

    assert population[0] == [1, 2, 3, 4]


def test_same_seed_produces_same_selection() -> None:
    population = create_population()
    costs = [400.0, 150.0, 300.0, 220.0]

    first_parent = select_parent_by_tournament(
        population=population,
        costs=costs,
        tournament_size=2,
        random_generator=random.Random(42),
    )

    second_parent = select_parent_by_tournament(
        population=population,
        costs=costs,
        tournament_size=2,
        random_generator=random.Random(42),
    )

    assert first_parent == second_parent


def test_population_and_costs_must_have_same_size() -> None:
    with pytest.raises(
        ValueError,
        match="Cada indivíduo",
    ):
        select_parent_by_tournament(
            population=create_population(),
            costs=[100.0, 200.0],
            tournament_size=2,
        )


def test_tournament_cannot_be_larger_than_population() -> None:
    with pytest.raises(
        ValueError,
        match="maior que a população",
    ):
        select_parent_by_tournament(
            population=create_population(),
            costs=[100.0, 200.0, 300.0, 400.0],
            tournament_size=5,
        )
