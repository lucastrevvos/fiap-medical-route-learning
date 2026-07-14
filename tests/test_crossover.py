import random

import pytest

from src.genetic.crossover import ordered_crossover


def create_parents() -> tuple[list[int], list[int]]:
    parent_a = [
        1, 2, 3, 4, 5,
        6, 7, 8,
    ]

    parent_b = [
        5, 7, 2, 1, 8,
        4, 6, 3,
    ]

    return parent_a, parent_b


def test_ordered_crossover_preserves_every_gene() -> None:
    parent_a, parent_b = create_parents()

    child_a, child_b = ordered_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        random_generator=random.Random(42),
    )

    assert sorted(child_a) == sorted(parent_a)
    assert sorted(child_b) == sorted(parent_b)

    assert len(child_a) == len(parent_a)
    assert len(child_b) == len(parent_b)


def test_ordered_crossover_does_not_modify_parents() -> None:
    parent_a, parent_b = create_parents()

    original_parent_a = parent_a.copy()
    original_parent_b = parent_b.copy()

    ordered_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        random_generator=random.Random(42),
    )

    assert parent_a == original_parent_a
    assert parent_b == original_parent_b


def test_same_seed_produces_same_children() -> None:
    parent_a, parent_b = create_parents()

    first_children = ordered_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        random_generator=random.Random(42),
    )

    second_children = ordered_crossover(
        parent_a=parent_a,
        parent_b=parent_b,
        random_generator=random.Random(42),
    )

    assert first_children == second_children


def test_parents_must_have_same_size() -> None:
    with pytest.raises(
        ValueError,
        match="mesmo tamanho",
    ):
        ordered_crossover(
            parent_a=[1, 2, 3, 4],
            parent_b=[1, 2, 3],
        )


def test_parents_must_have_same_genes() -> None:
    with pytest.raises(
        ValueError,
        match="mesmos genes",
    ):
        ordered_crossover(
            parent_a=[1, 2, 3, 4],
            parent_b=[1, 2, 3, 99],
        )


def test_parents_cannot_have_duplicate_genes() -> None:
    with pytest.raises(
        ValueError,
        match="duplicados",
    ):
        ordered_crossover(
            parent_a=[1, 2, 2, 4],
            parent_b=[1, 2, 2, 4],
        )
