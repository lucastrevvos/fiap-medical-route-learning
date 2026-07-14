import random

import pytest

from src.genetic.mutation import swap_mutation


def test_mutation_preserves_every_gene() -> None:
    chromosome = [
        1, 2, 3, 4, 5,
        6, 7, 8,
    ]

    mutated = swap_mutation(
        chromosome=chromosome,
        mutation_rate=1.0,
        random_generator=random.Random(42),
    )

    assert sorted(mutated) == sorted(chromosome)
    assert len(mutated) == len(chromosome)


def test_mutation_does_not_modify_original_chromosome() -> None:
    chromosome = [1, 2, 3, 4, 5]
    original_copy = chromosome.copy()

    swap_mutation(
        chromosome=chromosome,
        mutation_rate=1.0,
        random_generator=random.Random(42),
    )

    assert chromosome == original_copy


def test_zero_mutation_rate_preserves_content() -> None:
    chromosome = [1, 2, 3, 4, 5]

    result = swap_mutation(
        chromosome=chromosome,
        mutation_rate=0.0,
        random_generator=random.Random(42),
    )

    assert result == chromosome
    assert result is not chromosome


def test_full_mutation_rate_changes_two_positions() -> None:
    chromosome = [1, 2, 3, 4, 5]

    mutated = swap_mutation(
        chromosome=chromosome,
        mutation_rate=1.0,
        random_generator=random.Random(42),
    )

    changed_positions = sum(
        original_gene != mutated_gene
        for original_gene, mutated_gene in zip(
            chromosome,
            mutated,
        )
    )

    assert changed_positions == 2


def test_same_seed_produces_same_mutation() -> None:
    chromosome = [1, 2, 3, 4, 5]

    first_result = swap_mutation(
        chromosome=chromosome,
        mutation_rate=1.0,
        random_generator=random.Random(42),
    )

    second_result = swap_mutation(
        chromosome=chromosome,
        mutation_rate=1.0,
        random_generator=random.Random(42),
    )

    assert first_result == second_result


@pytest.mark.parametrize(
    "invalid_rate",
    [
        -0.1,
        1.1,
        10.0,
    ],
)
def test_invalid_mutation_rate_raises_error(
    invalid_rate: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="entre 0 e 1",
    ):
        swap_mutation(
            chromosome=[1, 2, 3],
            mutation_rate=invalid_rate,
        )


def test_single_gene_chromosome_returns_copy() -> None:
    chromosome = [1]

    result = swap_mutation(
        chromosome=chromosome,
        mutation_rate=1.0,
        random_generator=random.Random(42),
    )

    assert result == [1]
    assert result is not chromosome
