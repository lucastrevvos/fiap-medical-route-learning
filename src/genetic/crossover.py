from __future__ import annotations

import random

from src.genetic.population import Chromosome


def validate_parents(
    parent_a: Chromosome,
    parent_b: Chromosome,
) -> None:
    if len(parent_a) != len(parent_b):
        raise ValueError(
            "Os pais precisam possuir o mesmo tamanho."
        )

    if len(parent_a) < 2:
        raise ValueError(
            "O crossover precisa de cromossomos "
            "com pelo menos dois genes."
        )

    if sorted(parent_a) != sorted(parent_b):
        raise ValueError(
            "Os pais precisam conter os mesmos genes."
        )

    if len(parent_a) != len(set(parent_a)):
        raise ValueError(
            "O primeiro pai contém genes duplicados."
        )

    if len(parent_b) != len(set(parent_b)):
        raise ValueError(
            "O segundo pai contém genes duplicados."
        )


def build_ordered_child(
    primary_parent: Chromosome,
    secondary_parent: Chromosome,
    start_index: int,
    end_index: int,
) -> Chromosome:
    chromosome_size = len(primary_parent)

    child: list[int | None] = [
        None
        for _ in range(chromosome_size)
    ]

    child[start_index:end_index + 1] = (
        primary_parent[start_index:end_index + 1]
    )

    insertion_index = (
        end_index + 1
    ) % chromosome_size

    reading_index = (
        end_index + 1
    ) % chromosome_size

    for _ in range(chromosome_size):
        gene = secondary_parent[reading_index]

        if gene not in child:
            while child[insertion_index] is not None:
                insertion_index = (
                    insertion_index + 1
                ) % chromosome_size

            child[insertion_index] = gene

            insertion_index = (
                insertion_index + 1
            ) % chromosome_size

        reading_index = (
            reading_index + 1
        ) % chromosome_size

    return [
        gene
        for gene in child
        if gene is not None
    ]


def ordered_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    random_generator: random.Random | None = None,
) -> tuple[Chromosome, Chromosome]:
    validate_parents(
        parent_a=parent_a,
        parent_b=parent_b,
    )

    generator = random_generator or random.Random()

    start_index, end_index = sorted(
        generator.sample(
            population=range(len(parent_a)),
            k=2,
        )
    )

    child_a = build_ordered_child(
        primary_parent=parent_a,
        secondary_parent=parent_b,
        start_index=start_index,
        end_index=end_index,
    )

    child_b = build_ordered_child(
        primary_parent=parent_b,
        secondary_parent=parent_a,
        start_index=start_index,
        end_index=end_index,
    )

    return child_a, child_b
