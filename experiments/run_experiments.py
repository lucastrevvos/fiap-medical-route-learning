from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import statistics
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.baseline import evaluate_nearest_neighbor
from src.genetic.optimizer import (
    GeneticConfig,
    OptimizationResult,
    optimize_routes,
)
from src.io_utils import load_scenario


def run_genetic_experiment(
    base_config: GeneticConfig,
    repetitions: int = 3,
) -> tuple[
    list[OptimizationResult],
    list[float],
]:
    results: list[OptimizationResult] = []
    execution_times: list[float] = []

    for repetition in range(repetitions):
        config = replace(
            base_config,
            random_seed=(
                base_config.random_seed
                + repetition
            ),
        )

        started_at = time.perf_counter()

        result = optimize_routes(
            deliveries=DELIVERIES,
            vehicles=VEHICLES,
            depot=DEPOT,
            config=config,
        )

        elapsed_seconds = (
            time.perf_counter()
            - started_at
        )

        results.append(result)
        execution_times.append(elapsed_seconds)

    return results, execution_times


def create_result_row(
    experiment_name: str,
    config: GeneticConfig,
    results: list[OptimizationResult],
    execution_times: list[float],
    baseline_cost: float,
    baseline_distance_km: float,
) -> dict[str, object]:
    best_result = min(
        results,
        key=lambda result: (
            result.best_fitness.total_cost
        ),
    )

    costs = [
        result.best_fitness.total_cost
        for result in results
    ]

    distances = [
        result.best_fitness.total_distance_km
        for result in results
    ]

    return {
        "experiment": experiment_name,
        "population_size": config.population_size,
        "generations": config.generations,
        "mutation_rate": config.mutation_rate,
        "crossover_rate": config.crossover_rate,
        "tournament_size": config.tournament_size,
        "elite_size": config.elite_size,
        "mean_cost": round(
            statistics.mean(costs),
            3,
        ),
        "best_cost": round(
            min(costs),
            3,
        ),
        "mean_distance_km": round(
            statistics.mean(distances),
            3,
        ),
        "best_distance_km": round(
            min(distances),
            3,
        ),
        "mean_time_seconds": round(
            statistics.mean(execution_times),
            4,
        ),
        "best_priority_penalty": round(
            best_result.best_fitness.priority_penalty,
            3,
        ),
        "best_capacity_excess_kg": round(
            best_result
            .best_fitness
            .total_capacity_excess_kg,
            3,
        ),
        "best_autonomy_excess_km": round(
            best_result
            .best_fitness
            .total_autonomy_excess_km,
            3,
        ),
        "baseline_cost": round(
            baseline_cost,
            3,
        ),
        "baseline_distance_km": round(
            baseline_distance_km,
            3,
        ),
    }


def save_csv(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_markdown(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    headers = list(rows[0].keys())

    lines = [
        "# Resultados dos experimentos",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            ["---"] * len(headers)
        ) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(row[header])
                for header in headers
            )
            + " |"
        )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    DELIVERIES, VEHICLES, DEPOT = load_scenario(
        deliveries_path=(
            PROJECT_ROOT
            / "data"
            / "deliveries.json"
        ),
        vehicles_path=(
            PROJECT_ROOT
            / "data"
            / "vehicles.json"
        ),
        depot_path=(
            PROJECT_ROOT
            / "data"
            / "depot.json"
        ),
    )

    baseline = evaluate_nearest_neighbor(
        deliveries=DELIVERIES,
        vehicles=VEHICLES,
        depot=DEPOT,
    )

    experiments = [
        (
            "experiment_1",
            GeneticConfig(
                population_size=50,
                generations=100,
                crossover_rate=0.80,
                mutation_rate=0.05,
                tournament_size=3,
                elite_size=2,
                random_seed=42,
            ),
        ),
        (
            "experiment_2",
            GeneticConfig(
                population_size=100,
                generations=200,
                crossover_rate=0.85,
                mutation_rate=0.10,
                tournament_size=3,
                elite_size=2,
                random_seed=42,
            ),
        ),
        (
            "experiment_3",
            GeneticConfig(
                population_size=150,
                generations=300,
                crossover_rate=0.90,
                mutation_rate=0.15,
                tournament_size=4,
                elite_size=4,
                random_seed=42,
            ),
        ),
    ]

    print("BASELINE — VIZINHO MAIS PRÓXIMO")
    print(
        f"Custo: "
        f"{baseline.fitness.total_cost:.3f}"
    )
    print(
        f"Distância: "
        f"{baseline.fitness.total_distance_km:.3f} km"
    )
    print(
        f"Prioridade: "
        f"{baseline.fitness.priority_penalty:.3f}"
    )
    print()

    rows: list[dict[str, object]] = []

    for experiment_name, config in experiments:
        print(
            f"Executando {experiment_name}..."
        )

        results, execution_times = (
            run_genetic_experiment(
                base_config=config,
                repetitions=3,
            )
        )

        row = create_result_row(
            experiment_name=experiment_name,
            config=config,
            results=results,
            execution_times=execution_times,
            baseline_cost=(
                baseline.fitness.total_cost
            ),
            baseline_distance_km=(
                baseline
                .fitness
                .total_distance_km
            ),
        )

        rows.append(row)

        print(
            f"  custo médio: "
            f"{row['mean_cost']}"
        )
        print(
            f"  melhor custo: "
            f"{row['best_cost']}"
        )
        print(
            f"  distância média: "
            f"{row['mean_distance_km']} km"
        )
        print(
            f"  tempo médio: "
            f"{row['mean_time_seconds']} s"
        )
        print()

    csv_path = (
        PROJECT_ROOT
        / "experiments"
        / "results.csv"
    )

    markdown_path = (
        PROJECT_ROOT
        / "experiments"
        / "results.md"
    )

    save_csv(
        rows=rows,
        output_path=csv_path,
    )

    save_markdown(
        rows=rows,
        output_path=markdown_path,
    )

    print("Arquivos gerados:")
    print(csv_path)
    print(markdown_path)
