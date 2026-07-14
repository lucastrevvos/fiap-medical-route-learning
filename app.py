from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.baseline import evaluate_nearest_neighbor
from src.genetic.optimizer import (
    GeneticConfig,
    OptimizationResult,
    optimize_routes,
)
from src.io_utils import load_scenario
from src.map_service import create_routes_map


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIRECTORY = PROJECT_ROOT / "data"


st.set_page_config(
    page_title="Medical Route Optimizer",
    page_icon="🏥",
    layout="wide",
)


@st.cache_data
def load_application_scenario():
    return load_scenario(
        deliveries_path=(
            DATA_DIRECTORY
            / "deliveries.json"
        ),
        vehicles_path=(
            DATA_DIRECTORY
            / "vehicles.json"
        ),
        depot_path=(
            DATA_DIRECTORY
            / "depot.json"
        ),
    )


def create_routes_dataframe(
    result: OptimizationResult,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for route in result.best_fitness.routes:
        for stop_position, delivery in enumerate(
            route.deliveries,
            start=1,
        ):
            rows.append(
                {
                    "Veículo": route.vehicle.name,
                    "Ordem": stop_position,
                    "Destino": delivery.name,
                    "Item": delivery.item,
                    "Prioridade": delivery.priority,
                    "Carga (kg)": delivery.demand_kg,
                }
            )

    return pd.DataFrame(rows)


def create_history_dataframe(
    result: OptimizationResult,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Geração": summary.generation,
                "Melhor custo": summary.best_cost,
                "Custo médio": summary.average_cost,
                "Pior custo": summary.worst_cost,
            }
            for summary in result.history
        ]
    ).set_index("Geração")


deliveries, vehicles, depot = (
    load_application_scenario()
)


st.title("Medical Route Optimizer")

st.write(
    "Otimização de rotas para distribuição "
    "de medicamentos e insumos hospitalares "
    "utilizando algoritmo genético."
)


with st.sidebar:
    st.header("Configuração")

    population_size = st.slider(
        label="Tamanho da população",
        min_value=20,
        max_value=300,
        value=100,
        step=10,
    )

    generations = st.slider(
        label="Quantidade de gerações",
        min_value=20,
        max_value=500,
        value=200,
        step=20,
    )

    crossover_rate = st.slider(
        label="Taxa de crossover",
        min_value=0.50,
        max_value=1.00,
        value=0.85,
        step=0.01,
    )

    mutation_rate = st.slider(
        label="Taxa de mutação",
        min_value=0.00,
        max_value=0.50,
        value=0.10,
        step=0.01,
    )

    tournament_size = st.slider(
        label="Tamanho do torneio",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
    )

    elite_size = st.slider(
        label="Quantidade da elite",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

    random_seed = st.number_input(
        label="Semente aleatória",
        min_value=0,
        value=42,
        step=1,
    )

    run_optimization = st.button(
        label="Executar otimização",
        type="primary",
        use_container_width=True,
    )


if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = None

if "baseline_result" not in st.session_state:
    st.session_state.baseline_result = None


if run_optimization:
    config = GeneticConfig(
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        elite_size=elite_size,
        random_seed=int(random_seed),
    )

    with st.spinner(
        "Evoluindo as rotas..."
    ):
        st.session_state.optimization_result = (
            optimize_routes(
                deliveries=deliveries,
                vehicles=vehicles,
                depot=depot,
                config=config,
            )
        )

        st.session_state.baseline_result = (
            evaluate_nearest_neighbor(
                deliveries=deliveries,
                vehicles=vehicles,
                depot=depot,
            )
        )


result = st.session_state.optimization_result
baseline = st.session_state.baseline_result


if result is None or baseline is None:
    st.info(
        "Configure os parâmetros na barra lateral "
        "e clique em Executar otimização."
    )

    st.subheader("Cenário de entregas")

    scenario_dataframe = pd.DataFrame(
        [
            {
                "ID": delivery.id,
                "Destino": delivery.name,
                "Item": delivery.item,
                "Carga (kg)": delivery.demand_kg,
                "Prioridade": delivery.priority,
            }
            for delivery in deliveries
        ]
    )

    st.dataframe(
        scenario_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.stop()


st.subheader("Resultados principais")

metric_1, metric_2, metric_3, metric_4 = (
    st.columns(4)
)

metric_1.metric(
    label="Custo genético",
    value=(
        f"{result.best_fitness.total_cost:.2f}"
    ),
)

metric_2.metric(
    label="Custo do baseline",
    value=(
        f"{baseline.fitness.total_cost:.2f}"
    ),
    delta=(
        result.best_fitness.total_cost
        - baseline.fitness.total_cost
    ),
    delta_color="inverse",
)

metric_3.metric(
    label="Distância genética",
    value=(
        f"{result.best_fitness.total_distance_km:.2f} km"
    ),
)

metric_4.metric(
    label="Melhor geração",
    value=str(result.best_generation),
)


st.subheader("Comparação das abordagens")

comparison_dataframe = pd.DataFrame(
    [
        {
            "Abordagem": "Algoritmo genético",
            "Custo": (
                result.best_fitness.total_cost
            ),
            "Distância (km)": (
                result
                .best_fitness
                .total_distance_km
            ),
            "Penalidade de prioridade": (
                result
                .best_fitness
                .priority_penalty
            ),
            "Excesso de carga (kg)": (
                result
                .best_fitness
                .total_capacity_excess_kg
            ),
            "Excesso de autonomia (km)": (
                result
                .best_fitness
                .total_autonomy_excess_km
            ),
        },
        {
            "Abordagem": "Vizinho mais próximo",
            "Custo": (
                baseline.fitness.total_cost
            ),
            "Distância (km)": (
                baseline
                .fitness
                .total_distance_km
            ),
            "Penalidade de prioridade": (
                baseline
                .fitness
                .priority_penalty
            ),
            "Excesso de carga (kg)": (
                baseline
                .fitness
                .total_capacity_excess_kg
            ),
            "Excesso de autonomia (km)": (
                baseline
                .fitness
                .total_autonomy_excess_km
            ),
        },
    ]
)

st.dataframe(
    comparison_dataframe,
    use_container_width=True,
    hide_index=True,
)


st.subheader("Evolução do algoritmo")

history_dataframe = create_history_dataframe(
    result
)

st.line_chart(
    history_dataframe[
        [
            "Melhor custo",
            "Custo médio",
        ]
    ]
)


st.subheader("Rotas por veículo")

for route in result.best_fitness.routes:
    with st.expander(
        label=(
            f"{route.vehicle.name} — "
            f"{route.distance_km:.2f} km — "
            f"{route.load_kg:.1f} kg"
        ),
        expanded=True,
    ):
        route_dataframe = pd.DataFrame(
            [
                {
                    "Ordem": stop_position,
                    "Destino": delivery.name,
                    "Item": delivery.item,
                    "Prioridade": delivery.priority,
                    "Carga (kg)": delivery.demand_kg,
                }
                for stop_position, delivery in enumerate(
                    route.deliveries,
                    start=1,
                )
            ]
        )

        if route_dataframe.empty:
            st.write(
                "Nenhuma entrega atribuída."
            )
        else:
            st.dataframe(
                route_dataframe,
                use_container_width=True,
                hide_index=True,
            )

        if route.capacity_excess_kg > 0:
            st.error(
                "Excesso de carga: "
                f"{route.capacity_excess_kg:.2f} kg"
            )

        if route.autonomy_excess_km > 0:
            st.error(
                "Autonomia excedida em: "
                f"{route.autonomy_excess_km:.2f} km"
            )


st.subheader("Mapa das rotas")

routes_map = create_routes_map(
    depot=depot,
    fitness=result.best_fitness,
)

st_folium(
    routes_map,
    width=None,
    height=650,
    returned_objects=[],
)
