from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

from src.baseline import evaluate_nearest_neighbor
from src.genetic.optimizer import (
    GeneticConfig,
    OptimizationResult,
    optimize_routes,
)
from src.io_utils import load_scenario
from src.llm_service import (
    LlmServiceError,
    answer_route_question,
    evaluate_generated_report,
    generate_route_report,
)
from src.map_service import create_routes_map


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"

load_dotenv(
    PROJECT_ROOT / ".env",
    override=True,
)


st.set_page_config(
    page_title="Medical Route Optimizer",
    page_icon="🏥",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_application_scenario():
    return load_scenario(
        deliveries_path=DATA_DIRECTORY / "deliveries.json",
        vehicles_path=DATA_DIRECTORY / "vehicles.json",
        depot_path=DATA_DIRECTORY / "depot.json",
    )


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


def create_scenario_dataframe(deliveries) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": delivery.id,
                "Destino": delivery.name,
                "Item": delivery.item,
                "Carga (kg)": delivery.demand_kg,
                "Prioridade": delivery.priority,
                "Latitude": delivery.latitude,
                "Longitude": delivery.longitude,
            }
            for delivery in deliveries
        ]
    )


def initialize_session_state() -> None:
    initial_values = {
        "optimization_result": None,
        "baseline_result": None,
        "llm_report": None,
        "llm_answer": None,
    }

    for key, value in initial_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


deliveries, vehicles, depot = load_application_scenario()
initialize_session_state()


st.title("Medical Route Optimizer")

st.write(
    "Otimização de rotas para distribuição de medicamentos e insumos "
    "hospitalares utilizando algoritmo genético."
)

st.caption(
    "Os dados são sintéticos. As distâncias são estimadas pela fórmula "
    "de Haversine e não representam trajetos reais por ruas."
)


with st.sidebar:
    st.header("Configuração do algoritmo")

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


if run_optimization:
    st.session_state.optimization_result = None
    st.session_state.baseline_result = None
    st.session_state.llm_report = None
    st.session_state.llm_answer = None

    config = GeneticConfig(
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        elite_size=elite_size,
        random_seed=int(random_seed),
    )

    try:
        with st.spinner("Evoluindo as rotas..."):
            st.session_state.optimization_result = optimize_routes(
                deliveries=deliveries,
                vehicles=vehicles,
                depot=depot,
                config=config,
            )

            st.session_state.baseline_result = evaluate_nearest_neighbor(
                deliveries=deliveries,
                vehicles=vehicles,
                depot=depot,
            )

    except (ValueError, RuntimeError) as error:
        st.error(f"Não foi possível executar a otimização: {error}")


result = st.session_state.optimization_result
baseline = st.session_state.baseline_result


if result is None or baseline is None:
    st.info(
        "Configure os parâmetros na barra lateral e clique em "
        "**Executar otimização**."
    )

    st.subheader("Cenário de entregas")

    st.dataframe(
        create_scenario_dataframe(deliveries),
        use_container_width=True,
        hide_index=True,
    )

    vehicle_dataframe = pd.DataFrame(
        [
            {
                "Veículo": vehicle.name,
                "Capacidade (kg)": vehicle.capacity_kg,
                "Autonomia máxima (km)": vehicle.max_distance_km,
            }
            for vehicle in vehicles
        ]
    )

    st.subheader("Veículos disponíveis")

    st.dataframe(
        vehicle_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.stop()


st.subheader("Resultados principais")

cost_difference = (
    result.best_fitness.total_cost
    - baseline.fitness.total_cost
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    label="Custo genético",
    value=f"{result.best_fitness.total_cost:.2f}",
)

metric_2.metric(
    label="Custo do baseline",
    value=f"{baseline.fitness.total_cost:.2f}",
    delta=f"{cost_difference:.2f}",
    delta_color="inverse",
)

metric_3.metric(
    label="Distância genética",
    value=f"{result.best_fitness.total_distance_km:.2f} km",
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
            "Custo": result.best_fitness.total_cost,
            "Distância (km)": result.best_fitness.total_distance_km,
            "Penalidade de prioridade": result.best_fitness.priority_penalty,
            "Excesso de carga (kg)": (
                result.best_fitness.total_capacity_excess_kg
            ),
            "Excesso de autonomia (km)": (
                result.best_fitness.total_autonomy_excess_km
            ),
        },
        {
            "Abordagem": "Vizinho mais próximo",
            "Custo": baseline.fitness.total_cost,
            "Distância (km)": baseline.fitness.total_distance_km,
            "Penalidade de prioridade": baseline.fitness.priority_penalty,
            "Excesso de carga (kg)": (
                baseline.fitness.total_capacity_excess_kg
            ),
            "Excesso de autonomia (km)": (
                baseline.fitness.total_autonomy_excess_km
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

history_dataframe = create_history_dataframe(result)

st.line_chart(
    history_dataframe[
        [
            "Melhor custo",
            "Custo médio",
        ]
    ]
)


st.subheader("Melhor cromossomo encontrado")

st.code(
    str(list(result.best_chromosome)),
    language="python",
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
            st.write("Nenhuma entrega atribuída.")
        else:
            st.dataframe(
                route_dataframe,
                use_container_width=True,
                hide_index=True,
            )

        route_metric_1, route_metric_2, route_metric_3 = st.columns(3)

        route_metric_1.metric(
            label="Carga utilizada",
            value=f"{route.load_kg:.1f} kg",
        )

        route_metric_2.metric(
            label="Capacidade",
            value=f"{route.vehicle.capacity_kg:.1f} kg",
        )

        route_metric_3.metric(
            label="Distância",
            value=f"{route.distance_km:.2f} km",
        )

        if route.capacity_excess_kg > 0:
            st.error(
                "Excesso de carga: "
                f"{route.capacity_excess_kg:.2f} kg"
            )
        else:
            st.success("Capacidade respeitada.")

        if route.autonomy_excess_km > 0:
            st.error(
                "Autonomia excedida em: "
                f"{route.autonomy_excess_km:.2f} km"
            )
        else:
            st.success("Autonomia respeitada.")


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


st.subheader("Assistente de logística com LLM")

st.caption(
    "A LLM interpreta o resultado calculado pelo algoritmo. "
    "Ela não recalcula nem altera as rotas."
)

report_tab, question_tab = st.tabs(
    [
        "Relatório operacional",
        "Perguntas sobre as rotas",
    ]
)


with report_tab:
    st.write(
        "O relatório usa as rotas, cargas, distâncias e prioridades "
        "como contexto estruturado."
    )

    generate_report_button = st.button(
        label="Gerar relatório com LLM",
        type="primary",
        key="generate_llm_report",
    )

    if generate_report_button:
        with st.spinner("Gerando relatório com o Ollama..."):
            try:
                st.session_state.llm_report = generate_route_report(
                    result.best_fitness
                )

            except LlmServiceError as error:
                st.error(str(error))

    if st.session_state.llm_report:
        report = st.session_state.llm_report

        st.markdown(report)

        evaluation = evaluate_generated_report(
            report=report,
            fitness=result.best_fitness,
        )

        st.markdown("#### Avaliação automática do relatório")

        evaluation_column_1, evaluation_column_2 = st.columns(2)

        evaluation_column_1.metric(
            label="Cobertura das entregas",
            value=(
                f"{evaluation.delivery_coverage_percentage:.1f}%"
            ),
        )

        evaluation_column_2.metric(
            label="Rotas com ordem preservada",
            value=(
                f"{evaluation.routes_with_preserved_order}"
                f"/{evaluation.total_routes}"
            ),
        )

        if evaluation.missing_deliveries:
            st.warning(
                "Entregas não mencionadas: "
                + ", ".join(evaluation.missing_deliveries)
            )
        else:
            st.success("Todas as entregas foram mencionadas.")

        if evaluation.missing_critical_deliveries:
            st.error(
                "Entregas críticas não mencionadas: "
                + ", ".join(
                    evaluation.missing_critical_deliveries
                )
            )
        else:
            st.success(
                "Todas as entregas críticas foram mencionadas."
            )


with question_tab:
    question = st.text_input(
        label="Faça uma pergunta sobre as rotas",
        placeholder=(
            "Ex.: quais entregas críticas existem e "
            "qual veículo fará cada uma?"
        ),
        key="route_question",
    )

    ask_button = st.button(
        label="Perguntar à LLM",
        key="ask_llm_question",
    )

    if ask_button:
        if not question.strip():
            st.warning("Digite uma pergunta.")
        else:
            with st.spinner("Consultando as rotas..."):
                try:
                    st.session_state.llm_answer = (
                        answer_route_question(
                            fitness=result.best_fitness,
                            question=question,
                        )
                    )

                except LlmServiceError as error:
                    st.error(str(error))

    if st.session_state.llm_answer:
        st.markdown(st.session_state.llm_answer)
