from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

import requests

from src.genetic.fitness import FitnessResult


SYSTEM_PROMPT = """
Você é um assistente de logística hospitalar.

Sua responsabilidade é explicar rotas já calculadas por um algoritmo
genético. Você não deve recalcular, alterar ou substituir as rotas.

Regras obrigatórias:

1. Utilize exclusivamente os dados fornecidos no contexto.
2. Preserve exatamente a ordem das paradas de cada veículo.
3. Não invente endereços, horários, motoristas ou medicamentos.
4. Não forneça diagnóstico nem recomendação médica.
5. Destaque entregas de prioridade critical e high.
6. Informe claramente qualquer excesso de carga ou autonomia.
7. Considere que as distâncias são estimativas geodésicas,
   e não distâncias reais calculadas pelas ruas.
8. Responda em português do Brasil.
9. Quando uma informação não estiver no contexto, diga que ela
   não foi fornecida.
10. O valor total_cost é uma pontuação da função objetivo,
    sem unidade monetária. Nunca use R$, reais, preço ou economia.
11. O valor priority_penalty também é uma pontuação abstrata,
    e não possui unidade monetária.
12. Não afirme que houve economia ou redução de custos financeiros,
    a menos que o contexto forneça explicitamente essa comparação.
13. Não use elogios ou conclusões como desempenho excepcional,
    precisão garantida ou sucesso absoluto sem evidências no contexto.
14. Refira-se à solução como algoritmo genético de otimização de rotas.
15. Ao escrever o relatório, traduza as prioridades:
    critical como crítica, high como alta, medium como média
    e low como baixa.
""".strip()


class LlmServiceError(RuntimeError):
    """Erro controlado durante a comunicação com a LLM."""


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    timeout_seconds: float
    temperature: float

    @classmethod
    def from_environment(cls) -> "LlmConfig":
        try:
            timeout_seconds = float(
                os.getenv(
                    "OLLAMA_TIMEOUT_SECONDS",
                    "120",
                )
            )

            temperature = float(
                os.getenv(
                    "OLLAMA_TEMPERATURE",
                    "0.2",
                )
            )
        except ValueError as error:
            raise LlmServiceError(
                "As configurações numéricas do Ollama são inválidas."
            ) from error

        return cls(
            base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            ).rstrip("/"),
            model=os.getenv(
                "OLLAMA_MODEL",
                "qwen2.5-coder:7b",
            ),
            timeout_seconds=timeout_seconds,
            temperature=temperature,
        )


@dataclass(frozen=True)
class ReportEvaluation:
    delivery_coverage_percentage: float
    missing_deliveries: tuple[str, ...]
    missing_critical_deliveries: tuple[str, ...]
    routes_with_preserved_order: int
    total_routes: int


def fitness_to_context(
    fitness: FitnessResult,
) -> dict[str, Any]:
    return {
        "summary": {
    "objective_score": round(
        fitness.total_cost,
        3,
    ),
    "objective_score_description": (
        "Pontuação abstrata da função objetivo. "
        "Quanto menor, melhor. Não representa dinheiro."
    ),
    "total_distance_km": round(
        fitness.total_distance_km,
        3,
    ),
    "priority_penalty_score": round(
        fitness.priority_penalty,
        3,
    ),
    "priority_penalty_description": (
        "Penalidade abstrata causada pela posição "
        "das entregas prioritárias. Não representa dinheiro."
    ),
    "total_capacity_excess_kg": round(
        fitness.total_capacity_excess_kg,
        3,
    ),
    "total_autonomy_excess_km": round(
        fitness.total_autonomy_excess_km,
        3,
    ),
    "distance_calculation": (
        "Haversine: estimativa geodésica, "
        "não distância real pelas ruas."
    ),
},
        "routes": [
            {
                "vehicle": {
                    "id": route.vehicle.id,
                    "name": route.vehicle.name,
                    "capacity_kg": route.vehicle.capacity_kg,
                    "max_distance_km": (
                        route.vehicle.max_distance_km
                    ),
                },
                "route_metrics": {
                    "load_kg": round(
                        route.load_kg,
                        3,
                    ),
                    "distance_km": round(
                        route.distance_km,
                        3,
                    ),
                    "capacity_excess_kg": round(
                        route.capacity_excess_kg,
                        3,
                    ),
                    "autonomy_excess_km": round(
                        route.autonomy_excess_km,
                        3,
                    ),
                },
                "stops": [
                    {
                        "order": position,
                        "delivery_id": delivery.id,
                        "destination": delivery.name,
                        "item": delivery.item,
                        "priority": delivery.priority,
                        "demand_kg": delivery.demand_kg,
                    }
                    for position, delivery in enumerate(
                        route.deliveries,
                        start=1,
                    )
                ],
            }
            for route in fitness.routes
        ],
    }


def build_report_messages(
    fitness: FitnessResult,
) -> list[dict[str, str]]:
    context = json.dumps(
        fitness_to_context(fitness),
        ensure_ascii=False,
        indent=2,
    )

    user_prompt = f"""
Gere um relatório operacional baseado no contexto abaixo.

O relatório deve conter:

1. Resumo executivo objetivo.
2. Indicadores gerais.
3. Instruções numeradas para cada veículo.
4. Alertas para entregas críticas.
5. Alertas de capacidade ou autonomia.
6. Uma recomendação final baseada somente nos dados.

Regras específicas para este relatório:

- Não altere a ordem das paradas.
- Não use R$ ou qualquer unidade monetária.
- Chame objective_score de pontuação da função objetivo.
- Não descreva a pontuação como economia financeira.
- Não diga que o algoritmo garantiu precisão ou resultado ótimo.
- Use o termo algoritmo genético de otimização de rotas.

CONTEXTO DAS ROTAS:

{context}
""".strip()

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def build_question_messages(
    fitness: FitnessResult,
    question: str,
) -> list[dict[str, str]]:
    if not question.strip():
        raise ValueError(
            "A pergunta não pode ficar vazia."
        )

    context = json.dumps(
        fitness_to_context(fitness),
        ensure_ascii=False,
        indent=2,
    )

    user_prompt = f"""
Responda à pergunta usando exclusivamente o contexto das rotas.

Caso a resposta não esteja presente no contexto, informe isso
claramente. Não invente dados.

PERGUNTA:

{question.strip()}

CONTEXTO DAS ROTAS:

{context}
""".strip()

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def call_ollama_chat(
    messages: list[dict[str, str]],
    config: LlmConfig | None = None,
) -> str:
    effective_config = (
        config
        or LlmConfig.from_environment()
    )

    endpoint = (
        f"{effective_config.base_url.strip().rstrip('/')}"
        "/api/chat"
    )

    payload = {
        "model": effective_config.model.strip(),
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": (
                effective_config.temperature
            ),
        },
    }

    try:
        response = requests.post(
            url=endpoint,
            json=payload,
            timeout=effective_config.timeout_seconds,
        )

    except requests.Timeout as error:
        raise LlmServiceError(
            "O Ollama demorou mais que "
            f"{effective_config.timeout_seconds:.0f} segundos "
            "para responder. Aumente OLLAMA_TIMEOUT_SECONDS."
        ) from error

    except requests.ConnectionError as error:
        raise LlmServiceError(
            f"Não foi possível acessar {endpoint}. "
            "Confirme o endereço configurado no arquivo .env."
        ) from error

    except requests.RequestException as error:
        raise LlmServiceError(
            f"Falha inesperada ao chamar o Ollama: {error}"
        ) from error

    if not response.ok:
        response_details = response.text.strip()

        raise LlmServiceError(
            "O Ollama respondeu com erro HTTP "
            f"{response.status_code}.\n\n"
            f"Detalhes: {response_details}"
        )

    try:
        response_data = response.json()

        content = (
            response_data
            .get("message", {})
            .get("content", "")
            .strip()
        )

    except (ValueError, AttributeError) as error:
        raise LlmServiceError(
            "O Ollama respondeu, mas o conteúdo não é "
            f"um JSON válido. Resposta: {response.text[:500]}"
        ) from error

    if not content:
        raise LlmServiceError(
            "O Ollama respondeu sem conteúdo textual. "
            f"Resposta recebida: {response.text[:500]}"
        )

    return content


def generate_route_report(
    fitness: FitnessResult,
    config: LlmConfig | None = None,
) -> str:
    return call_ollama_chat(
        messages=build_report_messages(
            fitness
        ),
        config=config,
    )


def answer_route_question(
    fitness: FitnessResult,
    question: str,
    config: LlmConfig | None = None,
) -> str:
    return call_ollama_chat(
        messages=build_question_messages(
            fitness=fitness,
            question=question,
        ),
        config=config,
    )


def evaluate_generated_report(
    report: str,
    fitness: FitnessResult,
) -> ReportEvaluation:
    normalized_report = report.casefold()

    deliveries = [
        delivery
        for route in fitness.routes
        for delivery in route.deliveries
    ]

    missing_deliveries = tuple(
        delivery.name
        for delivery in deliveries
        if delivery.name.casefold()
        not in normalized_report
    )

    critical_deliveries = [
        delivery
        for delivery in deliveries
        if delivery.priority == "critical"
    ]

    missing_critical_deliveries = tuple(
        delivery.name
        for delivery in critical_deliveries
        if delivery.name.casefold()
        not in normalized_report
    )

    mentioned_count = (
        len(deliveries)
        - len(missing_deliveries)
    )

    coverage_percentage = (
        mentioned_count / len(deliveries) * 100
        if deliveries
        else 100.0
    )

    routes_with_preserved_order = 0
    routes_with_deliveries = [
        route
        for route in fitness.routes
        if route.deliveries
    ]

    for route in routes_with_deliveries:
        positions = [
            normalized_report.find(
                delivery.name.casefold()
            )
            for delivery in route.deliveries
        ]

        all_destinations_found = all(
            position >= 0
            for position in positions
        )

        if (
            all_destinations_found
            and positions == sorted(positions)
        ):
            routes_with_preserved_order += 1

    return ReportEvaluation(
        delivery_coverage_percentage=(
            coverage_percentage
        ),
        missing_deliveries=missing_deliveries,
        missing_critical_deliveries=(
            missing_critical_deliveries
        ),
        routes_with_preserved_order=(
            routes_with_preserved_order
        ),
        total_routes=len(
            routes_with_deliveries
        ),
    )
