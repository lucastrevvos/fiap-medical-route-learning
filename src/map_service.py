from __future__ import annotations

import folium

from src.genetic.fitness import FitnessResult
from src.models import Depot


ROUTE_COLORS = [
    "blue",
    "green",
    "purple",
    "orange",
    "darkred",
    "cadetblue",
]


def create_routes_map(
    depot: Depot,
    fitness: FitnessResult,
) -> folium.Map:
    routes_map = folium.Map(
        location=[
            depot.latitude,
            depot.longitude,
        ],
        zoom_start=11,
        tiles="OpenStreetMap",
    )

    folium.Marker(
        location=[
            depot.latitude,
            depot.longitude,
        ],
        tooltip=depot.name,
        popup=(
            f"<strong>{depot.name}</strong>"
            "<br>Saída e retorno dos veículos"
        ),
        icon=folium.Icon(
            color="red",
            icon="plus-sign",
        ),
    ).add_to(routes_map)

    for route_index, route in enumerate(
        fitness.routes
    ):
        color = ROUTE_COLORS[
            route_index % len(ROUTE_COLORS)
        ]

        coordinates = [
            [
                depot.latitude,
                depot.longitude,
            ]
        ]

        for stop_position, delivery in enumerate(
            route.deliveries,
            start=1,
        ):
            delivery_coordinates = [
                delivery.latitude,
                delivery.longitude,
            ]

            coordinates.append(
                delivery_coordinates
            )

            popup_content = (
                f"<strong>{stop_position}. "
                f"{delivery.name}</strong>"
                f"<br>Veículo: {route.vehicle.name}"
                f"<br>Item: {delivery.item}"
                f"<br>Prioridade: {delivery.priority}"
                f"<br>Carga: {delivery.demand_kg:.1f} kg"
            )

            folium.Marker(
                location=delivery_coordinates,
                tooltip=(
                    f"{stop_position}. "
                    f"{delivery.name}"
                ),
                popup=popup_content,
                icon=folium.Icon(
                    color=color,
                    icon="info-sign",
                ),
            ).add_to(routes_map)

        coordinates.append(
            [
                depot.latitude,
                depot.longitude,
            ]
        )

        if route.deliveries:
            folium.PolyLine(
                locations=coordinates,
                color=color,
                weight=5,
                opacity=0.8,
                tooltip=(
                    f"{route.vehicle.name}: "
                    f"{route.distance_km:.2f} km"
                ),
            ).add_to(routes_map)

    return routes_map
