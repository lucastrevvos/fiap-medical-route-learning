import folium

from src.genetic.fitness import evaluate_chromosome
from src.map_service import create_routes_map
from src.models import Delivery, Depot, Vehicle


def test_create_routes_map_returns_folium_map() -> None:
    depot = Depot(
        name="Hospital",
        latitude=-27.595,
        longitude=-48.548,
    )

    deliveries = [
        Delivery(
            id=1,
            name="Unidade 1",
            latitude=-27.60,
            longitude=-48.54,
            demand_kg=5.0,
            priority="critical",
            item="Insulina",
        ),
        Delivery(
            id=2,
            name="Unidade 2",
            latitude=-27.61,
            longitude=-48.53,
            demand_kg=5.0,
            priority="high",
            item="Antibióticos",
        ),
    ]

    vehicles = [
        Vehicle(
            id=1,
            name="Van 1",
            capacity_kg=20.0,
            max_distance_km=200.0,
        )
    ]

    fitness = evaluate_chromosome(
        chromosome=[1, 2],
        deliveries=deliveries,
        vehicles=vehicles,
        depot=depot,
    )

    routes_map = create_routes_map(
        depot=depot,
        fitness=fitness,
    )

    assert isinstance(
        routes_map,
        folium.Map,
    )


def test_map_html_contains_delivery_names() -> None:
    depot = Depot(
        name="Hospital",
        latitude=-27.595,
        longitude=-48.548,
    )

    delivery = Delivery(
        id=1,
        name="Unidade Teste",
        latitude=-27.60,
        longitude=-48.54,
        demand_kg=5.0,
        priority="critical",
        item="Insulina",
    )

    fitness = evaluate_chromosome(
        chromosome=[1],
        deliveries=[delivery],
        vehicles=[
            Vehicle(
                id=1,
                name="Van 1",
                capacity_kg=20.0,
                max_distance_km=200.0,
            )
        ],
        depot=depot,
    )

    routes_map = create_routes_map(
        depot=depot,
        fitness=fitness,
    )

    rendered_html = (
        routes_map
        .get_root()
        .render()
    )

    assert "Unidade Teste" in rendered_html
    assert "Insulina" in rendered_html
    assert "Hospital" in rendered_html
