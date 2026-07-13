import json
from pathlib import Path

from src.models import Delivery, Depot, Vehicle


VALID_PRIORITIES = {
    "low",
    "medium",
    "high",
    "critical",
}


def load_json(file_path: str | Path) -> object:
    path = Path(file_path)

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_scenario(
    deliveries_path: str | Path,
    vehicles_path: str | Path,
    depot_path: str | Path,
) -> tuple[list[Delivery], list[Vehicle], Depot]:
    deliveries_data = load_json(deliveries_path)
    vehicles_data = load_json(vehicles_path)
    depot_data = load_json(depot_path)

    deliveries = [
        Delivery(**delivery_data)
        for delivery_data in deliveries_data
    ]

    vehicles = [
        Vehicle(**vehicle_data)
        for vehicle_data in vehicles_data
    ]

    depot = Depot(**depot_data)

    validate_scenario(
        deliveries=deliveries,
        vehicles=vehicles,
        depot=depot,
    )

    return deliveries, vehicles, depot


def validate_scenario(
    deliveries: list[Delivery],
    vehicles: list[Vehicle],
    depot: Depot,
) -> None:
    if not deliveries:
        raise ValueError(
            "O cenário precisa conter pelo menos uma entrega."
        )

    if not vehicles:
        raise ValueError(
            "O cenário precisa conter pelo menos um veículo."
        )

    delivery_ids = [
        delivery.id
        for delivery in deliveries
    ]

    if len(delivery_ids) != len(set(delivery_ids)):
        raise ValueError(
            "Existem IDs de entrega duplicados."
        )

    vehicle_ids = [
        vehicle.id
        for vehicle in vehicles
    ]

    if len(vehicle_ids) != len(set(vehicle_ids)):
        raise ValueError(
            "Existem IDs de veículo duplicados."
        )

    for delivery in deliveries:
        if delivery.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Prioridade inválida na entrega {delivery.id}."
            )

        if delivery.demand_kg <= 0:
            raise ValueError(
                f"A entrega {delivery.id} possui carga inválida."
            )

        validate_coordinates(
            latitude=delivery.latitude,
            longitude=delivery.longitude,
            object_name=f"entrega {delivery.id}",
        )

    for vehicle in vehicles:
        if vehicle.capacity_kg <= 0:
            raise ValueError(
                f"O veículo {vehicle.id} possui capacidade inválida."
            )

        if vehicle.max_distance_km <= 0:
            raise ValueError(
                f"O veículo {vehicle.id} possui autonomia inválida."
            )

    validate_coordinates(
        latitude=depot.latitude,
        longitude=depot.longitude,
        object_name="depósito",
    )

    total_demand = sum(
        delivery.demand_kg
        for delivery in deliveries
    )

    total_capacity = sum(
        vehicle.capacity_kg
        for vehicle in vehicles
    )

    if total_demand > total_capacity:
        raise ValueError(
            "A carga total das entregas é maior que a capacidade "
            "total dos veículos."
        )


def validate_coordinates(
    latitude: float,
    longitude: float,
    object_name: str,
) -> None:
    if not -90 <= latitude <= 90:
        raise ValueError(
            f"Latitude inválida para {object_name}."
        )

    if not -180 <= longitude <= 180:
        raise ValueError(
            f"Longitude inválida para {object_name}."
        )
