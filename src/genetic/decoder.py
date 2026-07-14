from src.distance import route_distance_km
from src.genetic.population import (
    Chromosome,
    validate_chromosome,
)
from src.models import (
    Delivery,
    Depot,
    Vehicle,
    VehicleRoute,
)


def decode_chromosome(
    chromosome: Chromosome,
    deliveries: list[Delivery],
    vehicles: list[Vehicle],
    depot: Depot,
) -> list[VehicleRoute]:
    if not vehicles:
        raise ValueError(
            "Não é possível decodificar uma rota sem veículos."
        )

    validate_chromosome(
        chromosome=chromosome,
        deliveries=deliveries,
    )

    delivery_by_id = {
        delivery.id: delivery
        for delivery in deliveries
    }

    allocations: list[list[Delivery]] = [
        []
        for _ in vehicles
    ]

    vehicle_loads = [
        0.0
        for _ in vehicles
    ]

    current_vehicle_index = 0

    for delivery_id in chromosome:
        delivery = delivery_by_id[delivery_id]

        while (
            current_vehicle_index < len(vehicles) - 1
            and vehicle_loads[current_vehicle_index]
            + delivery.demand_kg
            > vehicles[current_vehicle_index].capacity_kg
        ):
            current_vehicle_index += 1

        allocations[current_vehicle_index].append(
            delivery
        )

        vehicle_loads[current_vehicle_index] += (
            delivery.demand_kg
        )

    routes: list[VehicleRoute] = []

    for vehicle, assigned_deliveries, load_kg in zip(
        vehicles,
        allocations,
        vehicle_loads,
    ):
        distance_km = route_distance_km(
            depot=depot,
            stops=assigned_deliveries,
        )

        capacity_excess_kg = max(
            0.0,
            load_kg - vehicle.capacity_kg,
        )

        autonomy_excess_km = max(
            0.0,
            distance_km - vehicle.max_distance_km,
        )

        route = VehicleRoute(
            vehicle=vehicle,
            deliveries=tuple(assigned_deliveries),
            load_kg=load_kg,
            distance_km=distance_km,
            capacity_excess_kg=capacity_excess_kg,
            autonomy_excess_km=autonomy_excess_km,
        )

        routes.append(route)

    return routes
