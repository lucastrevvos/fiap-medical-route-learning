from dataclasses import dataclass
from typing import Literal


Priority = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class Depot:
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Delivery:
    id: int
    name: str
    latitude: float
    longitude: float
    demand_kg: float
    priority: Priority
    item: str


@dataclass(frozen=True)
class Vehicle:
    id: int
    name: str
    capacity_kg: float
    max_distance_km: float
