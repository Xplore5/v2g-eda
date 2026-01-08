from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SimulationResult:
    """
    Dataclass to store the results of a single simulation run.
    """
    p_charge_mw: List[float]
    p_v2g_mw: List[float]
    p_net_mw: List[float]
    n_connected_cars: List[int]
    n_connected_buses: List[int]
    mean_peak_load: float
    p95_peak_load: float
    overload_frequency: float
    overload_probability: float
    average_weekly_efc_car: float
    average_weekly_soh_loss_car: float
    average_weekly_efc_bus: float
    average_weekly_soh_loss_bus: float
    battery_replacement_rate_car_per_year: float
    battery_replacement_rate_bus_per_year: float
