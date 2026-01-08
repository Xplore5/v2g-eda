from abc import ABC, abstractmethod
from typing import List

class Vehicle(ABC):
    """
    Abstract base class for a vehicle.
    """
    def __init__(self, id: int, battery_capacity_kwh: float, consumption_kwh_per_km: float):
        self.id = id
        self.battery_capacity_kwh = battery_capacity_kwh
        self.soc_kwh = battery_capacity_kwh  # Start with a full battery
        self.soh = 1.0  # State of Health starts at 100%
        self.consumption_kwh_per_km = consumption_kwh_per_km

    def update_soc(self, energy_change_kwh: float):
        """
        Updates the state of charge of the vehicle.
        A positive value represents charging, a negative value represents discharging.
        """
        self.soc_kwh += energy_change_kwh
        # Clamp the SoC to be within the battery capacity
        self.soc_kwh = max(0, min(self.soc_kwh, self.battery_capacity_kwh))

    def update_soh(self, energy_throughput_kwh: float, degradation_params):
        """
        Updates the state of health of the vehicle's battery.
        """
        efc = energy_throughput_kwh / self.battery_capacity_kwh
        delta_soh_cycle = degradation_params.alpha_cycle * efc
        delta_soh_calendar = degradation_params.beta_calendar / (365 * 24 * (60 / 15)) # Assuming 15 min time steps
        self.soh -= (delta_soh_cycle + delta_soh_calendar)
        self.soh = max(0, self.soh)

    @abstractmethod
    def get_daily_schedule(self):
        """
        Returns the daily schedule of the vehicle.
        """
        pass

class EVCar(Vehicle):
    """
    Represents an electric car.
    """
    def __init__(self, id: int, battery_capacity_kwh: float, consumption_kwh_per_km: float,
                 daily_km_mean: float, daily_km_std: float, p_fast_charge: float):
        super().__init__(id, battery_capacity_kwh, consumption_kwh_per_km)
        self.daily_km_mean = daily_km_mean
        self.daily_km_std = daily_km_std
        self.p_fast_charge = p_fast_charge

    def get_daily_schedule(self):
        # For simplicity, we can assume a fixed schedule for now.
        # This can be replaced with a more sophisticated model later.
        return {"driving": [False] * 96} # 96 time steps of 15 mins in a day


class EVBus(Vehicle):
    """
    Represents an electric bus.
    """
    def __init__(self, id: int, battery_capacity_kwh: float, consumption_kwh_per_km: float,
                 weekly_km_pattern: List[float]):
        super().__init__(id, battery_capacity_kwh, consumption_kwh_per_km)
        self.weekly_km_pattern = weekly_km_pattern

    def get_daily_schedule(self):
        # For simplicity, we can assume a fixed schedule for now.
        # This can be replaced with a more sophisticated model later.
        return {"driving": [False] * 96} # 96 time steps of 15 mins in a day
