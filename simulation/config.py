from dataclasses import dataclass
from typing import Dict, List, Literal

@dataclass
class FleetParams:
    n_cars: int
    n_buses: int
    car_battery_kwh: float
    bus_battery_kwh: float
    car_consumption_kwh_per_km: float
    bus_consumption_kwh_per_km: float
    car_daily_km_mean: float
    car_daily_km_std: float
    bus_weekly_km: float   # or per-route pattern

@dataclass
class ChargerParams:
    n_ac_chargers: int
    n_dc_car_chargers: int
    n_dc_bus_chargers: int
    ac_power_kw: float
    dc_car_power_kw: float
    dc_bus_power_kw: float
    p_car_uses_dc: float   # ratio AC vs DC

@dataclass
class BatteryDegradationParams:
    alpha_cycle: float     # %SoH per EFC
    beta_calendar: float   # %SoH per year
    soh_start: float = 1.0
    soh_replacement: float = 0.8

@dataclass
class V2GParams:
    participation_rate: float    # 0–1
    soc_min_v2g: float           # fraction of capacity
    soc_max_charge: float
    max_v2g_power_kw: float
    strategy: Literal["peak_shaving", "price_driven"]

@dataclass
class GridParams:
    max_feeder_power_mw: float
    base_load_profile_mw: List[float]  # length = timesteps_per_week

@dataclass
class SimulationConfig:
    dt_minutes: int = 15
    n_runs: int = 200
    random_seed: int = 42
