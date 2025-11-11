from typing import List
from simulation.config import FleetParams, ChargerParams, BatteryDegradationParams, V2GParams, GridParams, SimulationConfig
from simulation.simulator import Simulator
from simulation.results import SimulationResult
import numpy as np

class ScenarioRunner:
    """
    ScenarioRunner class to manage sweeps of simulations.
    """
    def __init__(self, fleet_params: FleetParams, charger_params: ChargerParams,
                 degradation_params: BatteryDegradationParams, v2g_params: V2GParams,
                 grid_params: GridParams, sim_config: SimulationConfig):
        self.fleet_params = fleet_params
        self.charger_params = charger_params
        self.degradation_params = degradation_params
        self.v2g_params = v2g_params
        self.grid_params = grid_params
        self.sim_config = sim_config

    def run_sweep(self) -> List[SimulationResult]:
        """
        Runs a sweep of simulations.
        """
        results = []
        for i in range(self.sim_config.n_runs):
            simulator = Simulator(self.fleet_params, self.charger_params, self.degradation_params,
                                  self.v2g_params, self.grid_params, self.sim_config)
            result = simulator.run_single_iteration()
            results.append(result)
        return results

    def aggregate_results(self, results: List[SimulationResult]) -> SimulationResult:
        """
        Aggregates the results of a sweep of simulations.
        """
        if not results:
            return None

        # Aggregate time-series data
        p_charge_mw = np.mean([r.p_charge_mw for r in results], axis=0).tolist()
        p_v2g_mw = np.mean([r.p_v2g_mw for r in results], axis=0).tolist()
        p_net_mw = np.mean([r.p_net_mw for r in results], axis=0).tolist()
        n_connected_cars = np.mean([r.n_connected_cars for r in results], axis=0).tolist()
        n_connected_buses = np.mean([r.n_connected_buses for r in results], axis=0).tolist()

        # Aggregate scalar metrics
        mean_peak_load = np.mean([r.mean_peak_load for r in results])
        p95_peak_load = np.mean([r.p95_peak_load for r in results])
        overload_frequency = np.mean([r.overload_frequency for r in results])
        overload_probability = np.mean([r.overload_probability for r in results])
        average_weekly_efc_car = np.mean([r.average_weekly_efc_car for r in results])
        average_weekly_soh_loss_car = np.mean([r.average_weekly_soh_loss_car for r in results])
        average_weekly_efc_bus = np.mean([r.average_weekly_efc_bus for r in results])
        average_weekly_soh_loss_bus = np.mean([r.average_weekly_soh_loss_bus for r in results])
        battery_replacement_rate_car_per_year = np.mean([r.battery_replacement_rate_car_per_year for r in results])
        battery_replacement_rate_bus_per_year = np.mean([r.battery_replacement_rate_bus_per_year for r in results])

        return SimulationResult(
            p_charge_mw=p_charge_mw,
            p_v2g_mw=p_v2g_mw,
            p_net_mw=p_net_mw,
            n_connected_cars=n_connected_cars,
            n_connected_buses=n_connected_buses,
            mean_peak_load=mean_peak_load,
            p95_peak_load=p95_peak_load,
            overload_frequency=overload_frequency,
            overload_probability=overload_probability,
            average_weekly_efc_car=average_weekly_efc_car,
            average_weekly_soh_loss_car=average_weekly_soh_loss_car,
            average_weekly_efc_bus=average_weekly_efc_bus,
            average_weekly_soh_loss_bus=average_weekly_soh_loss_bus,
            battery_replacement_rate_car_per_year=battery_replacement_rate_car_per_year,
            battery_replacement_rate_bus_per_year=battery_replacement_rate_bus_per_year,
        )
