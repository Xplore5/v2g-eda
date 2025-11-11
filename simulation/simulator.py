from simulation.config import FleetParams, ChargerParams, BatteryDegradationParams, V2GParams, GridParams, SimulationConfig
from simulation.entities import EVCar, EVBus
from simulation.environment import ChargerPool, Grid
from simulation.results import SimulationResult
import numpy as np

class Simulator:
    """
    Core simulator class with the main time loop.
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
        self.timesteps_per_week = int(7 * 24 * 60 / self.sim_config.dt_minutes)

    def run_single_iteration(self) -> SimulationResult:
        """
        Runs a single iteration of the simulation.
        """
        # Initialize vehicles
        cars = [EVCar(id=i,
                        battery_capacity_kwh=self.fleet_params.car_battery_kwh,
                        consumption_kwh_per_km=self.fleet_params.car_consumption_kwh_per_km,
                        daily_km_mean=self.fleet_params.car_daily_km_mean,
                        daily_km_std=self.fleet_params.car_daily_km_std,
                        p_fast_charge=self.charger_params.p_car_uses_dc)
                for i in range(self.fleet_params.n_cars)]
        buses = [EVBus(id=i,
                         battery_capacity_kwh=self.fleet_params.bus_battery_kwh,
                         consumption_kwh_per_km=self.fleet_params.bus_consumption_kwh_per_km,
                         weekly_km_pattern=[self.fleet_params.bus_weekly_km / 7] * 7)
                 for i in range(self.fleet_params.n_buses)]

        # Initialize charger pools
        ac_charger_pool = ChargerPool(num_chargers=self.charger_params.n_ac_chargers,
                                      power_kw=self.charger_params.ac_power_kw)
        dc_car_charger_pool = ChargerPool(num_chargers=self.charger_params.n_dc_car_chargers,
                                          power_kw=self.charger_params.dc_car_power_kw)
        dc_bus_charger_pool = ChargerPool(num_chargers=self.charger_params.n_dc_bus_chargers,
                                          power_kw=self.charger_params.dc_bus_power_kw)

        # Initialize grid
        grid = Grid(max_feeder_power_mw=self.grid_params.max_feeder_power_mw,
                    base_load_profile_mw=self.grid_params.base_load_profile_mw)

        # Initialize results
        p_charge_mw = [0.0] * self.timesteps_per_week
        p_v2g_mw = [0.0] * self.timesteps_per_week
        p_net_mw = [0.0] * self.timesteps_per_week
        n_connected_cars = [0] * self.timesteps_per_week
        n_connected_buses = [0] * self.timesteps_per_week

            # Time loop
        for t in range(self.timesteps_per_week):
            # Process cars
            for car in cars:
                # Simple driving model: 20% chance of driving at any time step
                if np.random.rand() < 0.2:
                    distance_per_step = car.daily_km_mean / (24 * 60 / self.sim_config.dt_minutes)
                    energy_consumed = distance_per_step * car.consumption_kwh_per_km
                    car.update_soc(-energy_consumed)
                else: # Parked
                    n_connected_cars[t] += 1
                    # Charge if SoC is below max charge threshold
                    if car.soc_kwh < self.v2g_params.soc_max_charge * car.battery_capacity_kwh:
                        if np.random.rand() < self.charger_params.p_car_uses_dc:
                            if dc_car_charger_pool.request_charger():
                                energy_added = self.charger_params.dc_car_power_kw * (self.sim_config.dt_minutes / 60)
                                car.update_soc(energy_added)
                                p_charge_mw[t] += self.charger_params.dc_car_power_kw / 1000 # convert to MW
                                dc_car_charger_pool.release_charger()
                        else:
                            if ac_charger_pool.request_charger():
                                energy_added = self.charger_params.ac_power_kw * (self.sim_config.dt_minutes / 60)
                                car.update_soc(energy_added)
                                p_charge_mw[t] += self.charger_params.ac_power_kw / 1000 # convert to MW
                                ac_charger_pool.release_charger()
            # Process buses
            for bus in buses:
                # Simple driving model: 50% chance of driving at any time step
                if np.random.rand() < 0.5:
                    distance_per_step = (bus.weekly_km_pattern[0] * 7) / (24 * 60 / self.sim_config.dt_minutes)
                    energy_consumed = distance_per_step * bus.consumption_kwh_per_km
                    bus.update_soc(-energy_consumed)
                else: # Parked
                    n_connected_buses[t] += 1
                    # Charge if SoC is below max charge threshold
                    if bus.soc_kwh < self.v2g_params.soc_max_charge * bus.battery_capacity_kwh:
                        if dc_bus_charger_pool.request_charger():
                            energy_added = self.charger_params.dc_bus_power_kw * (self.sim_config.dt_minutes / 60)
                            bus.update_soc(energy_added)
                            p_charge_mw[t] += self.charger_params.dc_bus_power_kw / 1000 # convert to MW
                            dc_bus_charger_pool.release_charger()

            # Update net load
            p_net_mw[t] = grid.get_current_base_load(t) + p_charge_mw[t] - p_v2g_mw[t]


        # Calculate end-of-week metrics
        p_net_mw_np = np.array(p_net_mw)
        mean_peak_load = np.mean(p_net_mw_np)
        p95_peak_load = np.percentile(p_net_mw_np, 95)
        
        overload_frequency = np.sum(p_net_mw_np > self.grid_params.max_feeder_power_mw)
        overload_probability = overload_frequency / self.timesteps_per_week

        # Battery degradation
        total_energy_throughput_car = sum([c.battery_capacity_kwh - c.soc_kwh for c in cars])
        average_weekly_efc_car = total_energy_throughput_car / self.fleet_params.car_battery_kwh / self.fleet_params.n_cars if self.fleet_params.n_cars > 0 else 0
        average_weekly_soh_loss_car = self.degradation_params.alpha_cycle * average_weekly_efc_car + self.degradation_params.beta_calendar / 52
        battery_replacement_rate_car_per_year = (1 - self.degradation_params.soh_replacement) / average_weekly_soh_loss_car / 52 if average_weekly_soh_loss_car > 0 else 0

        total_energy_throughput_bus = sum([b.battery_capacity_kwh - b.soc_kwh for b in buses])
        average_weekly_efc_bus = total_energy_throughput_bus / self.fleet_params.bus_battery_kwh / self.fleet_params.n_buses if self.fleet_params.n_buses > 0 else 0
        average_weekly_soh_loss_bus = self.degradation_params.alpha_cycle * average_weekly_efc_bus + self.degradation_params.beta_calendar / 52
        battery_replacement_rate_bus_per_year = (1 - self.degradation_params.soh_replacement) / average_weekly_soh_loss_bus / 52 if average_weekly_soh_loss_bus > 0 else 0

        
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
