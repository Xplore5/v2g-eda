import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.config import FleetParams, ChargerParams, BatteryDegradationParams, V2GParams, GridParams, SimulationConfig
from simulation.scenarios import ScenarioRunner

def main():
    """
    Main function to run the simulation.
    """
    # Create a dummy base load profile for a week
    base_load_profile_mw = [100] * (7 * 24 * 4) # 15 minute intervals

    fleet_params = FleetParams(
        n_cars=100,
        n_buses=10,
        car_battery_kwh=60,
        bus_battery_kwh=300,
        car_consumption_kwh_per_km=0.15,
        bus_consumption_kwh_per_km=1.2,
        car_daily_km_mean=40,
        car_daily_km_std=10,
        bus_weekly_km=1000
    )

    charger_params = ChargerParams(
        n_ac_chargers=50,
        n_dc_car_chargers=10,
        n_dc_bus_chargers=5,
        ac_power_kw=11,
        dc_car_power_kw=50,
        dc_bus_power_kw=150,
        p_car_uses_dc=0.2
    )

    degradation_params = BatteryDegradationParams(
        alpha_cycle=0.0005,
        beta_calendar=0.02
    )

    v2g_params = V2GParams(
        participation_rate=0.5,
        soc_min_v2g=0.3,
        soc_max_charge=0.9,
        max_v2g_power_kw=7,
        strategy="peak_shaving"
    )

    grid_params = GridParams(
        max_feeder_power_mw=200,
        base_load_profile_mw=base_load_profile_mw
    )

    sim_config = SimulationConfig(
        dt_minutes=15,
        n_runs=1, # Set to 1 for a single run for now
        random_seed=42
    )

    scenario_runner = ScenarioRunner(fleet_params, charger_params, degradation_params,
                                     v2g_params, grid_params, sim_config)

    results = scenario_runner.run_sweep()
    aggregated_results = scenario_runner.aggregate_results(results)

    print(aggregated_results)

if __name__ == "__main__":
    main()
