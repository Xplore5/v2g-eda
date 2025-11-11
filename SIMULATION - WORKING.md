## System Architecture & Design

This section outlines the proposed software architecture for the simulation engine. The design emphasizes modularity, configurability, and extensibility, adhering to the non-functional requirements.

### 1. Overall Architecture

The simulation engine is composed of several interconnected modules that manage configuration, simulation entities, the core simulation loop, and results aggregation. The `ScenarioRunner` acts as the main orchestrator, taking user-defined configurations and managing the Monte-Carlo runs, while the `Simulator` executes the core time-step-based logic for a single weekly run.

```mermaid
graph TD
    subgraph User Input
        A[Config Files / Scripts] --> B(ScenarioRunner);
    end

    subgraph Simulation Core
        B --> C{Simulator};
        D(Configuration) --> C;
        E(Entities: Vehicles) --> C;
        F(Environment: Grid, Chargers) --> C;
    end

    subgraph Simulation Logic
        C -- Manages --> E;
        C -- Interacts with --> F;
        C -- Produces --> G(SimulationResult);
    end

    subgraph Output
        B -- Aggregates --> G;
        G --> H[Metrics & Plots];
    end

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#ccf,stroke:#333,stroke-width:2px
    style F fill:#cfc,stroke:#333,stroke-width:2px
    style G fill:#fcf,stroke:#333,stroke-width:2px
```

### 2. Module Breakdown

The simulation will be built within a dedicated `simulation` submodule, with the following structure:

```
simulation/
├── __init__.py
├── config.py         # Pydantic/dataclasses for all input parameters
├── entities.py       # Vehicle, EVCar, EVBus classes
├── environment.py    # Grid, ChargerPool classes
├── main.py           # Top-level entry point for running simulations
├── results.py        # Dataclasses for storing and handling simulation results
├── scenarios.py      # ScenarioRunner class to manage sweeps
├── simulator.py      # Core Simulator class with the main time loop
└── utils.py          # Helper functions, distributions, etc.
```

---

### 3. Submodule Details

#### 3.1. `config.py` - Configuration Management

This module defines the data structures for all simulation inputs, using Pydantic or dataclasses to ensure type safety and clear validation. It directly maps to the `Inputs & API design` section.

```mermaid
graph TD
    subgraph config.py
        A(SimulationConfig)
        B(FleetParams)
        C(ChargerParams)
        D(BatteryDegradationParams)
        E(V2GParams)
        F(GridParams)
    end
    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
```

#### 3.2. `entities.py` - Vehicle Modeling

This module contains the classes that represent the agents in our simulation: the vehicles. A base `Vehicle` class will define common attributes and methods, with specialized classes for `EVCar` and `EVBus`.

```mermaid
classDiagram
    class Vehicle {
        <<abstract>>
        +id: int
        +battery_capacity_kwh: float
        +soc_kwh: float
        +soh: float
        +consumption_kwh_per_km: float
        +update_soc(energy_change_kwh)
        +update_soh(energy_throughput_kwh)
        +get_daily_schedule()
    }
    class EVCar {
        +daily_km_mean: float
        +daily_km_std: float
        +p_fast_charge: float
    }
    class EVBus {
        +weekly_km_pattern: List[float]
    }

    Vehicle <|-- EVCar
    Vehicle <|-- EVBus
```

#### 3.3. `environment.py` - Grid and Chargers

This module models the static components of the simulation world: the electrical grid and the charger infrastructure.

```mermaid
classDiagram
    class ChargerPool {
        +num_chargers: int
        +power_kw: float
        +available_chargers: int
        +request_charger(): bool
        +release_charger(): void
    }
    class Grid {
        +max_feeder_power_mw: float
        +base_load_profile_mw: List[float]
        +get_current_base_load(t): float
        +check_overload(net_load_mw): bool
    }
```

#### 3.4. `simulator.py` - Core Simulation Engine

This is the heart of the engine, containing the main time-step loop for a single Monte-Carlo run. It orchestrates the interactions between vehicles and the environment.

```mermaid
graph TD
    A([Start: run_single_iteration]) --> B[Initialize vehicles and environment]
    B --> C{More timesteps\nin week?}
  
    C -->|Yes| D[Start timestep t]
    D --> E{Loop over\neach vehicle}
  
    E -->|For each vehicle| F{Update state:\ndriving or parked?}
    F -->|Driving| G[Update SoC for driving]
    F -->|Parked| H{Decide action}
  
    H -->|Charge| I[Charging]
    H -->|V2G discharge| J[V2G discharging]
    H -->|Idle| K[No action]
  
    I --> L[Request charger from pool]
    J --> L
    L --> M[Update vehicle SoC]
  
    G --> N[Continue to next vehicle]
    K --> N
    M --> N
  
    N --> E
  
    E -->|All vehicles processed| O[Aggregate P_charge_t and P_v2g_t]
    O --> P[Calculate P_net_t]
    P --> Q[Check grid overload if needed]
    Q --> R[Complete timestep t]
    R --> C
  
    C -->|No, week complete| S[End-of-week calculations]
    S --> T[Calculate SoH degradation\nfor each vehicle]
    T --> U([End: return weekly_results])
```

#### 3.5. `scenarios.py` - Monte-Carlo and Scenario Management

This module wraps the `Simulator` to perform the Monte-Carlo analysis and manage sweeps across different scenarios. It is the primary entry point for users.

```mermaid

graph TD
    A([Start: run_sweep]) --> B[Load scenario configurations]
    B --> C{More scenarios\nto process?}
  
    C -->|Yes| D[Get next scenario config]
    D --> E[Initialize empty list for run results]
    E --> F{Run counter < n_runs?}
  
    F -->|Yes| G[Run simulator.run_single_iteration]
    G --> H[Append run result to list]
    H --> I[Increment run counter]
    I --> F
  
    F -->|No, MC loop complete| J[Aggregate results for scenario]
    J --> K[Store aggregated scenario result]
    K --> C
  
    C -->|No, all scenarios complete| L([End: return all scenario results])
```
