from typing import List

class ChargerPool:
    """
    Represents a pool of chargers.
    """
    def __init__(self, num_chargers: int, power_kw: float):
        self.num_chargers = num_chargers
        self.power_kw = power_kw
        self.available_chargers = num_chargers

    def request_charger(self) -> bool:
        """
        Requests a charger from the pool.
        Returns True if a charger is available, False otherwise.
        """
        if self.available_chargers > 0:
            self.available_chargers -= 1
            return True
        return False

    def release_charger(self):
        """
        Releases a charger back to the pool.
        """
        if self.available_chargers < self.num_chargers:
            self.available_chargers += 1

class Grid:
    """
    Represents the electrical grid.
    """
    def __init__(self, max_feeder_power_mw: float, base_load_profile_mw: List[float]):
        self.max_feeder_power_mw = max_feeder_power_mw
        self.base_load_profile_mw = base_load_profile_mw

    def get_current_base_load(self, timestep: int) -> float:
        """
        Returns the base load of the grid at a given timestep.
        """
        return self.base_load_profile_mw[timestep]

    def check_overload(self, net_load_mw: float) -> bool:
        """
        Checks if the grid is overloaded.
        """
        return net_load_mw > self.max_feeder_power_mw
