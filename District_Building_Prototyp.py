# ================================================================
# Python libs
# ================================================================
import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
import yaml


# ================================================================
# Global settings
# ================================================================
TIMESTEP_HOURS = 0.25
START_YEAR = 2025
END_YEAR = 2045

# Direct emission factor natural gas [kg CO2 / kWh]
EF_GAS = 0.201


#================================================================
# The scenario settings consist of three levels:
# Level 1: Interaction Level
# Level 2: Heating Technology Path
# Level 3: Electricity Options
#================================================================
class InteractionLevel(Enum):

    BUILDING_ONLY = 1
    DISTRICT_ELECTRICITY = 2
    DISTRICT_HEAT_AND_POWER = 3

#================================================================
# Simulationtype szeanrios
#================================================================
class SimulationType(Enum):

    DIRECT_HEAT = "DirectHeat"
    HEAT_LED = "HeatLed"
    OPT_SELF_CONSUMPTION_BUILDING = "Optimization_Self-Consumption_Buildings_Level"
    OPT_SELF_CONSUMPTION_DISTRICT = "Optimization_Self-Consumption_district_Level"

# ================================================================
# Central Dataclasses to describe Building and District
# ================================================================

@dataclass
class BuildingMetadata:
    """
    Metadaten of the building, important to fill the class BuildingEnergyData
    """
    id: np.ndarray
    building_type: np.ndarray
    efficiency_class: np.ndarray
    renovation_status: np.ndarray
    number_of_units: np.ndarray
    current_system_type: np.ndarray
    remaining_lifetime: np.ndarray
    pv_enabled: np.ndarray
    pv_peak: np.ndarray

@dataclass
class BuildingEnergyData:
    """
    Building Energy Data
    """

    heat_demand: np.ndarray
    electricity_demand: np.ndarray
    pv_generation: np.ndarray

@dataclass
class BuildingResults:
    """
    KPIs: Each building is considered individually here. There is no interaction between them within a neighborhood.
    """
    self_consumption: np.ndarray
    grid_import: np.ndarray
    grid_export: np.ndarray
    PAR: np.ndarray #

@dataclass
class BuildingsData:
    """
    using structs for each single building
    """
    metadata: BuildingMetadata
    energy: Optional[BuildingEnergyData]= None
    building_index: dict[str, int] = field(
        default_factory=dict
    )


@dataclass
class DistrictMetadata:
    id: np.ndarray
    Building_count: int


@dataclass
class DistrictEnergyData:

    heat_demand: np.ndarray
    electricity_demand: np.ndarray
    pv_generation: np.ndarray

@dataclass
class DistrictResults:
    """
    KPIs: At this level, interactions between buildings within a district are allowed, and the results are stored.
    """
    local_grid_import: np.ndarray
    local_grid_export: np.ndarray
    local_self_consumption: np.ndarray

@dataclass
class ScenarioResults:
    simulation_type: str
    building_results: BuildingResults
    district_results: DistrictResults

@dataclass
class DistrictSimulationData:
    """
    Using Struct for a District
    """
    metadata: DistrictMetadata
    buildings: BuildingsData
    energy: Optional[DistrictEnergyData] = None
    scenarios: dict[str, ScenarioResults] = field(default_factory=dict)

# ================================================================
# END: Central Dataclasses to describe Building and District
# ================================================================

#=================================================================
# District Configuration Metadata and Energy profiles for Buildings
#=================================================================

class DistrictRepository:

    @staticmethod
    def load(filepath: str) -> DistrictSimulationData:

        with open(
                filepath,
                "r",
                encoding="utf-8"
        ) as stream:

            data = yaml.safe_load(stream)

        district_data = data["district"]
        buildings = district_data["buildings"]

        building_metadata = BuildingMetadata(

            id=np.array([
                b["id"]
                for b in buildings
            ]),

            building_type=np.array([
                b["building_type"]
                for b in buildings
            ]),

            efficiency_class=np.array([
                b["efficiency_class"]
                for b in buildings
            ]),

            renovation_status=np.array([
                b["renovation_status"]
                for b in buildings
            ]),

            number_of_units=np.array([
                b["number_of_units"]
                for b in buildings
            ]),

            current_system_type=np.array([
                b["current_system_type"]["type"]
                for b in buildings
            ]),

            remaining_lifetime=np.array([
                b["current_system_type"]["remaining_lifetime"]
                for b in buildings
            ]),

            pv_enabled=np.array([
                b["pv_system"]["enabled"]
                for b in buildings
            ]),

            pv_peak=np.array([
                b["pv_system"]["p_peak"]
                for b in buildings
            ])
        )

        district_metadata = DistrictMetadata(
            id=district_data["id"],
            Building_count=len(buildings)
        )

        return DistrictSimulationData(
            metadata=district_metadata,
            buildings=BuildingsData(
                metadata=building_metadata
            )
        )

    @staticmethod
    def show_metadata(district: DistrictSimulationData):
        metadata = district.buildings.metadata

        df = pd.DataFrame({
            "ID": metadata.id,
            "Type": metadata.building_type,
            "Efficiency": metadata.efficiency_class,
            "Renovation": metadata.renovation_status,
            "Units": metadata.number_of_units,
            "Heating": metadata.current_system_type,
            "Remaining Lifetime": metadata.remaining_lifetime,
            "PV Enabled": metadata.pv_enabled,
            "PV Peak [kWp]": metadata.pv_peak
        })

        print("\n" + "=" * 80)
        print(f"District: {district.metadata.id}")
        print(f"Buildings: {district.metadata.Building_count}")
        print("=" * 80)

        print(df.to_string(index=False))

    @staticmethod
    def show_energy_profiles(
            district: DistrictSimulationData,
            rows: int = 25):
        """
        Show first rows of all energy profiles.

        Parameters
        ----------
        rows : int
            Number of timesteps shown.
        """

        if district.buildings.energy is None:
            print(
                f"District {district.metadata.id}: "
                f"No energy profiles loaded."
            )
            return

        energy = district.buildings.energy

        building_ids = (
            district.buildings.metadata.id
        )

        profiles = {
            "Heat Demand [kW]": energy.heat_demand,

            "Electricity Demand [kW]": energy.electricity_demand,

            "PV Generation [kW]": energy.pv_generation
        }

        print("\n")
        print("=" * 80)
        print(
            f"District: "
            f"{district.metadata.id}"
        )
        print("=" * 80)

        for title, matrix in profiles.items():
            print("\n")
            print("-" * 80)
            print(title)
            print("-" * 80)

            df = pd.DataFrame(
                matrix[:rows, :],
                columns=building_ids
            )

            df["District Total"] = (
                matrix[:rows, :]
                .sum(axis=1)
            )


            print(df.to_string())

#=================================================================
# End: District Configuration Metadata and Energy profiles for Buildings
#=================================================================
#=================================================================
# Services
#=================================================================
class BuildingEnergyProfileService:
    HEADER_PEL_DEMAND = "Pel_Demand"
    HEADER_PTH_SPACE_HEATING = "P_th_space_heating"
    HEADER_PEL_PV = "Pel_PV"

    def __init__(self, district: DistrictSimulationData, data_path="data"):
        self.Pel_Demand = pd.read_csv(
            f"{data_path}/MFH_Electricity.csv",
            sep=";"
        )

        self.P_th_space_heating = pd.read_csv(
            f"{data_path}/MFH_Heat.csv",
            sep=";"
        )

        self.Pel_PV = pd.read_csv(
            f"{data_path}/MFH_PV.csv",
            sep=";"
        )

    def load_energy_profiles(self, district: DistrictSimulationData):

        metadata = district.buildings.metadata

        heat_profiles = []
        electricity_profiles = []
        pv_profiles = []

        for i in range(district.metadata.Building_count):

            column_name = (
                f"{metadata.building_type[i]}_"
                f"{metadata.efficiency_class[i]}_"
                f"{metadata.renovation_status[i]}"
            )

            heat_profiles.append(self.P_th_space_heating[column_name].to_numpy())
            electricity_profiles.append(self.Pel_Demand[column_name].to_numpy())
            pv_profiles.append(self.Pel_PV[column_name].to_numpy())

        district.buildings.energy = (
            BuildingEnergyData(

                heat_demand=np.column_stack(
                    heat_profiles
                ),

                electricity_demand=np.column_stack(
                    electricity_profiles
                ),

                pv_generation=np.column_stack(
                    pv_profiles
                )
            )
        )
        district.energy = DistrictEnergyData(

            heat_demand=
            district.buildings.energy
            .heat_demand.sum(axis=1),

            electricity_demand=
            district.buildings.energy
            .electricity_demand.sum(axis=1),

            pv_generation=
            district.buildings.energy
            .pv_generation.sum(axis=1)
        )


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':
    district = DistrictRepository.load(
        "config/district_test.yml"
    )

    DistrictRepository.show_metadata(
        district
    )

    energydata_service = (
        BuildingEnergyProfileService(district)
    )

    energydata_service.load_energy_profiles(
        district
    )

    DistrictRepository.show_energy_profiles(
        district,
        rows=20
    )


