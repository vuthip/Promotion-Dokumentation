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


# ================================================================
# Scenario settings
# ================================================================
@dataclass
class SimulationScenario:
    # PV considered?
    use_pv: bool = False
    # CO2 costs considered?
    use_co2_costs: bool = True
    # Existing gas boiler replacement?
    use_transformation: bool = True

    # -----------------------------------------
    # Thermal Storage
    # -----------------------------------------

    use_thermal_storage: bool = False
    thermal_storage_capacity_kWh: float = 0.0
    thermal_storage_charge_power_kW: float = 0.0
    thermal_storage_discharge_power_kW: float = 0.0
    thermal_storage_efficiency: float = 0.95

    # Hysteresis
    thermal_storage_soc_min: float = 0.20
    thermal_storage_soc_max: float = 0.80

    # Standing losses [kW]
    thermal_storage_loss_power_kW: float = 0.01



# ================================================================
# PV System
# ================================================================

@dataclass
class PVSystem:
    # Installed PV power [kWp]
    P_peak: float = 0.0
    # Investment costs [EUR]
    CAPEX: float = 0.0
    # Lifetime [a]
    lifetime: int = 25
    # Maintenance rate [-]
    maintenance_rate: float = 0.01
    # Annual generation [kWh/a]
    Wel_annual_generation_kWh_a: float = 0.0
    # Results
    CAPEX_ann: float = 0.0
    LCOE: float = 0.0


# ================================================================
# Heating System Types
# ================================================================
class HeatingSystemType(Enum):
    ASHP = "Air_Source_Heat_Pump"
    GSHP = "Ground_Source_Heat_Pump"
    GB = "Gas_Boiler"
    CHP = "Combined_Heat_and_Power"
    DH = "District_Heating"



# ================================================================
# Data Models
# ================================================================

@dataclass
class ComponentResult:
    """
    Results for auxiliary system components,
    e.g. electric heating rod.
    """
    # Power [W]
    P_th_max: float = 0.0
    P_el_max: float = 0.0

    # Annual energy [Wh/a]
    W_th: float = 0.0
    W_el: float = 0.0

    # Efficiency [-]
    eta: Optional[float] = None


@dataclass
class ThermalStorage:

    capacity_kWh: float = 0.0
    soc_kWh: float = 0.0
    charge_power_kW: float = 0.0
    discharge_power_kW: float = 0.0
    eta_storage: float = 0.95
    soc_min: float = 0.20
    soc_max: float = 0.80
    loss_power_kW: float = 0.0

@dataclass
class SystemState:
    """
    State of the heating system during the simulation.
    """
    is_existing: bool = False
    # Remaining lifetime at simulation start [a]
    remaining_lifetime: int = 0
    # Year of actual or potential investment
    investment_year: Optional[int] = None


@dataclass
class TechnicalParameters:
    """
    Technical input parameters.
    """
    COP: Optional[float] = None
    JAZ: Optional[float] = None
    eta: Optional[float] = None


@dataclass
class TechnicalResult:
    """
    Technical parameters and simulation results
    of a heating system.
    """

    # Installed power [W]
    P_th_max: float = 0.0
    P_el_max: float = 0.0
    P_aux_max: float = 0.0

    # Annual energy [Wh/a]
    W_th: float = 0.0
    W_el: float = 0.0
    W_gas: float = 0.0

    # Electricity balance [Wh/a]
    W_el_grid: float = 0.0
    W_el_PV_self: float = 0.0
    W_el_PV_export: float = 0.0

    # Time series
    operation: Optional[pd.DataFrame] = None

    # Auxiliary system
    auxiliary: Optional[ComponentResult] = None

    generator_on_profile: Optional[pd.Series] = None
    thermal_storage_soc_profile_kWh: Optional[pd.Series] = None
    thermal_storage_charge_profile: Optional[pd.Series] = None
    thermal_storage_discharge_profile: Optional[pd.Series] = None

    thermal_storage_result: Optional[ThermalStorageResult] = None

    generator_power_profile_W: Optional[pd.Series] = None

    generator_heat_supplied_profile_W: Optional[pd.Series] = None

@dataclass
class ThermalStorageResult:

    W_charge_kWh: float = 0.0
    W_discharge_kWh: float = 0.0
    storage_cycles: float = 0.0
    soc_profile_kWh: Optional[pd.Series] = None
    generator_on_profile: Optional[pd.Series] = None

@dataclass
class OperationResult:
    strategy_name: str = ""

    # Time series results

    P_th_generator: Optional[pd.Series] = None
    P_el_generator: Optional[pd.Series] = None

    P_th_auxiliary: Optional[pd.Series] = None
    P_el_auxiliary: Optional[pd.Series] = None

    thermal_storage_soc: Optional[pd.Series] = None
    thermal_storage_charge_soc: Optional[pd.Series] = None
    thermal_storage_discharge_soc: Optional[pd.Series] = None

    P_th_storage_charge: Optional[pd.Series] = None
    P_th_storage_discharge: Optional[pd.Series] = None

    generator_on: Optional[pd.Series] = None

    # Energy
    W_th_generator: float = 0.0
    W_el_generator: float = 0.0

    W_th_auxiliary: float = 0.0
    W_el_auxiliary: float = 0.0

    W_th_total: float = 0.0
    W_el_total: float = 0.0

    W_gas_generator: float = 0.0

    # PV
    W_el_grid: float = 0.0
    W_el_pv_self: float = 0.0
    W_el_export: float = 0.0

    # Environmental: CO2 Emissions
    CO2_emissions: float = 0.0
    CO2_specific: float = 0.0
    co2_emissions_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )
    co2_specific_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )


    # Economic history
    energy_price_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    co2_price_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    opex_energy_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    opex_co2_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    opex_maint_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    opex_total_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    c_ann_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    lcoh_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    capex_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    capex_ann_history: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    # Transformation path
    transformed: bool = False

    path_history: pd.Series = field(
        default_factory=lambda:
        pd.Series(dtype=object)
    )


@dataclass
class InvestmentCostCurve:
    """
    Time-dependent investment cost function:

    CAPEX_plant = a * P_th_max[kW] ** b
    """

    years: tuple[int, ...]
    a_values: tuple[float, ...]
    b_values: tuple[float, ...]

    # Building conversion costs [EUR/unit]
    conversion_cost_per_unit: float = 0.0

    # Additional fixed investment costs [EUR]
    additional_cost: float = 0.0

    def __post_init__(self):

        if not (
                len(self.years)
                == len(self.a_values)
                == len(self.b_values)
        ):
            raise ValueError(
                "years, a_values and b_values "
                "must have the same length."
            )

        if tuple(sorted(self.years)) != self.years:
            raise ValueError(
                "years must be sorted in ascending order."
            )


@dataclass
class EnvironmentalResult:
    """
    Environmental results.
    """
    # Emissions [kg CO2/a]
    CO2_emissions: float = 0.0

    # Specific emissions [kg CO2/kWh_th]
    CO2_specific: float = 0.0


@dataclass
class EconomicResult:
    """
    Economic parameters and results.
    """

    # Investment Cost Model [EUR]
    # Heating system itself
    CAPEX_plant: float = 0.0
    # Building conversion
    CAPEX_conversion: float = 0.0
    # Additional investment costs
    CAPEX_additional: float = 0.0
    # invest thermal_storage cost
    CAPEX_storage: float = 0.0
    # Total Investment costs [EUR]
    CAPEX: float = 0.0
    # Annualized capital costs [EUR/a]
    CAPEX_ann: float = 0.0

    # Lifetime [a]
    lifetime: int = 20

    # Maintenance
    # Reference investment for maintenance calculation [EUR]
    maintenance_reference_cost: float = 0.0
    # Rate [-]
    maintenance_rate: float = 0.0

    # Operating costs [EUR/a]
    OPEX_energy: float = 0.0
    # Maintanance costs [EUR/a]
    OPEX_maint: float = 0.0
    # CO2 cost of the operation [EUR/a]
    OPEX_CO2: float = 0.0
    # Operating cost in total [EUR/a]
    OPEX_total: float = 0.0

    # Total annual costs [EUR/a]
    C_ann: float = 0.0

    # Levelized Cost of Heat [EUR/kWh_th]
    LCOH: float = 0.0


@dataclass
class HeatingSystem:
    system_type: HeatingSystemType
    name: str

    state: SystemState = field(
        default_factory=SystemState
    )

    parameters: TechnicalParameters = field(
        default_factory=TechnicalParameters
    )

    technical: TechnicalResult = field(
        default_factory=TechnicalResult
    )

    investment: Optional[InvestmentCostCurve] = None

    environmental: EnvironmentalResult = field(
        default_factory=EnvironmentalResult
    )

    economic: EconomicResult = field(
        default_factory=EconomicResult
    )
    thermal_storage: Optional[ThermalStorage] = None

    operation_results: dict[str, OperationResult] = field(
        default_factory=dict
    )
    transformation_results: dict[str, TransformationResult] = field(
        default_factory=dict
    )



class TransformationDecision:
    @staticmethod
    def decide_by_strategy(
            building,
            strategy_name,
            year
    ):
        candidates = []

        for system in (
                building.heating_systems.values()
        ):
            result = (
                system.operation_results[
                    strategy_name
                ]
            )

            lcoh = (
                result.lcoh_history.loc[
                    year
                ]
            )

            candidates.append(
                (
                    system,
                    lcoh
                )
            )

        selected_system = min(
            candidates,
            key=lambda x: x[1]
        )[0]

        return selected_system



# ================================================================
# Building
# ================================================================
@dataclass
class Building:
    """
    Representation of one building including energy profile
    and evaluated heating systems.
    """

    id: str

    building_type: str
    efficiency_class: str
    renovation_status: str

    number_of_units: int = 1

    energy_profile: Optional[pd.DataFrame] = None

    pv_system: Optional[PVSystem] = None

    heating_systems: dict[
        HeatingSystemType,
        HeatingSystem
    ] = field(default_factory=dict)

    # Actually installed heating system
    installed_system: Optional[HeatingSystem] = None

    # Simulation history:

    # ------------------------------------------------------------
    # Simulation histories
    # ------------------------------------------------------------

    # All potential technology paths
    technology_history: Optional[pd.DataFrame] = None

    # Actual decision events
    decision_history: Optional[pd.DataFrame] = None

    # Actually realized path
    realized_history: Optional[pd.DataFrame] = None
    # Current building state:
    # Currently installed technology
    current_system_type: Optional[
        HeatingSystemType
    ] = None

    # Was a decision already made?
    decision_year: Optional[int] = None
    # Transformation status
    is_transformed: bool = False
    # First transformation year
    transformation_year: Optional[int] = None

    # Heating System Management
    def add_heating_system(self, system: HeatingSystem):
        self.heating_systems[system.system_type] = system

    def get_heating_system(self, system_type: HeatingSystemType) -> HeatingSystem:
        return self.heating_systems[system_type]

    def get_economic_comparison(self):

        rows = []

        for system in self.heating_systems.values():
            rows.append({
                "Building_ID": self.id,
                "System": system.name,
                "Type": system.system_type.name,
                "P_th_max_kW": system.technical.P_th_max / 1000,
                "P_el_max [kW]": system.technical.P_el_max / 1000,
                "W_th [kWh/a]": system.technical.W_th / 1000,
                "W_el [kWh/a]": system.technical.W_el / 1000,
                "W_gas [kWh/a]": system.technical.W_gas / 1000,
                "JAZ": system.parameters.JAZ,
                "eta": system.parameters.eta,
                "CAPEX plant [EUR]": system.economic.CAPEX_plant,
                "CAPEX conversion [EUR]": system.economic.CAPEX_conversion,
                "CAPEX additional [EUR]": system.economic.CAPEX_additional,
                "CAPEX total [EUR]": system.economic.CAPEX,
                "CAPEX_ann [EUR/a]": system.economic.CAPEX_ann,
                "OPEX_energy [EUR/a]": system.economic.OPEX_energy,
                "OPEX_maint [EUR/a]": system.economic.OPEX_maint,
                "OPEX_CO2 [EUR/a]": system.economic.OPEX_CO2,
                "OPEX_total [EUR/a]": system.economic.OPEX_total,
                "CO2 [kg/a]": system.environmental.CO2_emissions,
                "CO2 specific [kg/kWh_th]": system.environmental.CO2_specific,
                "C_ann [EUR/a]": system.economic.C_ann,
                "LCOH [EUR/kWh]": system.economic.LCOH
            })

        comparison_df = pd.DataFrame(rows)

        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values(by="LCOH [EUR/kWh]", ascending=True).reset_index(drop=True)

        return comparison_df

    def show_heating_systems_table(self):
        """
        Displays all heating systems of the building
        as a table.
        """
        systems_df = self.get_economic_comparison()
        if systems_df.empty:
            print(
                f"No heating systems available "
                f"for building {self.id}."
            )
            return systems_df

        print()
        print("=" * 160)
        print(f"Heating Systems - Building {self.id}")
        print("=" * 160)

        print(systems_df.to_string(index=False))

        return systems_df

    def plot_economic_development(self):

        if self.technology_history is None:
            raise ValueError(
                "No economic history available."
            )

        technology_history_df = self.technology_history.copy()

        if technology_history_df.empty:
            raise ValueError(
                "Economic history is empty."
            )

        systems = technology_history_df["System"].unique()

        years = sorted(technology_history_df["Year"].unique())

        system_colors = {
            HeatingSystemType.ASHP.name:
                "tab:blue",

            HeatingSystemType.GSHP.name:
                "tab:green",

            HeatingSystemType.GB.name:
                "tab:gray"
        }
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        # ========================================================
        # Subplot 1: Annual Costs
        # ========================================================
        ax1 = axes[0]

        n_systems = len(systems)
        bar_width = 0.25

        x = list(range(len(years)))

        for system_index, system_name in enumerate(systems):

            system_data = (
                technology_history_df[
                    technology_history_df["System"]
                    == system_name
                    ]
                .sort_values("Year")
            )

            if system_data.empty:
                continue

            system_type = (
                system_data["Type"]
                .iloc[0]
            )

            color = system_colors.get(
                system_type,
                "tab:gray"
            )

            x_position = [
                value
                + (
                        system_index
                        - (n_systems - 1) / 2
                )
                * bar_width
                for value in x
            ]

            capex = (
                system_data["CAPEX_ann"]
                .to_numpy()
            )

            opex_energy = (
                system_data["OPEX_energy"]
                .to_numpy()
            )

            opex_maint = (
                system_data["OPEX_maint"]
                .to_numpy()
            )

            opex_co2 = (
                system_data["OPEX_CO2"]
                .to_numpy()
            )

            # CAPEX
            ax1.bar(
                x_position,
                capex,
                width=bar_width,
                color=color,
                alpha=0.40,
                hatch="//",
                edgecolor="black",
                linewidth=0.5,
                zorder=3
            )

            # Energy OPEX
            ax1.bar(
                x_position,
                opex_energy,
                width=bar_width,
                bottom=capex,
                color=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
                zorder=3
            )

            # Maintenance OPEX
            ax1.bar(
                x_position,
                opex_maint,
                width=bar_width,
                bottom=capex + opex_energy,
                color=color,
                alpha=0.55,
                hatch="..",
                edgecolor="black",
                linewidth=0.5,
                zorder=3
            )

            # CO2 OPEX
            ax1.bar(
                x_position,
                opex_co2,
                width=bar_width,
                bottom=(
                        capex
                        + opex_energy
                        + opex_maint
                ),
                color=color,
                alpha=0.70,
                hatch="xx",
                edgecolor="black",
                linewidth=0.5,
                zorder=3
            )

        ax1.set_xticks(x)
        ax1.set_xticklabels(years, rotation=45)
        ax1.set_ylabel("Annual Costs [EUR/a]")
        ax1.set_title("Development of Annual Heating Costs")
        ax1.set_axisbelow(True)
        ax1.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=0.7,
            alpha=0.5
        )
        # --------------------------------------------------------
        # Legends
        # --------------------------------------------------------
        from matplotlib.patches import Patch
        technology_handles = [
            Patch(
                facecolor=color,
                label=system_type
            )
            for system_type, color
            in system_colors.items()
        ]
        technology_legend = ax1.legend(
            handles=technology_handles,
            title="Technology",
            loc="upper left"
        )

        ax1.add_artist(
            technology_legend
        )
        cost_handles = [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="//",
                label="Annualized CAPEX"
            ),

            Patch(
                facecolor="gray",
                edgecolor="black",
                label="Energy OPEX"
            ),

            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="..",
                label="Maintenance OPEX"
            ),
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch="xx",
                label="CO2 OPEX"
            )
        ]
        ax1.legend(
            handles=cost_handles,
            title="Cost Component",
            loc="upper right"
        )
        # ========================================================
        # Subplot 2: LCOH
        # ========================================================
        ax2 = axes[1]

        for system_name in systems:

            system_data = (
                technology_history_df[
                    technology_history_df["System"]
                    == system_name
                    ]
                .sort_values("Year")
            )

            if system_data.empty:
                continue

            system_type = (
                system_data["Type"]
                .iloc[0]
            )
            color = system_colors.get(
                system_type,
                "tab:gray"
            )
            ax2.plot(
                system_data["Year"],
                system_data["LCOH"],
                marker="o",
                markersize=4,
                linewidth=2,
                color=color,
                label=system_name,
                zorder=3
            )
        ax2.set_xlabel("Year")
        ax2.set_ylabel("LCOH [EUR/kWh_th]")
        ax2.set_title("Development of Levelized Cost of Heat")
        ax2.set_xticks(years)

        ax2.tick_params(
            axis="x",
            rotation=45
        )

        ax2.set_axisbelow(True)

        ax2.grid(
            True,
            axis="both",
            linestyle="--",
            linewidth=0.7,
            alpha=0.5
        )
        ax2.legend(
            title="Technology",
            loc="best"
        )

        fig.suptitle(
            f"Economic Development - Building {self.id}",
            fontsize=16
        )

        plt.tight_layout()
        plt.show()

    def plot_heating_operation(
            self,
            system: HeatingSystem
    ):

        if (
                system.technical.generator_on_profile
                is None
        ):
            raise ValueError(
                "No heating operation simulation available."
            )

        soc = (
            system.technical
            .thermal_storage_soc_profile_kWh
        )

        generator_on = (
            system.technical
            .generator_on_profile
        )

        charge = (
            system.technical
            .thermal_storage_charge_profile
        )

        discharge = (
            system.technical
            .thermal_storage_discharge_profile
        )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(16, 10),
            sharex=True
        )

        # ============================================================
        # Plot 1
        # ============================================================

        ax1 = axes[0]

        ax1.plot(
            soc,
            color="tab:green",
            label="SOC"
        )

        if system.thermal_storage is not None:
            ax1.axhline(
                y=
                system.thermal_storage.capacity_kWh
                * system.thermal_storage.soc_min,

                color="tab:red",

                linestyle="--",

                label="SOC Min"
            )

            ax1.axhline(
                y=
                system.thermal_storage.capacity_kWh
                * system.thermal_storage.soc_max,

                color="tab:blue",

                linestyle="--",

                label="SOC Max"
            )

        ax1.set_ylabel(
            "SOC [kWh]"
        )

        ax1.set_title(
            f"Hysteresis Control - {system.name}"
        )

        ax1.grid(True)

        ax1.legend()

        # ============================================================
        # Plot 2
        # ============================================================

        ax2 = axes[1]

        ax2.step(
            generator_on.index,
            generator_on,
            where="post",
            color="black",
            linewidth=1.5,
            label="Generator ON"
        )

        ax2.set_ylabel(
            "ON / OFF"
        )

        ax2.set_ylim(
            -0.1,
            1.1
        )

        ax2.grid(True)

        ax2.legend()

        # ============================================================
        # Plot 3
        # ============================================================

        ax3 = axes[2]

        ax3.fill_between(
            charge.index,
            charge,
            color="tab:green",
            alpha=0.4,
            label="Charge"
        )

        ax3.fill_between(
            discharge.index,
            discharge,
            color="tab:red",
            alpha=0.4,
            label="Discharge"
        )

        ax3.set_ylabel(
            "Energy [kWh/timestep]"
        )

        ax3.set_xlabel(
            "Timestep"
        )

        ax3.grid(True)

        ax3.legend()

        plt.tight_layout()

        plt.show()

class OperationStrategy:

    def simulate(self, building, system, heat_demand) -> OperationResult:
        raise NotImplementedError(
            "OperationStrategy is abstract."
        )

    @staticmethod
    def plot_direct_heat(
            result: OperationResult,
            heat_demand,
            title="Direct Heat Control",
            start=None,
            end=None
    ):

        if start is None:
            start = 0

        if end is None:
            end = len(heat_demand)

        demand = heat_demand.iloc[start:end]

        generator = (
            result.P_th_generator.iloc[start:end]
        )

        auxiliary = (
            result.P_th_auxiliary.iloc[start:end]
        )

        fig, ax = plt.subplots(
            figsize=(14, 6)
        )

        ax.plot(
            demand.values,
            color="black",
            linewidth=2,
            label="Heat Demand"
        )

        ax.stackplot(
            range(len(demand)),
            generator.values,
            auxiliary.values,
            labels=[
                "Generator",
                "Auxiliary"
            ],
            alpha=0.7
        )

        ax.set_title(title)

        ax.set_ylabel(
            "Thermal Power [W]"
        )

        ax.set_xlabel(
            "Timestep"
        )

        ax.grid(True)

        ax.legend()

        plt.tight_layout()

        plt.show()

    @staticmethod
    def plot_heat_led(
            result: OperationResult,
            heat_demand,
            storage,
            title="Heat Led Storage Control",
            start=None,
            end=None
    ):

        if start is None:
            start = 0

        if end is None:
            end = len(heat_demand)

        demand = heat_demand

        generator = (
            result.P_th_generator
        )

        auxiliary = (
            result.P_th_auxiliary
        )

        soc = (
            result.thermal_storage_soc
        )

        balance = (
                generator
                +
                auxiliary
                -
                demand
        )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(16, 10),
            sharex=True
        )

        # -----------------------------------
        # Plot 1
        # -----------------------------------

        axes[0].plot(
            demand.values,
            color="black",
            linewidth=2,
            label="Heat Demand"
        )

        axes[0].stackplot(
            range(len(demand)),
            generator.values,
            auxiliary.values,
            labels=[
                "Generator",
                "Auxiliary"
            ],
            alpha=0.7
        )

        axes[0].set_ylabel(
            "Thermal Power [W]"
        )

        axes[0].legend()

        axes[0].grid(True)

        # -----------------------------------
        # Plot 2
        # -----------------------------------

        axes[1].fill_between(
            balance.index,
            balance,
            0,
            alpha=0.5
        )

        axes[1].axhline(
            0,
            color="black"
        )

        axes[1].set_ylabel(
            "Generation - Demand [W]"
        )

        axes[1].grid(True)

        # -----------------------------------
        # Plot 3
        # -----------------------------------

        axes[2].plot(
            soc.values,
            color="tab:green",
            linewidth=2,
            label="SOC"
        )

        axes[2].axhline(
            storage.capacity_kWh
            * storage.soc_min,
            color="red",
            linestyle="--",
            label="SOC Min"
        )

        axes[2].axhline(
            storage.capacity_kWh
            * storage.soc_max,
            color="blue",
            linestyle="--",
            label="SOC Max"
        )

        axes[2].set_ylabel(
            "SOC [kWh]"
        )

        axes[2].set_xlabel(
            "Timestep"
        )

        axes[2].legend()

        axes[2].grid(True)

        fig.suptitle(title)

        plt.tight_layout()

        plt.show()


class DirectHeatStrategy(OperationStrategy):

    def simulate(
        self,
        building,
        system,
        heat_demand
    ) -> OperationResult:

        result = OperationResult()

        dt = TIMESTEP_HOURS

        # --------------------------------------------------
        # Thermische Leistung
        # --------------------------------------------------

        P_th_generator = []
        P_th_auxiliary = []

        for heat_load_W in heat_demand:

            if heat_load_W <= system.technical.P_th_max:

                P_th_generator.append(
                    heat_load_W
                )

                P_th_auxiliary.append(
                    0.0
                )

            else:

                P_th_generator.append(
                    system.technical.P_th_max
                )

                P_th_auxiliary.append(
                    heat_load_W
                    - system.technical.P_th_max
                )

        # --------------------------------------------------
        # In Pandas Series umwandeln
        # --------------------------------------------------

        P_th_generator = pd.Series(
            P_th_generator,
            index=heat_demand.index
        )

        P_th_auxiliary = pd.Series(
            P_th_auxiliary,
            index=heat_demand.index
        )

        # --------------------------------------------------
        # Elektrische Leistung
        # --------------------------------------------------

        if system.system_type in (
            HeatingSystemType.ASHP,
            HeatingSystemType.GSHP
        ):

            P_el_generator = (
                P_th_generator
                / system.parameters.JAZ
            )

            # Heizstab
            P_el_auxiliary = (
                P_th_auxiliary
            )

        elif system.system_type == (
            HeatingSystemType.GB
        ):

            P_el_generator = pd.Series(
                0.0,
                index=heat_demand.index
            )

            P_el_auxiliary = pd.Series(
                0.0,
                index=heat_demand.index
            )

        else:

            raise NotImplementedError(
                f"{system.system_type}"
            )


        # --------------------------------------------------
        # Generatorstatus
        # --------------------------------------------------

        generator_on = (
            P_th_generator > 0
        ).astype(int)

        # --------------------------------------------------
        # Ergebnisse speichern
        # --------------------------------------------------

        result.strategy_name = (
            "DirectHeat"
        )

        result.P_th_generator = (
            P_th_generator
        )

        result.P_el_generator = (
            P_el_generator
        )

        result.P_th_auxiliary = (
            P_th_auxiliary
        )

        result.P_el_auxiliary = (
            P_el_auxiliary
        )

        result.generator_on = (
            generator_on
        )

        result.thermal_storage_soc = (
            None
        )

        # --------------------------------------------------
        # Energiemengen
        # --------------------------------------------------

        result.W_th_generator = (
            P_th_generator.sum()
            * dt
            / 1000
        )

        result.W_el_generator = (
            P_el_generator.sum()
            * dt
            / 1000
        )

        result.W_th_auxiliary = (
            P_th_auxiliary.sum()
            * dt
            / 1000
        )

        result.W_el_auxiliary = (
            P_el_auxiliary.sum()
            * dt
            / 1000
        )

        result.W_th_total = (
            result.W_th_generator
            + result.W_th_auxiliary
        )

        result.W_el_total = (
            result.W_el_generator
            + result.W_el_auxiliary
        )

        if system.system_type == HeatingSystemType.GB:
            result.W_gas_generator = (
                    result.W_th_total
                    / system.parameters.eta
            )

        else:
            result.W_gas_generator = 0.0

        return result





class HeatLedStorageStrategy(OperationStrategy):

    def simulate(self, building, system, heat_demand) -> OperationResult:

        if system.thermal_storage is None:
            raise ValueError(
                "HeatLedStorageStrategy requires a thermal storage."
            )

        dt = TIMESTEP_HOURS

        result = OperationResult()

        storage = system.thermal_storage
        storage_charge = []
        storage_discharge = []
        storage_SOC = []

        generator_power_kW_max = system.technical.P_th_max / 1000

        soc = 0

        generator_on = False

        generator_on_profile = []

        soc_profile = []

        charge_profile = []

        discharge_profile = []

        p_th_generator_profile = []
        p_th_generator_aux_profile = []

        p_el_generator_profile = []

        W_th_total = 0.0

        W_el_total = 0.0


        for t, heat_load_W in enumerate(heat_demand):

            heat_load_kWh = heat_load_W * dt / 1000
            # Step 1: Determining the Previous Thermal Storage and Generator Status
            if t==0:
                # special case: First Timestep - Init start values
                SOC = 0
                generator_on = True
            else:
                SOC = soc_profile[t-1]
                generator_on = generator_on_profile[t-1]

                # Step 2: Hysteresis Control
                # Case A: thermal storage are low --> Generator turn on
                if soc_profile[t-1] <= (system.thermal_storage.soc_min * system.thermal_storage.capacity_kWh):
                    generator_on = True

                # Case B: thermal storage are high --> Generator turn off
                elif soc_profile[t-1] >= (system.thermal_storage.soc_max * system.thermal_storage.capacity_kWh):
                    generator_on = False
                # Case C: Genrator has the old status
                else:
                    generator_on = generator_on_profile[t - 1]

            generator_on_profile.append(generator_on)

            # Step 3: Calculation of Energy Flows and Storage Status
            if generator_on:
                p_th_generator_profile.append(system.technical.P_th_max)
                p_el_generator_profile.append(system.technical.P_el_max)
                W_th_generator_kWh = p_th_generator_profile[t]* dt /1000
            else:
                p_th_generator_profile.append(0)
                p_el_generator_profile.append(0)
                W_th_generator_kWh = 0

            if (SOC + W_th_generator_kWh) > heat_load_kWh:
                soc_kwh = (SOC + W_th_generator_kWh - heat_load_kWh)
                p_th_generator_aux_profile.append(0)
                W_el_aux_kWh = 0
            else:
                p_th_generator_aux_profile.append((heat_load_kWh - (SOC + W_th_generator_kWh)) * 1000 / dt)
                W_el_aux_kWh = p_th_generator_aux_profile[t] * dt / 1000

            soc_kwh = (SOC + W_th_generator_kWh + W_el_aux_kWh - heat_load_kWh)
            if (W_th_generator_kWh + W_el_aux_kWh - heat_load_kWh) >=0:
                storage_charge.append((W_th_generator_kWh + W_el_aux_kWh - heat_load_kWh)*1000/ dt )
                storage_discharge.append(0)
            else:
                storage_charge.append(0)
                storage_discharge.append(((W_th_generator_kWh + W_el_aux_kWh - heat_load_kWh)*-1) *1000/ dt )



            if t < 1000:
                print(
                    f"{t:4d}",
                    f"Demand={heat_load_kWh:.2f}",
                    f"Generator={p_th_generator_profile[t] * dt / 1000:.2f}",
                    f"Generator_aux={p_th_generator_aux_profile[t] * dt / 1000:.2f}",
                    f"SOC={soc_kwh}"
                )

            if soc_kwh > system.thermal_storage.capacity_kWh:
                soc_profile.append(system.thermal_storage.capacity_kWh)
            elif soc_kwh < 0:
                soc_profile.append(0)
            else:
                soc_profile.append(soc_kwh)

        # Store the results
        result.strategy_name = "HeatLed"
        result.P_th_generator = pd.Series(p_th_generator_profile)
        result.P_el_generator = pd.Series(p_el_generator_profile)
        result.P_th_auxiliary = pd.Series(p_th_generator_aux_profile)
        result.P_el_auxiliary = pd.Series(p_th_generator_aux_profile)
        result.thermal_storage_soc = pd.Series(soc_profile)
        result.P_th_storage_charge = pd.Series(storage_charge)
        result.P_th_storage_discharge = pd.Series(storage_discharge)
        result.generator_on = pd.Series(generator_on_profile)

        result.W_th_generator = (result.P_th_generator.sum() * dt / 1000)
        result.W_el_generator = (result.P_el_generator.sum() * dt / 1000)

        result.W_th_auxiliary = (result.P_th_auxiliary.sum() * dt / 1000)
        result.W_el_auxiliary = (result.P_th_auxiliary.sum() * dt / 1000)

        result.W_th_total = (result.W_th_generator + result.W_th_auxiliary)
        result.W_el_total = (result.W_el_generator + result.W_el_auxiliary)

        if system.system_type == HeatingSystemType.GB:
            result.W_gas_generator = (
                    result.W_th_total
                    / system.parameters.eta
            )

        else:
            result.W_gas_generator = 0.0

        return result



"""
        # ----------------------------------
        # Speicherverluste
        # ----------------------------------

        storage_loss_kWh = (
            storage.loss_power_kW
            * dt
        )

        soc = max(
            0.0,
            soc - storage_loss_kWh
        )

        # ----------------------------------
        # SOC bestimmen
        # ----------------------------------

        if storage.capacity_kWh > 0:

            soc_rel = (
                soc
                / storage.capacity_kWh
            )

        else:

            soc_rel = 0.0

        # ----------------------------------
        # Hysterese
        # ----------------------------------

        if soc_rel <= storage.soc_min:

            generator_on = True

        elif soc_rel >= storage.soc_max:

            generator_on = False

        # ----------------------------------
        # Speicher zuerst entladen
        # ----------------------------------

        discharge_kWh = min(
            heat_load_kWh,
            soc,
            storage.discharge_power_kW
            * dt
        )

        soc -= discharge_kWh

        remaining_load_kWh = (
            heat_load_kWh
            - discharge_kWh
        )

        # ----------------------------------
        # Wärmeerzeuger
        # ----------------------------------

        generator_energy_kWh = 0.0

        charge_kWh = 0.0

        heat_to_building_kWh = 0.0

        if generator_on:
            generator_energy_kWh = (
                    generator_power_kW_max
                    * dt
            )

            heat_to_building_kWh = min(
                generator_energy_kWh,
                remaining_load_kWh
            )

            excess_kWh = (
                    generator_energy_kWh
                    - heat_to_building_kWh
            )

            charge_kWh = min(
                excess_kWh,
                storage.charge_power_kW
                * dt,
                storage.capacity_kWh
                - soc
            )

            soc += (
                charge_kWh
                * storage.eta_storage
            )

        delivered_heat_kWh = (
            heat_to_building_kWh
            + discharge_kWh
        )

        # ----------------------------------
        # thermische Leistung
        # ----------------------------------

        P_th_generator_W = (
            generator_energy_kWh
            * 1000
            / dt
        )

        # ----------------------------------
        # elektrische Leistung
        # ----------------------------------

        if system.system_type in (
            HeatingSystemType.ASHP,
            HeatingSystemType.GSHP
        ):

            P_el_generator_W = (
                P_th_generator_W
                / system.parameters.JAZ
            )

        elif (
            system.system_type
            == HeatingSystemType.GB
        ):

            P_el_generator_W = 0.0

        else:

            P_el_generator_W = 0.0

        # ----------------------------------
        # Jahresenergiemengen
        # ----------------------------------

        W_th_total += (
            delivered_heat_kWh
            * 1000
        )

        W_el_total += (
            P_el_generator_W
            * dt
        )

        # ----------------------------------
        # Profile speichern
        # ----------------------------------

        generator_on_profile.append(
            int(generator_on)
        )

        soc_profile.append(
            soc
        )

        charge_profile.append(
            charge_kWh
        )

        discharge_profile.append(
            discharge_kWh
        )

        p_th_generator_profile.append(
            P_th_generator_W
        )

        p_el_generator_profile.append(
            P_el_generator_W
        )

        # ======================================
        # Ergebnisse schreiben
        # ======================================

        result.generator_on = pd.Series(
            generator_on_profile
        )

        result.storage_soc = pd.Series(
            soc_profile
        )

        result.P_storage_charge = pd.Series(
            charge_profile
        )

        result.P_storage_discharge = pd.Series(
            discharge_profile
        )

        result.P_th_generator = pd.Series(
            p_th_generator_profile
        )

        result.P_el_generator = pd.Series(
            p_el_generator_profile
        )

        result.W_th = W_th_total

        result.W_el = W_el_total

        return result
"""




# ================================================================
# Transformation Decision
# ================================================================
class TransformationDecision_old:
    """
    Selects a heating technology.
        Initial decision rule:
        technology with lowest LCOH.
    Later:
        CAPEX
        OwnerBehaviour
        Förderung
        Amortisation
        CO2
        Technologieaversion
        Kapitalverfügbarkeit
        Nahwärme verfügbar?
    """

    @staticmethod
    def decide(building: Building) -> HeatingSystem:
        if not building.heating_systems:
            raise ValueError(
                "Building contains no heating systems."
            )

        selected_system = min(building.heating_systems.values(), key=lambda system: system.economic.LCOH)
        return selected_system


# ================================================================
# Flexibility Model
# ================================================================
class FlexibilityModel:
    """
    Exaple function:
        calculate_load_shift_potential()
        calculate_peak_reduction()
        calculate_pv_matching()

    """
    pass


# ================================================================
# District Model
# ================================================================
@dataclass
class District:

    id: str
    buildings: list[Building]
    energy_profile: Optional[pd.DataFrame] = None
    district_result: Optional[DistrictResult] = None

    def add_building(self,building: Building):
        self.buildings.append(building)

@dataclass
class DistrictResult:

    district_id: str = ""

    # aggregate Profiles

    heat_demand_profile: Optional[pd.Series] = None
    electricity_demand_profile: Optional[pd.Series] = None
    pv_generation_profile: Optional[pd.Series] = None
    residual_load_profile: Optional[pd.Series] = None

    # Energy

    W_th_total: float = 0.0
    W_el_total: float = 0.0
    W_pv_total: float = 0.0

    # Grid
    W_grid_import: float = 0.0
    W_grid_export: float = 0.0

    # KPI
    peak_load: float = 0.0
    peak_residual_load: float = 0.0
    autarky_rate: float = 0.0
    self_consumption_rate: float = 0.0


class DistrictModel:

    @staticmethod
    def evaluate(district: District) -> DistrictResult:

        result = DistrictResult()
        result.district_id = district.id

        # ----------------------------------
        # Energy profiles
        # ----------------------------------

        heat_profiles = []
        electricity_profiles = []
        pv_profiles = []

        for building in district.buildings:
            profile = building.energy_profile

            heat_profiles.append(profile[BuildingEnergyProfileRepository.HEADER_PTH_SPACE_HEATING])
            electricity_profiles.append(profile[BuildingEnergyProfileRepository.HEADER_PEL_DEMAND])
            pv_profiles.append(profile[BuildingEnergyProfileRepository.HEADER_PEL_PV])

        # ----------------------------------
        # Aggregation
        # ----------------------------------

        result.heat_demand_profile = sum(heat_profiles)
        result.electricity_demand_profile = sum(electricity_profiles)
        result.pv_generation_profile = sum(pv_profiles)

        # ----------------------------------
        # Residualload
        # ----------------------------------

        result.residual_load_profile = (result.electricity_demand_profile - result.pv_generation_profile)

        # ----------------------------------
        # Energy
        # ----------------------------------

        result.W_th_total = (result.heat_demand_profile.sum() * TIMESTEP_HOURS / 1000)
        result.W_el_total = (result.electricity_demand_profile.sum() * TIMESTEP_HOURS / 1000)
        result.W_pv_total = (result.pv_generation_profile.sum() * TIMESTEP_HOURS / 1000)

        # ----------------------------------
        # Peak Loads
        # ----------------------------------

        result.peak_load = (result.electricity_demand_profile.max())
        result.peak_residual_load = (result.residual_load_profile.max())

        district.district_result = result
        return result



@dataclass
class OwnerProfile:
    attitude: str

    max_capex: Optional[float] = None

    environmental_weight: float = 0.0
    economic_weight: float = 1.0
    peer_effect_weight: float = 0.0


# ================================================================
# Energy Price Repository
# ================================================================

class EnergyPriceRepository:
    """
    Reads energy prices from an Excel file and provides
    the prices for a selected year.
    """

    # EnergyPriceRepository
    HEADER_GAS_PRICE = "Gas_Price"
    HEADER_ELECTRICITY_PRICE = "Electricity_Price"
    HEADER_CO2_PRICE = "CO2_Price"
    HEADER_EF_Electricity = "EF_Electricity"

    def __init__(self, filepath="data/Energy_prices.xlsx"):
        self.filepath = filepath
        self.data = pd.read_excel(self.filepath)

    def get_energy_prices(self, year):
        energy_prices_df = self.data
        energy_prices_row = energy_prices_df[energy_prices_df["Year"] == year]

        # Überrüfen ob Daten vorhanden sind, wenn nicht Fehlermeldung ausgeben
        if energy_prices_row.empty:
            raise ValueError(f"No energy prices found for year {year}.")

        row = energy_prices_row.iloc[0]

        print(row)

        return {
            self.HEADER_GAS_PRICE: row["Gas Price"],
            self.HEADER_ELECTRICITY_PRICE: row["Electrical Price"],
            self.HEADER_CO2_PRICE: row["CO2-Price"],
            self.HEADER_EF_Electricity: row["EF_Electricity"]
        }

    @classmethod
    def show(cls):
        print('Available field names for "EnergyPrice":')
        for name, value in vars(cls).items():
            if not name.startswith("_") and isinstance(value, str):
                print(f"{name:35} -> {value}")


class BuildingEnergyProfileRepository:
    HEADER_PEL_DEMAND = "Pel_Demand"
    HEADER_PTH_SPACE_HEATING = "P_th_space_heating"
    HEADER_PTH_DHW = "P_th_DHW"
    HEADER_PEL_PV = "Pel_PV"

    def __init__(self, data_path="data"):
        self.data_path = data_path

        # Files are read only once
        self.Pel_Demand = pd.read_csv(f"{data_path}/MFH_Electricity.csv", sep=";")
        self.P_th_space_heating = pd.read_csv(f"{data_path}/MFH_Heat.csv", sep=";")
        self.Pel_PV = pd.read_csv(f"{data_path}/MFH_PV.csv", sep=";")

    def get_energy_profile(self, building: Building):
        column_name = (
            f"{building.building_type}_"
            f"{building.efficiency_class}_"
            f"{building.renovation_status}"
        )

        datasets = {
            "Electricity": self.Pel_Demand,
            "Heat": self.P_th_space_heating,
            "PV": self.Pel_PV
        }
        for name, dataframe in datasets.items():
            if column_name not in dataframe.columns:
                raise ValueError(
                    f'Column "{column_name}" '
                    f'not found in {name} dataset.'
                )
        profile = pd.DataFrame({
            self.HEADER_PEL_DEMAND: self.Pel_Demand[column_name],
            self.HEADER_PTH_SPACE_HEATING: self.P_th_space_heating[column_name],
            self.HEADER_PEL_PV: self.Pel_PV[column_name]
        })

        building.energy_profile = profile
        return profile

    @staticmethod
    def plot_energy_profile(building: Building):

        if building.energy_profile is None:
            raise ValueError("Building does not contain an energy profile.")

        profile = building.energy_profile

        plt.figure(figsize=(12, 6))

        plt.plot(
            profile.index,
            profile["Pel_Demand"],
            label="Pel_Demand"
        )

        plt.plot(
            profile.index,
            profile["P_th_space_heating"],
            label="P_th_space_heating"
        )

        plt.plot(
            profile.index,
            profile["Pel_PV"],
            label="Pel_PV"
        )

        plt.xlabel("Time step")
        plt.ylabel("Power [W]")

        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


# ================================================================
# District Repository
# ================================================================

class DistrictRepository:

    @staticmethod
    def load(filepath: str) -> District:

        with open(
                filepath,
                "r",
                encoding="utf-8"
        ) as stream:

            data = yaml.safe_load(
                stream
            )

        district_data = data["district"]

        district = District(
            id=district_data["id"],
            buildings=[]
        )

        for entry in district_data["buildings"]:

            building = Building(

                id=entry["id"],

                building_type=entry[
                    "building_type"
                ],

                efficiency_class=entry[
                    "efficiency_class"
                ],

                renovation_status=entry[
                    "renovation_status"
                ],

                number_of_units=entry[
                    "number_of_units"
                ],

                current_system_type=
                HeatingSystemType[
                    entry[
                        "current_system_type"
                    ]["type"]
                ]
            )

            # ------------------------------------
            # PV-System
            # ------------------------------------

            if (
                entry["pv_system"]["enabled"]
            ):

                building.pv_system = PVSystem(

                    P_peak=
                    entry["pv_system"][
                        "p_peak"
                    ],

                    CAPEX=
                    entry["pv_system"][
                        "capex"
                    ],

                    lifetime=
                    entry["pv_system"][
                        "lifetime"
                    ],

                    maintenance_rate=
                    entry["pv_system"][
                        "maintenance_rate"
                    ]
                )

            district.add_building(
                building
            )

        return district

    @staticmethod
    def load_energy_profiles(district: District):

        repository = (
            BuildingEnergyProfileRepository()
        )

        heat_profiles = []
        electricity_profiles = []
        pv_profiles = []

        # --------------------------------------
        # Gebäudeprofile laden
        # --------------------------------------

        for building in district.buildings:
            profile = (
                repository.get_energy_profile(
                    building
                )
            )

            heat_profiles.append(
                profile[
                    BuildingEnergyProfileRepository
                    .HEADER_PTH_SPACE_HEATING
                ]
            )

            electricity_profiles.append(
                profile[
                    BuildingEnergyProfileRepository
                    .HEADER_PEL_DEMAND
                ]
            )

            pv_profiles.append(
                profile[
                    BuildingEnergyProfileRepository
                    .HEADER_PEL_PV
                ]
            )

        # --------------------------------------
        # Quartiersprofil bilden
        # --------------------------------------

        district.energy_profile = pd.DataFrame({

            BuildingEnergyProfileRepository
            .HEADER_PTH_SPACE_HEATING:
                sum(heat_profiles),

            BuildingEnergyProfileRepository
            .HEADER_PEL_DEMAND:
                sum(electricity_profiles),

            BuildingEnergyProfileRepository
            .HEADER_PEL_PV:
                sum(pv_profiles)

        })








class HeatingOperatingStrategy:
    """
    Heat-led operation strategy
    using a thermal storage.

    Priority:

    1. Thermal storage
    2. Generator
    3. Auxiliary heater
    """
    @staticmethod
    def simulate(
            system: HeatingSystem,
            heat_demand: pd.Series
    ):

        if system.thermal_storage is None:
            return system

        if system.technical.operation is None:
            raise ValueError(
                "Technical operation must be simulated first."
            )

        storage = system.thermal_storage

        operation = system.technical.operation

        dt = TIMESTEP_HOURS

        soc = storage.soc_kWh

        generator_on = False

        generator_profile = []
        soc_profile = []

        charge_profile = []
        discharge_profile = []

        generator_power_profile = []

        heat_supplied_profile = []

        W_charge = 0.0
        W_discharge = 0.0

        # ------------------------------------------------------
        # maximum thermal generator power
        # ------------------------------------------------------

        generator_power_kW_max = (
                system.technical.P_th_max
                / 1000
        )

        for idx, row in operation.iterrows():

            heat_load_W = heat_demand.iloc[idx]

            heat_load_kWh = (
                    heat_load_W
                    * dt
                    / 1000
            )

            # --------------------------------------------------
            # storage loss
            # --------------------------------------------------

            storage_loss_kWh = (
                    storage.loss_power_kW
                    * dt
            )

            soc = max(
                0.0,
                soc - storage_loss_kWh
            )

            # --------------------------------------------------
            # relative SOC
            # --------------------------------------------------

            if storage.capacity_kWh > 0:

                soc_rel = (
                        soc
                        / storage.capacity_kWh
                )

            else:

                soc_rel = 0.0

            # --------------------------------------------------
            # hysteresis
            # --------------------------------------------------

            if soc_rel <= storage.soc_min:
                generator_on = True

            elif soc_rel >= storage.soc_max:
                generator_on = False

            # --------------------------------------------------
            # discharge storage first
            # --------------------------------------------------

            discharge_kWh = min(

                storage.discharge_power_kW
                * dt,

                soc,

                heat_load_kWh
            )

            soc -= discharge_kWh

            remaining_heat_load_kWh = (
                    heat_load_kWh
                    - discharge_kWh
            )

            W_discharge += discharge_kWh

            # --------------------------------------------------
            # generator
            # --------------------------------------------------

            generator_energy_kWh = 0.0

            charge_kWh = 0.0

            if generator_on:

                generator_energy_kWh = min(

                    generator_power_kW_max
                    * dt,

                    remaining_heat_load_kWh
                    + (
                            storage.capacity_kWh
                            - soc
                    )
                )

                # heat for building

                heat_to_building_kWh = min(
                    generator_energy_kWh,
                    remaining_heat_load_kWh
                )

                # remaining energy

                excess_kWh = max(
                    0.0,
                    generator_energy_kWh
                    - heat_to_building_kWh
                )

                charge_kWh = min(

                    excess_kWh,

                    storage.charge_power_kW
                    * dt,

                    storage.capacity_kWh
                    - soc
                )

                soc += (
                        charge_kWh
                        * storage.eta_storage
                )

                W_charge += charge_kWh

            else:

                heat_to_building_kWh = 0.0

            # --------------------------------------------------
            # actual supplied heat
            # --------------------------------------------------

            delivered_heat_kWh = (
                    heat_to_building_kWh
                    + discharge_kWh
            )

            # --------------------------------------------------
            # store profiles
            # --------------------------------------------------

            generator_profile.append(
                int(generator_on)
            )

            soc_profile.append(
                soc
            )

            charge_profile.append(
                charge_kWh
            )

            discharge_profile.append(
                discharge_kWh
            )

            generator_power_profile.append(

                generator_energy_kWh
                * 1000
                / dt
            )

            heat_supplied_profile.append(

                delivered_heat_kWh
                * 1000
                / dt
            )

        # ------------------------------------------------------
        # store results
        # ------------------------------------------------------

        system.technical.generator_on_profile = (
            pd.Series(generator_profile)
        )

        system.technical.thermal_storage_soc_profile_kWh = (
            pd.Series(soc_profile)
        )

        system.technical.thermal_storage_charge_profile = (
            pd.Series(charge_profile)
        )

        system.technical.thermal_storage_discharge_profile = (
            pd.Series(discharge_profile)
        )
        """
        system.technical.generator_power_profile_W = (
            pd.Series(generator_power_profile)
        )

        system.technical.generator_heat_supplied_profile_W = (
            pd.Series(heat_supplied_profile)
        )

        system.technical.thermal_storage_result = (
            ThermalStorageResult(
                W_charge_kWh=W_charge,
                W_discharge_kWh=W_discharge,
                storage_cycles=(
                    W_discharge
                    / storage.capacity_kWh
                    if storage.capacity_kWh > 0
                    else 0.0
                ),
                soc_profile_kWh=pd.Series(
                    soc_profile
                ),
                generator_on_profile=pd.Series(
                    generator_profile
                )
            )
        )
        """
        return system

# ================================================================
# Heat Pump Model
# ================================================================

class HeatPumpModel:
    """
    Technical calculation model for heat pumps.

    The model itself does not store the simulation results.
    It writes them into a HeatingSystem object.
    """

    # Time-series column names
    PTH_HP = "P_th_HP"
    WTH_HP = "W_th_HP"

    PEL_HP = "P_el_HP"
    WEL_HP = "W_el_HP"

    PTH_AUX = "P_th_Aux"
    WTH_AUX = "W_th_Aux"

    PEL_AUX = "P_el_Aux"
    WEL_AUX = "W_el_Aux"

    def __init__(self, sizing_percentage=0.05, rounding_step=1000, timestep_hours=TIMESTEP_HOURS):
        self.sizing_percentage = sizing_percentage
        self.rounding_step = rounding_step
        self.timestep_hours = timestep_hours

    # ------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------

    def simulate(self, system: HeatingSystem, heat_demand: pd.Series):
        if system.system_type not in (HeatingSystemType.ASHP, HeatingSystemType.GSHP):
            raise ValueError("HeatPumpModel requires ASHP or GSHP.")
        if system.parameters.JAZ is None:
            raise ValueError(f"No JAZ defined for {system.name}.")

        self._calculate_simple_size(system, heat_demand)
        self._calculate_operation(system, heat_demand)
        return system

    # ------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------

    def _calculate_simple_size(self, system: HeatingSystem, heat_demand: pd.Series):
        heat_sorted = (heat_demand.dropna().sort_values(ascending=False).reset_index(drop=True))
        if heat_sorted.empty:
            raise ValueError(
                "Heat demand contains no valid values."
            )
        index = int(len(heat_sorted) * self.sizing_percentage)

        # index = min(index, len(heat_sorted) - 1)

        P_th_raw = float(heat_sorted.iloc[index])

        P_th_max = (math.floor(P_th_raw / self.rounding_step) * self.rounding_step)

        P_building_max = float(heat_sorted.max())
        P_aux_max = max(0.0, P_building_max - P_th_max)
        P_aux_max = (math.ceil(P_aux_max / self.rounding_step) * self.rounding_step)

        JAZ = system.parameters.JAZ

        system.technical.P_th_max = (P_th_max)
        system.technical.P_el_max = (P_th_max / JAZ)

        system.technical.auxiliary = (ComponentResult(P_th_max=P_aux_max, P_el_max=P_aux_max, eta=1.0))

    # ------------------------------------------------------------
    # Operation
    # ------------------------------------------------------------
    def _calculate_operation(self, system: HeatingSystem, heat_demand: pd.Series):
        P_th_max = system.technical.P_th_max
        JAZ = system.parameters.JAZ
        dt = self.timestep_hours

        # Heat pump
        P_th_hp = heat_demand.clip(upper=P_th_max)
        W_th_hp = (P_th_hp * dt)
        P_el_hp = (P_th_hp / JAZ)
        W_el_hp = (P_el_hp * dt)

        # Auxiliary heating rod
        P_th_aux = (heat_demand - P_th_hp).clip(lower=0)
        W_th_aux = (P_th_aux * dt)
        P_el_aux = P_th_aux
        W_el_aux = (P_el_aux * dt)

        operation = pd.DataFrame({
            BuildingEnergyProfileRepository.HEADER_PTH_SPACE_HEATING: heat_demand,
            self.PTH_HP: P_th_hp,
            self.WTH_HP: W_th_hp,

            self.PEL_HP: P_el_hp,
            self.WEL_HP: W_el_hp,

            self.PTH_AUX: P_th_aux,
            self.WTH_AUX: W_th_aux,

            self.PEL_AUX: P_el_aux,
            self.WEL_AUX: W_el_aux
        })

        # Annual results of complete heating system
        system.technical.W_th = (W_th_hp.sum() + W_th_aux.sum())
        system.technical.W_el = (W_el_hp.sum() + W_el_aux.sum())

        # Auxiliary heating rod results
        system.technical.auxiliary.W_th = W_th_aux.sum()
        system.technical.auxiliary.W_el = W_el_aux.sum()

        system.technical.operation = operation


# ================================================================
# Gas Boiler Model
# ================================================================
class GasBoilerModel:
    PTH_GB = "P_th_GB"
    PGAS_GB = "P_gas_GB"

    WTH_GB = "W_th_GB"
    WGAS_GB = "W_gas_GB"

    def __init__(self, timestep_hours=TIMESTEP_HOURS):
        self.timestep_hours = timestep_hours

    def simulate(self, system: HeatingSystem, heat_demand: pd.Series):
        if system.system_type != HeatingSystemType.GB:
            raise ValueError("GasBoilerModel requires a GB system.")

        if system.parameters.eta is None:
            raise ValueError("No efficiency defined for Gas Boiler.")

        eta = system.parameters.eta
        dt = self.timestep_hours

        # Boiler thermal power
        P_th_gb = heat_demand
        # Required gas power
        P_gas_gb = (P_th_gb / eta)

        # Energy per timestep
        W_th_gb = (P_th_gb * dt)
        W_gas_gb = (P_gas_gb * dt)

        operation = pd.DataFrame({
            self.PTH_GB: P_th_gb,
            self.PGAS_GB: P_gas_gb,
            self.WTH_GB: W_th_gb,
            self.WGAS_GB: W_gas_gb
        })
        # Annual technical results
        system.technical.P_th_max = (P_th_gb.max())
        system.technical.W_th = (W_th_gb.sum())
        system.technical.W_gas = (W_gas_gb.sum())
        system.technical.operation = (operation)

        return system


# ================================================================
# Investment Cost Model
# ================================================================
class InvestmentCostModel:
    """
    Calculates investment costs depending on
    heating system capacity and investment year.
    """

    @staticmethod
    def _interpolate(year: int, years: tuple, values: tuple) -> float:

        if len(years) != len(values):
            raise ValueError("Years and values must have the same length.")

        if year < years[0] or year > years[-1]:
            raise ValueError(
                f"Year {year} outside investment cost "
                f"range {years[0]}-{years[-1]}."
            )

        # Exact support year
        if year in years:
            index = years.index(year)
            return values[index]

        # Linear interpolation
        for i in range(len(years) - 1):

            year_1 = years[i]
            year_2 = years[i + 1]

            if year_1 < year < year_2:
                value_1 = values[i]
                value_2 = values[i + 1]

                fraction = ((year - year_1) / (year_2 - year_1))

                return (value_1 + fraction * (value_2 - value_1))

        raise ValueError(f"Could not interpolate year {year}.")

    def get_cost_parameters(self, system: HeatingSystem, year: int):

        if system.investment is None:
            raise ValueError(
                f"No investment cost curve defined "
                f"for {system.name}."
            )

        curve = system.investment

        a = self._interpolate(year, curve.years, curve.a_values)
        b = self._interpolate(year, curve.years, curve.b_values)

        return a, b

    @staticmethod
    def requires_investment(system: HeatingSystem, year: int, start_year: int) -> bool:
        """
        Checks whether an investment is required
        in the current simulation year.
        """

        # New systems require investment immediately
        if not system.state.is_existing:
            return True

        # Existing system:
        replacement_year = (start_year + system.state.remaining_lifetime)

        return year >= replacement_year

    def calculate_plant_capex(self, system: HeatingSystem, year: int):

        P_th_kw = (system.technical.P_th_max / 1000)

        if P_th_kw <= 0:
            raise ValueError(
                f"P_th_max of {system.name} must be > 0."
            )

        a, b = self.get_cost_parameters(system, year)

        CAPEX_plant = (a * P_th_kw ** b)

        return CAPEX_plant

    @staticmethod
    def calculate_conversion_capex(building: Building, system: HeatingSystem):

        if system.investment is None:
            return 0.0

        return (building.number_of_units * system.investment.conversion_cost_per_unit)

    @staticmethod
    def calculate_additional_capex(system: HeatingSystem):

        if system.investment is None:
            return 0.0

        return system.investment.additional_cost

    @staticmethod
    def calculate_storage_capex(system: HeatingSystem):

        if system.thermal_storage is None:
            return 0.0

        STORAGE_COST_EUR_PER_KWH = 50.0

        return (
                system.thermal_storage.capacity_kWh
                * STORAGE_COST_EUR_PER_KWH
        )

    def evaluate(self, building: Building, system: HeatingSystem, year: int, start_year: int):

        # Check whether investment is required
        investment_required = self.requires_investment(system, year, start_year)

        # Existing system before replacement
        if not investment_required:
            system.economic.CAPEX_plant = 0.0
            system.economic.CAPEX_conversion = 0.0
            system.economic.CAPEX_additional = 0.0
            system.economic.CAPEX_storage = 0.0
            system.economic.CAPEX = 0.0
            system.economic.CAPEX_ann = 0.0
            system.state.investment_year = None
            return system

        CAPEX_plant = self.calculate_plant_capex(system, year)
        CAPEX_conversion = self.calculate_conversion_capex(building, system)
        CAPEX_additional = self.calculate_additional_capex(system)
        CAPEX_storage = self.calculate_storage_capex(system)

        CAPEX_total = (CAPEX_plant + CAPEX_conversion + CAPEX_additional + CAPEX_storage)

        # Store results
        system.economic.CAPEX_plant = CAPEX_plant
        system.economic.CAPEX_conversion = CAPEX_conversion
        system.economic.CAPEX_additional = CAPEX_additional
        system.economic.CAPEX_storage = CAPEX_storage
        system.economic.CAPEX = CAPEX_total

        # Important:
        # maintenance calculation after investment
        # uses the investment value.
        system.economic.maintenance_reference_cost = CAPEX_total
        # Do NOT set investment_year here.
        # Potential paths are hypothetical.
        system.economic.CAPEX_ann = EconomicModel().calculate_annualized_capex(system.economic.CAPEX,system.economic.lifetime)

        return system


    @staticmethod
    def store_capex_history(
            operation_result: OperationResult,
            system: HeatingSystem,
            year: int
    ):
        print(system.economic.CAPEX)
        operation_result.capex_history.loc[
            year
        ] = system.economic.CAPEX

        operation_result.capex_ann_history.loc[
            year
        ] = system.economic.CAPEX_ann

        return operation_result


class PVModel:

    @staticmethod
    def calculate_lcoe(pv_system: PVSystem, economic_model):
        print(pv_system.Wel_annual_generation_kWh_a)
        if pv_system.Wel_annual_generation_kWh_a <= 0:
            raise ValueError(
                "Annual PV generation must be > 0."
            )

        pv_system.CAPEX_ann = economic_model.calculate_annualized_capex(pv_system.CAPEX, pv_system.lifetime)
        maintenance_cost = pv_system.CAPEX * pv_system.maintenance_rate
        pv_system.LCOE = (pv_system.CAPEX_ann + maintenance_cost) / pv_system.Wel_annual_generation_kWh_a

        return pv_system


class ElectricityBalanceModel:

    @staticmethod
    def calculate_hp_pv_balance(building: Building, system: HeatingSystem, operation_result: OperationResult):

        # --------------------------------------------------
        # Gas Boiler:
        # no electrical consumption modelled
        # --------------------------------------------------

        if system.system_type == HeatingSystemType.GB:

            operation_result.W_el_pv_self = 0.0
            operation_result.W_el_grid = 0.0
            operation_result.W_el_export = 0.0

            return operation_result

        # --------------------------------------------------
        # Safety checks
        # --------------------------------------------------
        # --------------------------------------
        # No PV system available
        # --------------------------------------

        if building.pv_system is None:
            operation_result.W_el_pv_self = 0.0
            operation_result.W_el_grid = (operation_result.W_el_total)
            operation_result.W_el_export = 0.0
            return operation_result

        if (operation_result.P_el_generator is None or operation_result.P_el_auxiliary is None):

            raise ValueError(
                "Electrical operation profiles missing."
            )

        # --------------------------------------------------
        # PV generation is available
        # --------------------------------------------------
        pv_generation = (building.energy_profile[BuildingEnergyProfileRepository.HEADER_PEL_PV])

        # --------------------------------------------------
        # Electrical load
        # --------------------------------------------------
        electrical_load = (operation_result.P_el_generator + operation_result.P_el_auxiliary)

        # --------------------------------------------------
        # PV self-consumption
        # --------------------------------------------------
        pv_self = pd.concat([pv_generation, electrical_load],axis=1).min(axis=1)

        # --------------------------------------------------
        # Grid import
        # --------------------------------------------------
        grid_import = (electrical_load - pv_self).clip(lower=0)

        # --------------------------------------------------
        # PV export
        # --------------------------------------------------
        pv_export = (pv_generation - pv_self).clip(lower=0)

        # --------------------------------------------------
        # Annual energies [kWh/a]
        # --------------------------------------------------

        operation_result.W_el_pv_self = (
            pv_self.sum()
            * TIMESTEP_HOURS
            / 1000
        )

        operation_result.W_el_grid = (
            grid_import.sum()
            * TIMESTEP_HOURS
            / 1000
        )

        operation_result.W_el_export = (
            pv_export.sum()
            * TIMESTEP_HOURS
            / 1000
        )

        return operation_result

class EmissionModel:

    @staticmethod
    def calculate_emissions(
        system: HeatingSystem,
        operation_result: OperationResult,
        emission_factor
    ):

        if emission_factor < 0:

            raise ValueError(
                "Emission factor must be >= 0."
            )

        # ------------------------------
        # Final energy
        # ------------------------------

        if system.system_type in (
            HeatingSystemType.ASHP,
            HeatingSystemType.GSHP
        ):

            energy_kwh = (
                operation_result.W_el_total
            )

        elif system.system_type == (
            HeatingSystemType.GB
        ):

            energy_kwh = (
                operation_result.W_gas_generator
            )

        else:

            raise NotImplementedError()

        emissions = (
            energy_kwh
            * emission_factor
        )

        operation_result.CO2_emissions = (
            emissions
        )

        if operation_result.W_th_total > 0:

            operation_result.CO2_specific = (
                emissions
                /
                operation_result.W_th_total
            )

        return operation_result

# ================================================================
# Economic Model
# ================================================================

class EconomicModel:

    def __init__(self, discount_rate=0.03):
        if discount_rate < 0: raise ValueError("Discount rate must be >= 0.")
        self.discount_rate = (discount_rate)

    # ------------------------------------------------------------
    # Annuity
    # ------------------------------------------------------------

    def calculate_annuity_factor(self, lifetime):

        if lifetime <= 0: raise ValueError("Lifetime must be greater than 0.")

        i = self.discount_rate
        n = lifetime

        if i == 0:
            return 1 / n

        return (i * (1 + i) ** n / ((1 + i) ** n - 1))

    def calculate_annualized_capex(self, CAPEX, lifetime):
        return (CAPEX * self.calculate_annuity_factor(lifetime))

    # ------------------------------------------------------------
    # OPEX
    # ------------------------------------------------------------

    @staticmethod
    def calculate_energy_opex(energy_demand_wh, energy_price):

        energy_demand_kwh = (energy_demand_wh / 1000)

        return (energy_demand_kwh * energy_price)

    @staticmethod
    def calculate_maintenance_cost(CAPEX, maintenance_rate):
        return (CAPEX * maintenance_rate)

    # ------------------------------------------------------------
    # CO2 cost
    # ------------------------------------------------------------

    @staticmethod
    def calculate_co2_cost(emissions_kg, co2_price):
        """
        Parameters
        ----------
        emissions_kg:
            Annual CO2 emissions [kg CO2/a]

        co2_price:
            CO2 price [EUR/t CO2]

        Returns
        -------
        Annual CO2 costs [EUR/a]
        """

        if emissions_kg < 0:
            raise ValueError(
                "CO2 emissions must be >= 0."
            )

        if co2_price < 0:
            raise ValueError(
                "CO2 price must be >= 0."
            )

        emissions_t = emissions_kg / 1000

        return (emissions_t * co2_price)

    def calculate_opex(
            self,
            system: HeatingSystem,
            operation_result: OperationResult,
            year: int,
            energy_price: float,
            co2_price: float,
            pv_lcoe: float = 0.0
    ):

        # ----------------------------------
        # Energy
        # ----------------------------------

        if system.system_type in (
                HeatingSystemType.ASHP,
                HeatingSystemType.GSHP
        ):

            opex_energy = (

                    operation_result.W_el_grid
                    * energy_price

                    +

                    operation_result.W_el_pv_self
                    * pv_lcoe

            )

        elif system.system_type == (
                HeatingSystemType.GB
        ):

            opex_energy = (

                    operation_result.W_gas_generator
                    * energy_price

            )

        else:

            raise NotImplementedError()

        # ----------------------------------
        # Maintenance
        # ----------------------------------

        opex_maint = (

                system.economic.CAPEX
                * system.economic.maintenance_rate

        )

        # ----------------------------------
        # CO2
        # ----------------------------------

        opex_co2 = (

            self.calculate_co2_cost(
                operation_result.CO2_emissions,
                co2_price
            )

        )

        # ----------------------------------
        # Total
        # ----------------------------------

        opex_total = (

                opex_energy
                + opex_maint
                + opex_co2

        )

        # ----------------------------------
        # Store
        # ----------------------------------

        operation_result.energy_price_history.loc[
            year
        ] = energy_price

        operation_result.co2_price_history.loc[
            year
        ] = co2_price

        operation_result.opex_energy_history.loc[
            year
        ] = opex_energy

        operation_result.opex_maint_history.loc[
            year
        ] = opex_maint

        operation_result.opex_co2_history.loc[
            year
        ] = opex_co2

        operation_result.opex_total_history.loc[
            year
        ] = opex_total

        return operation_result

    # ------------------------------------------------------------
    # LCOH
    # ------------------------------------------------------------
    @staticmethod
    def calculate_lcoh(C_ann, W_th):

        if W_th <= 0: raise ValueError("Useful heat must be greater than 0.")

        W_th_kwh = (W_th / 1000)

        return (C_ann / W_th_kwh)

    @staticmethod
    def calculate_energy_opex_with_pv(W_el_grid, electricity_price, W_el_PV_self, pv_lcoe):

        W_grid_kWh = (W_el_grid / 1000)
        W_pv_kWh = (W_el_PV_self / 1000)

        return (W_grid_kWh * electricity_price + W_pv_kWh * pv_lcoe)

    # ------------------------------------------------------------
    # Complete evaluation
    # ------------------------------------------------------------

    def evaluate(
            self,
            system: HeatingSystem,
            operation_result: OperationResult,
            year: int
    ):

        # --------------------------------------------------
        # Annual costs
        # --------------------------------------------------

        C_ann = (

                operation_result.capex_ann_history.loc[year]

                +

                operation_result.opex_total_history.loc[year]

        )

        # --------------------------------------------------
        # LCOH
        # --------------------------------------------------

        LCOH = (

            self.calculate_lcoh(

                C_ann,

                operation_result.W_th_total * 1000

            )

        )

        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        operation_result.c_ann_history.loc[
            year
        ] = C_ann

        operation_result.lcoh_history.loc[
            year
        ] = LCOH

        return operation_result

# ================================================================
# Helper Functions
# ================================================================

def get_energy_price_for_system(system: HeatingSystem, electricity_price: float, gas_price: float):
    if system.system_type in (HeatingSystemType.ASHP, HeatingSystemType.GSHP):
        return (
            electricity_price
        )
    if (system.system_type == HeatingSystemType.GB):
        return (gas_price)

    raise NotImplementedError(
        f"No energy price mapping for "
        f"{system.system_type.value}."
    )


def get_emission_factor_for_system(system: HeatingSystem, ef_electricity: float):
    if system.system_type in (
            HeatingSystemType.ASHP,
            HeatingSystemType.GSHP
    ):
        return (
            ef_electricity
        )

    if (
            system.system_type
            == HeatingSystemType.GB
    ):
        return (
            EF_GAS
        )

    raise NotImplementedError(
        f"No emission factor mapping for "
        f"{system.system_type.value}."
    )


def get_co2_price_for_system(system: HeatingSystem, co2_price: float):
    # Owner perspective:
    # no separate electricity CO2 surcharge.
    if system.system_type in (
            HeatingSystemType.ASHP,
            HeatingSystemType.GSHP
    ):
        return (
            0.0
        )

    if (
            system.system_type
            == HeatingSystemType.GB
    ):
        return (
            co2_price
        )

    raise NotImplementedError(
        f"No CO2 price mapping for "
        f"{system.system_type.value}."
    )


class PlotBase:
    """
    central class for all other PlotClasses

    """
    pass

class OperationResultPlot(PlotBase):
    """
    OperationResultPlot
    ├── plot_operation_results()
    ├── plot_storage_operation()
    ├── plot_generator_runtime()
    └── plot_load_duration_curve()
    """

    @staticmethod
    def plot_operation_strategy_comparison(
            system: HeatingSystem,
            heat_demand: pd.Series
    ):
        """
        Vergleich aller vorhandenen Betriebsstrategien.

        HeatLed:

            Generator - StorageCharge
            + StorageCharge
            + Auxiliary
            + StorageDischarge

        Dadurch wird sichtbar:

        - Lastdeckung durch Wärmepumpe
        - Speicherladung
        - Speicherentladung
        - Auxiliary-Einsatz
        """

        if not system.operation_results:
            raise ValueError(
                f"No operation results available "
                f"for {system.name}."
            )

        results = system.operation_results

        n_results = len(results)

        fig, axes = plt.subplots(
            n_results,
            1,
            figsize=(18, 5 * n_results),
            sharex=True
        )

        if n_results == 1:
            axes = [axes]

        x = heat_demand.index

        for ax, (
                strategy_name,
                result
        ) in zip(
            axes,
            results.items()
        ):

            # ==================================================
            # Heat Demand
            # ==================================================

            ax.plot(
                x,
                heat_demand,
                color="black",
                linewidth=2,
                label="Heat Demand",
                zorder=10
            )

            # ==================================================
            # Direct Heat
            # ==================================================

            if strategy_name == "DirectHeat":

                ax.stackplot(
                    x,
                    result.P_th_generator,
                    result.P_th_auxiliary,
                    labels=[
                        "Generator",
                        "Auxiliary"
                    ],
                    colors=[
                        "tab:blue",
                        "tab:red"
                    ],
                    alpha=0.80
                )

            # ==================================================
            # Heat Led Storage
            # ==================================================

            else:

                charge = result.P_th_storage_charge
                discharge = result.P_th_storage_discharge

                # ----------------------------------------------
                # Generator für Gebäudelast bereinigen
                # ----------------------------------------------

                if charge is not None:

                    generator_net = (
                            result.P_th_generator
                            - charge
                    ).clip(lower=0)

                else:

                    generator_net = (
                        result.P_th_generator
                    )

                stack_data = []
                stack_labels = []
                stack_colors = []

                # ----------------------------------------------
                # 1 Generator deckt Last
                # ----------------------------------------------

                stack_data.append(
                    generator_net
                )

                stack_labels.append(
                    "Generator"
                )

                stack_colors.append(
                    "tab:blue"
                )

                # ----------------------------------------------
                # 2 Speicherladung
                # ----------------------------------------------

                if charge is not None:
                    stack_data.append(
                        charge
                    )

                    stack_labels.append(
                        "Storage Charge"
                    )

                    stack_colors.append(
                        "green"
                    )

                # ----------------------------------------------
                # 3 Auxiliary
                # ----------------------------------------------

                stack_data.append(
                    result.P_th_auxiliary
                )

                stack_labels.append(
                    "Auxiliary"
                )

                stack_colors.append(
                    "tab:red"
                )

                # ----------------------------------------------
                # 4 Speicherentladung
                # ----------------------------------------------

                if discharge is not None:
                    stack_data.append(
                        discharge
                    )

                    stack_labels.append(
                        "Storage Discharge"
                    )

                    stack_colors.append(
                        "orange"
                    )

                ax.stackplot(
                    x,
                    *stack_data,
                    labels=stack_labels,
                    colors=stack_colors,
                    alpha=0.80
                )

            # ==================================================
            # Layout pro Subplot
            # ==================================================

            ax.set_title(
                strategy_name,
                fontsize=12,
                fontweight="bold"
            )

            ax.set_ylabel(
                "Thermal Power [W]"
            )

            ax.grid(
                True,
                linestyle="--",
                linewidth=0.7,
                alpha=0.4
            )

            ax.legend(
                loc="upper right"
            )

        # ======================================================
        # Gesamtlayout
        # ======================================================

        axes[-1].set_xlabel(
            "Timestep [-]"
        )

        fig.suptitle(
            (
                f"{system.name}\n"
                f"Comparison of Operation Strategies"
            ),
            fontsize=16,
            fontweight="bold"
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )

        plt.show()


class EconomicResultPlot(PlotBase):
    """
    EconomicResultPlot
    ├── plot_lcoh()
    ├── plot_capex_opex()
    └── plot_cost_breakdown()
    """

    @staticmethod
    def plot_economic_strategy_comparison(
            building: Building
    ):
        """
        Wirtschaftlicher Vergleich aller Betriebsstrategien.

        Zeilen:
            Betriebsstrategien

        Linke Spalte:
            Kostenstruktur
            (CAPEX + OPEX)

        Rechte Spalte:
            LCOH-Verlauf
        """

        if not building.heating_systems:
            raise ValueError(
                "No heating systems available."
            )

        first_system = next(
            iter(
                building.heating_systems.values()
            )
        )

        strategy_names = list(
            first_system.operation_results.keys()
        )

        n_strategies = len(
            strategy_names
        )

        fig, axes = plt.subplots(
            n_strategies,
            2,
            figsize=(20, 4.2 * n_strategies),
            squeeze=False
        )

        technology_colors = {
            HeatingSystemType.ASHP: "tab:blue",
            HeatingSystemType.GSHP: "tab:green",
            HeatingSystemType.GB: "tab:gray",
            HeatingSystemType.DH: "tab:orange"
        }

        systems = list(
            building.heating_systems.values()
        )

        for row, strategy_name in enumerate(
                strategy_names
        ):

            ax_cost = axes[row, 0]
            ax_lcoh = axes[row, 1]

            reference_result = (
                systems[0]
                .operation_results[strategy_name]
            )

            years = list(
                reference_result
                .lcoh_history.index
            )

            x = list(
                range(len(years))
            )

            bar_width = (
                    0.8 / len(systems)
            )

            # =====================================================
            # Spaltenüberschriften nur in erster Zeile
            # =====================================================

            if row == 0:
                ax_cost.set_title(
                    "Economic Cost Structure",
                    fontsize=13,
                    fontweight="bold"
                )

                ax_lcoh.set_title(
                    "Levelized Cost of Heat (LCOH)",
                    fontsize=13,
                    fontweight="bold"
                )

            # =====================================================
            # Strategien als Zeilenbeschriftung
            # =====================================================

            ax_cost.text(
                -0.18,
                0.50,
                strategy_name,
                transform=ax_cost.transAxes,
                rotation=90,
                fontsize=12,
                fontweight="bold",
                va="center",
                ha="center"
            )

            # =====================================================
            # COST STRUCTURE
            # =====================================================

            for i, system in enumerate(
                    systems
            ):
                result = (
                    system.operation_results[
                        strategy_name
                    ]
                )

                color = technology_colors.get(
                    system.system_type,
                    "black"
                )

                x_pos = [
                    value
                    + (
                            i
                            - (len(systems) - 1) / 2
                    ) * bar_width
                    for value in x
                ]

                capex = (
                    result.capex_ann_history
                    .reindex(years)
                    .to_numpy()
                )

                opex_energy = (
                    result.opex_energy_history
                    .reindex(years)
                    .to_numpy()
                )

                opex_maint = (
                    result.opex_maint_history
                    .reindex(years)
                    .to_numpy()
                )

                opex_co2 = (
                    result.opex_co2_history
                    .reindex(years)
                    .to_numpy()
                )

                ax_cost.bar(
                    x_pos,
                    capex,
                    width=bar_width,
                    color=color,
                    alpha=0.35,
                    hatch="//",
                    edgecolor="black"
                )

                ax_cost.bar(
                    x_pos,
                    opex_energy,
                    width=bar_width,
                    bottom=capex,
                    color=color,
                    alpha=0.90
                )

                ax_cost.bar(
                    x_pos,
                    opex_maint,
                    width=bar_width,
                    bottom=(
                            capex
                            + opex_energy
                    ),
                    color=color,
                    alpha=0.60,
                    hatch="..",
                    edgecolor="black"
                )

                ax_cost.bar(
                    x_pos,
                    opex_co2,
                    width=bar_width,
                    bottom=(
                            capex
                            + opex_energy
                            + opex_maint
                    ),
                    color=color,
                    alpha=0.75,
                    hatch="xx",
                    edgecolor="black"
                )

            ax_cost.set_ylabel(
                "Costs [EUR/a]"
            )

            ax_cost.grid(
                True,
                axis="y",
                linestyle="--",
                alpha=0.4
            )

            # Jahresachse nur unten

            if row == n_strategies - 1:

                ax_cost.set_xticks(x)

                ax_cost.set_xticklabels(
                    years,
                    rotation=45
                )

            else:

                ax_cost.set_xticks([])

            # =====================================================
            # LEGENDEN NUR OBEN
            # =====================================================

            if row == 0:
                technology_handles = [
                    Patch(
                        facecolor=color,
                        label=tech.name
                    )
                    for tech, color
                    in technology_colors.items()
                    if any(
                        system.system_type == tech
                        for system in systems
                    )
                ]

                tech_legend = ax_cost.legend(
                    handles=technology_handles,
                    title="Technology",
                    loc="upper left"
                )

                ax_cost.add_artist(
                    tech_legend
                )

                cost_handles = [
                    Patch(
                        facecolor="white",
                        edgecolor="black",
                        hatch="//",
                        label="CAPEX ann."
                    ),
                    Patch(
                        facecolor="gray",
                        edgecolor="black",
                        label="Energy OPEX"
                    ),
                    Patch(
                        facecolor="white",
                        edgecolor="black",
                        hatch="..",
                        label="Maintenance OPEX"
                    ),
                    Patch(
                        facecolor="white",
                        edgecolor="black",
                        hatch="xx",
                        label="CO₂ OPEX"
                    )
                ]

                ax_cost.legend(
                    handles=cost_handles,
                    title="Cost Components",
                    loc="upper right"
                )

            # =====================================================
            # LCOH
            # =====================================================

            for system in systems:
                result = (
                    system.operation_results[
                        strategy_name
                    ]
                )

                color = technology_colors.get(
                    system.system_type,
                    "black"
                )

                ax_lcoh.plot(
                    result.lcoh_history.index,
                    result.lcoh_history.values,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    color=color,
                    label=system.name
                )

            ax_lcoh.set_ylabel(
                "LCOH [EUR/kWh]"
            )

            ax_lcoh.grid(
                True,
                linestyle="--",
                alpha=0.4
            )

            if row == 0:
                ax_lcoh.legend(
                    title="Technology",
                    loc="best"
                )

            if row == n_strategies - 1:

                ax_lcoh.set_xticks(
                    years
                )

                ax_lcoh.tick_params(
                    axis="x",
                    rotation=45
                )

            else:

                ax_lcoh.set_xticks([])

        fig.suptitle(
            (
                f"Economic Comparison of "
                f"Operation Strategies\n"
                f"Building {building.id}"
            ),
            fontsize=16,
            fontweight="bold"
        )

        plt.tight_layout(
            rect=[0.05, 0.02, 1, 0.95],
            h_pad=0.8
        )

        plt.show()


class TransformationResultPlot(PlotBase):
    """
    TransformationPlot
    ├── plot_decision_history()
    └── plot_transformation_path()
    """

    pass



class DistrictResultPlot(PlotBase):
    """
    DistrictPlot
    ├── plot_district_load()
    └── plot_district_comparison()
    """

    @staticmethod
    def plot_energy_balance(
            district: District
    ):
        """
        Quartiersbilanz

        Obere Grafik:
            Strombedarf
            PV-Erzeugung

        Untere Grafik:
            Residuallast
        """

        result = district.district_result

        if result is None:

            raise ValueError(
                "No district results available."
            )

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(16, 8),
            sharex=True,
            gridspec_kw={
                "height_ratios": [2, 1]
            }
        )

        # =====================================================
        # Oberer Subplot
        # Strombedarf + PV
        # =====================================================

        ax1 = axes[0]

        ax1.plot(
            result.electricity_demand_profile.index,
            result.electricity_demand_profile,
            color="black",
            linewidth=1.5,
            label="Electricity Demand"
        )

        ax1.plot(
            result.pv_generation_profile.index,
            result.pv_generation_profile,
            color="green",
            linewidth=1.5,
            label="PV Generation"
        )

        ax1.set_ylabel(
            "Power [W]"
        )

        ax1.set_title(
            f"{district.id} - Electricity Balance",
            fontweight="bold"
        )

        ax1.grid(
            True,
            linestyle="--",
            alpha=0.3
        )

        ax1.legend(
            loc="upper right"
        )

        # =====================================================
        # Subplot 2
        # Residualload
        # =====================================================

        ax2 = axes[1]

        residual = (
            result.residual_load_profile
        )

        ax2.plot(
            residual.index,
            residual,
            color="tab:red",
            linewidth=1.5,
            label="Residual Load"
        )

        ax2.fill_between(
            residual.index,
            0,
            residual,
            color="tab:red",
            alpha=0.3
        )

        ax2.axhline(
            0,
            color="black",
            linewidth=0.8
        )

        ax2.set_ylabel(
            "Power [W]"
        )

        ax2.set_xlabel(
            "Timestep [-]"
        )

        ax2.grid(
            True,
            linestyle="--",
            alpha=0.3
        )

        ax2.legend(
            loc="upper right"
        )

        # =====================================================
        # Titel
        # =====================================================

        fig.suptitle(
            f"District Energy Balance - {district.id}",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )

        plt.show()

@dataclass
class TechnologyParameter:

    # --------------------------------------------------
    # Allgemein
    # --------------------------------------------------
    name: str
    system_type: HeatingSystemType
    lifetime: int

    # --------------------------------------------------
    # Technische Parameter
    # --------------------------------------------------
    JAZ: Optional[float] = None
    eta: Optional[float] = None

    # --------------------------------------------------
    # Investitionsfunktion
    #
    # CAPEX = a * Pth^b
    # --------------------------------------------------
    years: tuple[int, ...] = ()
    capex_a_values: tuple[float, ...] = ()
    capex_b_values: tuple[float, ...] = ()

    # --------------------------------------------------
    # Zusätzliche Kosten
    # abhängig von der Leistung
    # --------------------------------------------------
    p_th_reference: tuple[float, ...] = ()
    p_el_reference: tuple[float, ...] = ()
    capex_add_values: tuple[float, ...] = ()

    # --------------------------------------------------
    # Wartungskosten
    # abhängig von der Leistung
    # --------------------------------------------------
    opex_maintenance_values: tuple[float, ...] = ()

    # --------------------------------------------------
    # Umbaukosten
    # --------------------------------------------------
    conversion_cost_per_unit: float = 0.0

    additional_cost: float = 0.0

class TechnologyRepository:
    """
    Link to the data for each Energy system
    https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fapi.kww-halle.de%2Ffileadmin%2FPDFs%2FKWW-Technikkatalog-Waermeplanung_08-2026_Version_1.1.xlsx&wdOrigin=BROWSELINK
    """
    @staticmethod
    def get_ashp():
        return TechnologyParameter(

            name="Air Source Heat Pump",

            system_type=HeatingSystemType.ASHP,
            lifetime=20,
            JAZ = 3.75,

            # Investment costs as parameters a and b as a function of time
            years=(2025, 2026, 2030, 2035, 2040, 2045),
            capex_a_values=(4708, 4708, 4614, 4472, 4378, 4237),
            capex_b_values=(0.73, 0.73, 0.73, 0.73, 0.73, 0.73),

            # CAPEX_Add and OPEX_Maintanance varies depending on the plant's output
            p_th_reference = (5, 10, 20, 30, 40, 50, 60, 80, 100),
            capex_add_values = (1160, 980, 830, 760, 710, 680, 660, 630, 600),
            opex_maintenance_values =(200, 400, 600, 800, 1000, 1200, 1300, 1600, 1900),

            conversion_cost_per_unit=5000
        )

    @staticmethod
    def get_gshp():
        return TechnologyParameter(

            name="Ground Source Heat Pump",

            system_type=HeatingSystemType.GSHP,
            lifetime=20,
            JAZ = 4.6,

            # Investment costs as parameters a and b as a function of time
            years=(2025, 2026, 2030, 2035, 2040, 2045),
            capex_a_values =( 7810, 7810, 7654, 7419, 7263, 7029),
            capex_b_values =(0.77, 0.77, 0.77, 0.77, 0.77, 0.77),

            # CAPEX_Add and OPEX_Maintanance varies depending on the plant's output
            p_th_reference =(5, 10, 20, 30, 40, 50, 60, 80, 100),
            capex_add_values = (3390, 3060, 2780, 2620, 2520, 2450, 2400, 2.320, 2260),
            opex_maintenance_values =(300, 400, 600, 700, 800, 900, 1000, 1100, 1300),

            conversion_cost_per_unit=5000
        )

    @staticmethod
    def get_gb():
        return TechnologyParameter(

            name="Gas Boiler",

            system_type=HeatingSystemType.GB,
            lifetime=20,

            # Investment costs as parameters a and b as a function of time
            years=(2025, 2026, 2030, 2035, 2040, 2045),
            capex_a_values =(3528, 3528, 3528, 3704, 3881, 4057),
            capex_b_values =(0.52, 0.52, 0.52, 0.52, 0.52, 0.52),

            # CAPEX_Add and OPEX_Maintanance varies depending on the plant's output
            p_th_reference = (10, 20, 30, 60, 100),
            capex_add_values = (540, 400, 340, 260, 220),
            opex_maintenance_values = (200, 300, 300, 400, 500),

            conversion_cost_per_unit=5000
        )



    @staticmethod
    def get_chp():
        return TechnologyParameter(

            name="Combined Heat and Power",

            system_type=HeatingSystemType.CHP,
            lifetime=20,

            # Investment costs as parameters a and b as a function of time
            years=(2025, 2026, 2030, 2035, 2040, 2045),
            capex_a_values = (10156, 10156, 9953, 9648, 9445, 9148),
            capex_b_values = (0.59, 0.59, 0.59, 0.59, 0.59, 0.59),

            # CAPEX_Add and OPEX_Maintanance varies depending on the plant's output
            p_th_reference = (5, 10, 20, 40, 60, 100),
            p_el_reference = (2, 4, 9, 19, 30, 52),
            capex_add_values = (750, 690, 640, 600, 570, 540),
            opex_maintenance_values = (1800, 2600, 3700, 5200, 6400, 8200),

            conversion_cost_per_unit=5000
        )

    @staticmethod
    def interpolate_power_dependent_values(p_th_kw, x_values, y_values):
        return np.interp(p_th_kw, x_values, y_values)

# ================================================================
# Heating System Factory
# ================================================================

class HeatingSystemFactory:

    @staticmethod
    def assign_candidate_systems(
            building: Building,
    ):

        systems = [
            HeatingSystemFactory.create_ashp(),
            HeatingSystemFactory.create_gshp(),
            HeatingSystemFactory.create_gas_boiler(building)
        ]

        for system in systems:

            building.add_heating_system(
                system
            )

        return building

    @staticmethod
    def create_ashp(
    ) -> HeatingSystem:

        tech = (
            TechnologyRepository
            .get_ashp()
        )

        system = HeatingSystem(

            system_type=
                HeatingSystemType.ASHP,

            name=tech.name,

            parameters=
                TechnicalParameters(
                    JAZ=tech.JAZ
                ),

            investment=
                InvestmentCostCurve(

                    years=
                        tech.years,

                    a_values=
                        tech.capex_a_values,

                    b_values=
                        tech.capex_b_values,

                    conversion_cost_per_unit=
                        tech.conversion_cost_per_unit,

                    additional_cost=
                        0.0
                ),

            economic=
                EconomicResult(
                    lifetime=
                        tech.lifetime
                )
        )

        #HeatingSystemFactory._add_storage()

        return system

    @staticmethod
    def create_gshp() -> HeatingSystem:

        tech = (
            TechnologyRepository
            .get_gshp()
        )

        system = HeatingSystem(

            system_type=
                HeatingSystemType.GSHP,

            name=tech.name,

            parameters=
                TechnicalParameters(
                    JAZ=tech.JAZ
                ),

            investment=
                InvestmentCostCurve(

                    years=
                        tech.years,

                    a_values=
                        tech.capex_a_values,

                    b_values=
                        tech.capex_b_values,

                    conversion_cost_per_unit=
                        tech.conversion_cost_per_unit,

                    additional_cost=
                        0.0
                ),

            economic=
                EconomicResult(
                    lifetime=
                        tech.lifetime
                )
        )

        #HeatingSystemFactory._add_storage()

        return system

    @staticmethod
    def create_gas_boiler(building: Building) -> HeatingSystem:

        tech = (
            TechnologyRepository
            .get_gb()
        )

        system = HeatingSystem(

            system_type=
                HeatingSystemType.GB,

            name=tech.name,

            state=
                SystemState(
                    is_existing=True,

                    remaining_lifetime=15
                ),

            parameters=
                TechnicalParameters(
                    eta=0.90
                ),

            investment=
                InvestmentCostCurve(

                    years=
                        tech.years,

                    a_values=
                        tech.capex_a_values,

                    b_values=
                        tech.capex_b_values,

                    conversion_cost_per_unit=
                        tech.conversion_cost_per_unit,

                    additional_cost=
                        0.0
                ),

            economic=
                EconomicResult(
                    lifetime=
                        tech.lifetime
                )
        )

        return system


class BuildingSimulation:


    def __init__(self, scenario: SimulationScenario, strategies: dict):
        self.scenario = scenario
        self.strategies = strategies
        self.heat_pump_model = HeatPumpModel()
        self.gas_boiler_model = GasBoilerModel()
        self.electricity_balance_model = ElectricityBalanceModel()
        self.emission_model = EmissionModel()

    def simulate_technical(self, building: Building):
        heat_demand = building.energy_profile[BuildingEnergyProfileRepository.HEADER_PTH_SPACE_HEATING]

        for system in (building.heating_systems.values()):
            if system.system_type in (HeatingSystemType.ASHP,HeatingSystemType.GSHP):
                self.heat_pump_model.simulate(
                    system,
                    heat_demand
                )
            elif system.system_type == HeatingSystemType.GB:
                self.gas_boiler_model.simulate(
                    system,
                    heat_demand
                )

    def simulate_operation(self, building: Building):

        heat_demand = building.energy_profile[BuildingEnergyProfileRepository.HEADER_PTH_SPACE_HEATING]

        for name, strategy in self.strategies.items():
            for system in building.heating_systems.values():
                system.operation_results[name] = strategy.simulate(building, system, heat_demand)


# ================================================================
# Main
# ================================================================

if __name__ == '__main__':

    technology = TechnologyRepository()

    tech = TechnologyRepository.get_chp()

    capex_add_Euro_per_kW = (
        TechnologyRepository
        .interpolate_power_dependent_values(
            35,
            tech.p_th_reference,
            tech.capex_add_values
        )
    )
    opex_main_Euro_per_year = (
        TechnologyRepository
        .interpolate_power_dependent_values(
            35,
            tech.p_th_reference,
            tech.opex_maintenance_values
        )
    )


    print(capex_add_Euro_per_kW)
    print(opex_main_Euro_per_year)

#----------------------------------
# Scenario
#----------------------------------

    profile_repository = BuildingEnergyProfileRepository()
    #----------------------------------
    # Lodaing config YAML district and Buildings
    #----------------------------------
    district = DistrictRepository.load(
        "config/district_test.yml"
    )

    #district = DistrictRepository.load_energy_profiles(district)
    #----------------------------------
    # Loading Energy Profiles for Buildings and in sum for a District
    #----------------------------------
    DistrictRepository.load_energy_profiles(district)

    print(district.energy_profile)

    print(district.id)
    for building in district.buildings:
        #profile_repository.get_energy_profile(building)

        HeatingSystemFactory.assign_candidate_systems(building)

        print(
            building.id,
            building.building_type,
            building.current_system_type
        )

    """
    Ich bin bis hierhin gekommen!!!
    Die nächsten Schritte: 
    - Zuweisung der Technologien und Auslegung je Gebäude
    - Zuweisung der Technologien und Auslegung für das Quartier 
    """

    print("Test YAML")



#----------------------------------
# Energy profile and  Technology for each building
#----------------------------------
    profile_repository = BuildingEnergyProfileRepository()
    for building in district.buildings:
        profile_repository.get_energy_profile(building)

        air_hp = []
        ground_hp = []
        gas_hp = []

        building.add_heating_system(air_hp)
        building.add_heating_system(ground_hp)
        building.add_heating_system(gas_boiler)

#--------------------------------------
# OLD
#--------------------------------------

    scenario = SimulationScenario(
        use_pv=True,
        use_thermal_storage = True,

        thermal_storage_capacity_kWh = 50,
        thermal_storage_charge_power_kW = 20,
        thermal_storage_discharge_power_kW = 20,
        thermal_storage_efficiency = 0.95,
        thermal_storage_soc_min = 0.20,
        thermal_storage_soc_max = 0.80,
        thermal_storage_loss_power_kW = 0.10
    )

    ef_electricity = 0

    strategies = {
        "DirectHeat": DirectHeatStrategy(),
        "HeatLed": HeatLedStorageStrategy()
    }

    # ============================================================
    # Create Building
    # ============================================================

    building = Building(
        id="B001",
        building_type="MFH",
        efficiency_class="J",
        renovation_status="0",
        number_of_units=8,
        current_system_type=HeatingSystemType.GB
    )

    district = District(
        id="Q001",
        buildings=[]
    )

    district.add_building(building)


    if scenario.use_pv:
        building.pv_system = PVSystem(
            P_peak=30,
            CAPEX=45000,
            lifetime=25,
            maintenance_rate=0.01
        )
    else:
        building.pv_system = None

    # ============================================================
    # Load Energy Profiles
    # ============================================================
    profile_repository = BuildingEnergyProfileRepository()
    profile_repository.get_energy_profile(building)
    heat_demand = building.energy_profile[BuildingEnergyProfileRepository.HEADER_PTH_SPACE_HEATING]
    if scenario.use_pv:
        building.pv_system.Wel_annual_generation_kWh_a = building.energy_profile[
                                                             BuildingEnergyProfileRepository.HEADER_PEL_PV].sum() * TIMESTEP_HOURS / 1000

    district_model = DistrictModel()

    district_model.evaluate(district)
    DistrictResultPlot.plot_energy_balance(district)
    # ============================================================
    # Create Heating Systems
    # ============================================================

    air_hp = HeatingSystem(
        system_type=HeatingSystemType.ASHP,
        name="Air Source Heat Pump",
        parameters=TechnicalParameters(JAZ=3.75),
        investment=InvestmentCostCurve(
            years=(2025, 2026, 2030, 2035, 2040, 2045),
            a_values=(4708, 4708, 4614, 4472, 4378, 4237),
            b_values=(0.73, 0.73, 0.73, 0.73, 0.73, 0.73),
            conversion_cost_per_unit=5000.0,
            additional_cost=0.0
        ),
        economic=EconomicResult(
            lifetime=20,
            maintenance_rate=0.015
        )
    )
    heat_demand = building.energy_profile[BuildingEnergyProfileRepository.HEADER_PTH_SPACE_HEATING]
    ground_hp = HeatingSystem(
        system_type=HeatingSystemType.GSHP,
        name="Ground Source Heat Pump",
        parameters=TechnicalParameters(JAZ=4.6),
        investment=InvestmentCostCurve(
            years=(2025, 2026, 2030, 2035, 2040, 2045),
            a_values=(7810, 7810, 7654, 7419, 7263, 7029),
            b_values=(0.77, 0.77, 0.77, 0.77, 0.77, 0.77),
            conversion_cost_per_unit=5000.0,
            additional_cost=0.0
        ),
        economic=EconomicResult(
            lifetime=25,
            maintenance_rate=0.015
        )
    )

    gas_boiler = HeatingSystem(
        system_type=HeatingSystemType.GB,
        name="Gas Boiler",
        state=SystemState(
            is_existing=True,
            remaining_lifetime=15
        ),
        parameters=TechnicalParameters(eta=0.90),
        investment=InvestmentCostCurve(
            years=(2025, 2026, 2030, 2035, 2040, 2045),
            a_values=(3528, 3528, 3528, 3704, 3881, 4057),
            b_values=(0.52, 0.52, 0.52, 0.52, 0.52, 0.52),
            conversion_cost_per_unit=0.0,
            additional_cost=0.0
        ),
        economic=EconomicResult(
            lifetime=20,
            maintenance_reference_cost=15000.0,
            maintenance_rate=0.015
        )
    )
    # ============================================================
    # Add Thermal Storage to the Heating System
    # ============================================================
    if scenario.use_thermal_storage:
        for system in [air_hp, ground_hp, gas_boiler]:
            system.thermal_storage = ThermalStorage(
                capacity_kWh= scenario.thermal_storage_capacity_kWh,
                charge_power_kW= scenario.thermal_storage_charge_power_kW,
                discharge_power_kW= scenario.thermal_storage_discharge_power_kW,
                eta_storage= scenario.thermal_storage_efficiency,
                soc_min= scenario.thermal_storage_soc_min,
                soc_max= scenario.thermal_storage_soc_max,
                loss_power_kW= scenario.thermal_storage_loss_power_kW
            )

    # ============================================================
    # Add Systems to Building
    # ============================================================
    building.add_heating_system(air_hp)
    building.add_heating_system(ground_hp)
    building.add_heating_system(gas_boiler)

    building.installed_system = deepcopy(gas_boiler)
    # ============================================================
    # Technical Models
    # ============================================================
    heat_pump_model = HeatPumpModel()
    gas_boiler_model = GasBoilerModel()
    investment_cost_model = InvestmentCostModel()
    emission_model = EmissionModel()
    economic_model = EconomicModel(discount_rate=0.03)
    transformation_decision = TransformationDecision()
    price_repository = EnergyPriceRepository()
    pv_model = PVModel()
    electricity_balance_model = ElectricityBalanceModel()

    if scenario.use_pv:
        building.pv_system = pv_model.calculate_lcoe(building.pv_system, economic_model)
        print("PV LOCE: ")
        print(building.pv_system)

    # ============================================================
    # Technical Simulation
    # ============================================================
    heat_pump_model.simulate(
        air_hp,
        heat_demand
    )

    heat_pump_model.simulate(
        ground_hp,
        heat_demand
    )
    gas_boiler_model.simulate(
        gas_boiler,
        heat_demand
    )

    for name, strategy in strategies.items():

        gas_boiler.operation_results[name] = (
            strategy.simulate(
                building,
                gas_boiler,
                heat_demand
            )
        )

        air_hp.operation_results[name] = (
            strategy.simulate(
                building,
                air_hp,
                heat_demand
            )
        )

        ground_hp.operation_results[name] = (
            strategy.simulate(
                building,
                ground_hp,
                heat_demand
            )
        )

        electricity_balance_model.calculate_hp_pv_balance(
            building,
            air_hp,
            air_hp.operation_results[name]
        )

        electricity_balance_model.calculate_hp_pv_balance(
            building,
            ground_hp,
            ground_hp.operation_results[name]
        )

        electricity_balance_model.calculate_hp_pv_balance(
            building,
            gas_boiler,
            gas_boiler.operation_results[name]
        )

        emission_model.calculate_emissions(
            air_hp,
            air_hp.operation_results[name],
            ef_electricity
        )

        emission_model.calculate_emissions(
            ground_hp,
            ground_hp.operation_results[name],
            ef_electricity
        )

        emission_model.calculate_emissions(
            gas_boiler,
            gas_boiler.operation_results[name],
            EF_GAS
        )

        for year in range(START_YEAR, END_YEAR + 1):
            # 1. Prices for each year
            prices = price_repository.get_energy_prices(year)

            electricity_price = prices[
                EnergyPriceRepository.HEADER_ELECTRICITY_PRICE
            ]
            gas_price = prices[
                EnergyPriceRepository.HEADER_GAS_PRICE
            ]
            co2_price = prices[
                EnergyPriceRepository.HEADER_CO2_PRICE
            ]

            # Invest model for the technologies
            investment_cost_model.evaluate(
                building,
                air_hp,
                year=year,
                start_year=START_YEAR
            )
            investment_cost_model.store_capex_history(
                air_hp.operation_results[name],
                air_hp,
                year
            )
            investment_cost_model.evaluate(
                building,
                ground_hp,
                year=year,
                start_year=START_YEAR
            )
            investment_cost_model.store_capex_history(
                ground_hp.operation_results[name],
                ground_hp,
                year
            )
            investment_cost_model.evaluate(
                building,
                gas_boiler,
                year=year,
                start_year=START_YEAR
            )
            investment_cost_model.store_capex_history(
                gas_boiler.operation_results[name],
                gas_boiler,
                year
            )
            economic_model.calculate_opex(
                air_hp,
                air_hp.operation_results[name],
                year,
                electricity_price,
                0.0
            )
            economic_model.calculate_opex(
                ground_hp,
                ground_hp.operation_results[name],
                year,
                electricity_price,
                0.0
            )
            economic_model.calculate_opex(
                gas_boiler,
                gas_boiler.operation_results[name],
                year,
                gas_price,
                co2_price
            )

            economic_model.evaluate(
                air_hp,
                air_hp.operation_results[name],
                year
            )
            economic_model.evaluate(
                ground_hp,
                ground_hp.operation_results[name],
                year
            )
            economic_model.evaluate(
                gas_boiler,
                gas_boiler.operation_results[name],
                year
            )

    decision_trigger_year = (
            START_YEAR
            + gas_boiler.state.remaining_lifetime
    )

    technology_history = []

    decision_history = []

    realized_history = []

    strategy_realized_paths = {
        strategy_name: []
        for strategy_name in strategies.keys()
    }

    strategy_decisions = []

    for strategy_name in strategies.keys():
        selected_system = (
            TransformationDecision
            .decide_by_strategy(
                building,
                strategy_name,
                decision_trigger_year
            )
        )

        selected_lcoh = (
            selected_system
            .operation_results[
                strategy_name
            ]
            .lcoh_history.loc[
                decision_trigger_year
            ]
        )

        strategy_decisions.append({

            "Strategy":
                strategy_name,

            "Decision_Year":
                decision_trigger_year,

            "Selected_System":
                selected_system.name,

            "Selected_Type":
                selected_system.system_type.name,

            "LCOH":
                round(
                    selected_lcoh,
                    4
                )

        })

    for strategy_name in strategies.keys():

        selected_system = (
            TransformationDecision
            .decide_by_strategy(
                building,
                strategy_name,
                decision_trigger_year
            )
        )

        for year in range(
                START_YEAR,
                END_YEAR + 1
        ):

            if year < decision_trigger_year:

                technology = (
                    gas_boiler.name
                )

            else:

                technology = (
                    selected_system.name
                )

            strategy_realized_paths[
                strategy_name
            ].append({

                "Strategy":
                    strategy_name,

                "Year":
                    year,

                "Technology":
                    technology

            })

    decision_df = pd.DataFrame(
        strategy_decisions
    )
    realized_df = pd.concat(

        [
            pd.DataFrame(path)

            for path in
            strategy_realized_paths.values()
        ],

        ignore_index=True

    )

    print()

    print("=" * 100)
    print("REALIZED TRANSFORMATION PATHS")
    print("=" * 100)

    print(
        realized_df.to_string(
            index=False
        )
    )

    print()

    print("=" * 100)
    print("TRANSFORMATION DECISIONS")
    print("=" * 100)

    print(
        decision_df.to_string(
            index=False
        )
    )

    OperationResultPlot.plot_operation_strategy_comparison(
        air_hp,
        heat_demand
    )

    EconomicResultPlot.plot_economic_strategy_comparison(building)

    OperationStrategy.plot_direct_heat(
        air_hp.operation_results["DirectHeat"],
        heat_demand

    )

    OperationStrategy.plot_heat_led(
        air_hp.operation_results["HeatLed"],
        heat_demand,
        air_hp.thermal_storage
    )



"""
    for name, result in gas_boiler.operation_results.items():
        print()
        print(name)

        print("PV Self:", result.W_el_pv_self)
        print("Grid Import:", result.W_el_grid)
        print("PV Export:", result.W_el_export)
        print("CO2:", result.CO2_emissions)
        print("Specific CO2:",result.CO2_specific)
        print("Gas: ",result.W_gas_generator)
        print("Capex: ",result.capex_ann_history)
        print("OPEX_Energy:", result.opex_energy_history)
        print("OPEX_Total:", result.opex_total_history)
        print("C_ann:",result.c_ann_history)
        print("LCOH:",result.lcoh_history)

    print ("Finish Operation")
    
    


    if scenario.use_pv:
        electricity_balance_model.calculate_hp_pv_balance(building, air_hp)
        electricity_balance_model.calculate_hp_pv_balance(building, ground_hp)
    else:
        air_hp.technical.W_el_grid = air_hp.technical.W_el
        air_hp.technical.W_el_PV_self = 0.0
        air_hp.technical.W_el_PV_export = 0.0

        ground_hp.technical.W_el_grid = ground_hp.technical.W_el
        ground_hp.technical.W_el_PV_self = 0.0
        ground_hp.technical.W_el_PV_export = 0.0

    print("Wel: ", air_hp.technical.W_el)
    print("W_el_grid: ", air_hp.technical.W_el_grid)
    print("W_el_PV_self: ", air_hp.technical.W_el_PV_self)

    # ============================================================
    # Initial Installed System
    # ============================================================

    # Important:
    # Create the copy AFTER technical simulation.
    building.installed_system = deepcopy(gas_boiler)
    building.installed_system.state.is_existing = (True)
    building.installed_system.state.investment_year = (None)

    # Existing plant is already paid.
    building.installed_system.economic.CAPEX_plant = (0.0)
    building.installed_system.economic.CAPEX_conversion = (0.0)
    building.installed_system.economic.CAPEX_additional = (0.0)
    building.installed_system.economic.CAPEX = (0.0)

    # ============================================================
    # Decision Year
    # ============================================================

    decision_trigger_year = (START_YEAR + gas_boiler.state.remaining_lifetime)
    # ============================================================
    # Histories
    # ============================================================
    technology_history = []
    decision_history = []
    realized_history = []

    for year in range(START_YEAR, END_YEAR + 1):
        # 1. Potential Investment Costs
        investment_cost_model.evaluate(
            building,
            air_hp,
            year=year,
            start_year=START_YEAR
        )
        investment_cost_model.evaluate(
            building,
            ground_hp,
            year=year,
            start_year=START_YEAR
        )
        investment_cost_model.evaluate(
            building,
            gas_boiler,
            year=year,
            start_year=START_YEAR
        )


        # --------------------------------------------------------
        # 3. Potential Technology Emissions
        # --------------------------------------------------------

        emission_model.calculate_emissions(
            air_hp,
            ef_electricity
        )

        emission_model.calculate_emissions(
            ground_hp,
            ef_electricity
        )

        emission_model.calculate_emissions(
            gas_boiler,
            EF_GAS
        )

        # ============================================================
        # 4. Potential Technology Economics
        # ============================================================
        if (scenario.use_pv and building.pv_system is not None):
            pv_lcoe = building.pv_system.LCOE
        else:
            pv_lcoe = 0.0
        economic_model.evaluate(
            air_hp,
            electricity_price,
            0,  # co2_price
            pv_lcoe=pv_lcoe
        )

        economic_model.evaluate(
            ground_hp,
            electricity_price,
            0,  # co2_price
            pv_lcoe=pv_lcoe

        )

        economic_model.evaluate(
            gas_boiler,
            gas_price,
            co2_price
        )

        # ============================================================
        # 5. Store Potential Technology Paths
        # ============================================================
        for system in building.heating_systems.values():
            technology_history.append({
                "Year": year,
                "System": system.name,
                "Type": system.system_type.name,
                # Technical
                "P_th_max_kW": system.technical.P_th_max / 1000,
                "W_th_kWh": system.technical.W_th / 1000,
                "W_el_kWh": system.technical.W_el / 1000,
                "W_gas_kWh": system.technical.W_gas / 1000,
                "PV Self Consumption [kWh]": system.technical.W_el_PV_self / 1000,
                "Grid Import [kWh]": system.technical.W_el_grid / 1000,

                "CAPEX_ann": system.economic.CAPEX_ann,
                "OPEX_energy": system.economic.OPEX_energy,
                "OPEX_maint": system.economic.OPEX_maint,
                "OPEX_CO2": system.economic.OPEX_CO2,
                "OPEX_total": system.economic.OPEX_total,

                "C_ann": system.economic.C_ann,
                "LCOH": system.economic.LCOH,
                # Environmental
                "CO2_emissions": system.environmental.CO2_emissions,
                "CO2_specific": system.environmental.CO2_specific
            })
        # --------------------------------------------------------
        # 6. Transformation Decision
        # --------------------------------------------------------
        if (year == decision_trigger_year and building.decision_year is None):

            previous_system_type = building.current_system_type
            selected_system = transformation_decision.decide(building)
            selected_system_type = selected_system.system_type
            transformed = selected_system_type != previous_system_type
            # ----------------------------------------------------
            # Store Decision
            # ----------------------------------------------------
            decision_history.append({

                "Year": year,
                "Previous_System": previous_system_type.name,
                "Selected_System": selected_system_type.name,
                "Decision_Made": True,
                "Transformed": transformed,
                "Reason": "Lowest_LCOH",
                "ASHP_LCOH": air_hp.economic.LCOH,
                "GSHP_LCOH": ground_hp.economic.LCOH,
                "GB_LCOH": gas_boiler.economic.LCOH,
                "ASHP_CAPEX": air_hp.economic.CAPEX,
                "GSHP_CAPEX": ground_hp.economic.CAPEX,
                "GB_CAPEX": gas_boiler.economic.CAPEX
            })
            # ----------------------------------------------------
            # Apply Decision
            # ----------------------------------------------------

            building.current_system_type = selected_system_type
            # Decision happened regardless of technology change
            building.decision_year = year
            building.is_transformed = transformed
            # transformation_year only means:
            # actual technology change
            if transformed:
                building.transformation_year = year

            # Most important point:
            # Freeze the actually selected installation.
            building.installed_system = deepcopy(selected_system)
            building.installed_system.state.is_existing = True
            building.installed_system.state.investment_year = year

        # --------------------------------------------------------
        # 7. Evaluate Realized System
        # --------------------------------------------------------
        current_system = building.installed_system
        if current_system is None:
            raise ValueError(
                "Building has no installed heating system."
            )
        # --------------------------------------------------------
        # Current-year emission factor
        # --------------------------------------------------------

        realized_emission_factor = (
            get_emission_factor_for_system(
                current_system,
                ef_electricity
            )
        )

        emission_model.calculate_emissions(
            current_system,
            realized_emission_factor
        )
        # --------------------------------------------------------
        # Current-year prices
        # --------------------------------------------------------

        realized_energy_price = (
            get_energy_price_for_system(
                current_system,
                electricity_price,
                gas_price
            )
        )

        realized_co2_price = (
            get_co2_price_for_system(
                current_system,
                co2_price
            )
        )
        # --------------------------------------------------------
        # Update operating economics only
        # CAPEX remains frozen at investment year.
        # --------------------------------------------------------
        economic_model.evaluate(
            current_system,
            realized_energy_price,
            realized_co2_price
        )
        # --------------------------------------------------------
        # 8. Store Realized Path
        # --------------------------------------------------------
        realized_history.append({
            "Year": year,
            "System": system.name,
            "Type": system.system_type.name,
            "Is_Transformed": building.is_transformed,
            "Transformation_Year": building.transformation_year,

            # Technical
            "P_th_max_kW": current_system.technical.P_th_max / 1000,
            "W_th_kWh": current_system.technical.W_th / 1000,
            "W_el_kWh": current_system.technical.W_el / 1000,
            "W_gas_kWh": current_system.technical.W_gas / 1000,
            # Economic
            "CAPEX_ann": system.economic.CAPEX_ann,
            "OPEX_energy": system.economic.OPEX_energy,
            "OPEX_maint": system.economic.OPEX_maint,
            "OPEX_CO2": system.economic.OPEX_CO2,
            "OPEX_total": system.economic.OPEX_total,

            "C_ann": system.economic.C_ann,
            "LCOH": system.economic.LCOH,

            # Environmental
            "CO2_emissions": current_system.environmental.CO2_emissions,
            "CO2_specific": current_system.environmental.CO2_specific
        })
    

    # ============================================================
    # Store Histories
    # ============================================================
    building.technology_history = (
        pd.DataFrame(
            technology_history
        )
    )

    building.decision_history = (
        pd.DataFrame(
            decision_history
        )
    )

    building.realized_history = (
        pd.DataFrame(
            realized_history
        )
    )
    # ============================================================
    # Sanity Checks
    # ============================================================

    expected_technology_rows = (
            (END_YEAR - START_YEAR + 1)
            * len(building.heating_systems)
    )

    expected_realized_rows = (
            END_YEAR
            - START_YEAR
            + 1
    )
    if (
            len(building.technology_history)
            != expected_technology_rows
    ):
        raise RuntimeError(
            "Unexpected number of rows "
            "in technology_history."
        )

    if (
            len(building.realized_history)
            != expected_realized_rows
    ):
        raise RuntimeError(
            "Unexpected number of rows "
            "in realized_history."
        )

    if (
            len(building.decision_history)
            != 1
    ):
        raise RuntimeError(
            "Expected exactly one decision "
            "in the current prototype."
        )

    # ============================================================
    # Output
    # ============================================================

    print()
    print("=" * 100)
    print("TECHNOLOGY HISTORY")
    print("=" * 100)

    print(
        building.technology_history
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("DECISION HISTORY")
    print("=" * 100)

    print(
        building.decision_history
        .to_string(
            index=False
        )
    )
    print()
    print("=" * 100)
    print("REALIZED HISTORY")
    print("=" * 100)

    print(
        building.realized_history
        .to_string(
            index=False
        )
    )
    # Potential technology paths
    building.plot_economic_development()
    building.plot_heating_operation(
        air_hp
    )
    
    
    print("Programm finish")
    """


