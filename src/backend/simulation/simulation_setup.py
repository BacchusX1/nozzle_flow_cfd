"""
Simulation Setup Module for SU2

Handles CFD simulation configuration, boundary conditions,
solver settings, and SU2 case generation.
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class BoundaryType(Enum):
    """Boundary condition types."""
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    FARFIELD = "farfield"


class SolverType(Enum):
    """Available SU2 solver types."""
    EULER = "EULER"
    NAVIER_STOKES = "NAVIER_STOKES"
    RANS = "RANS"
    INC_EULER = "INC_EULER"
    INC_NAVIER_STOKES = "INC_NAVIER_STOKES"
    INC_RANS = "INC_RANS"


class TurbulenceModelType(Enum):
    """Available turbulence models in SU2."""
    NONE = "NONE"
    SA = "SA"           # Spalart-Allmaras
    SA_NEG = "SA_NEG"   # Negative SA
    SST = "SST"         # Menter k-omega SST
    SST_SUST = "SST_SUST"  # SST with sustaining terms


class FluidModelType(Enum):
    """Fluid model types."""
    STANDARD_AIR = "STANDARD_AIR"
    IDEAL_GAS = "IDEAL_GAS"
    INC_IDEAL_GAS = "INC_IDEAL_GAS"
    CONSTANT_DENSITY = "CONSTANT_DENSITY"
    PR_GAS = "PR_GAS"  # Peng-Robinson EOS


def _is_compressible_solver(solver_type: SolverType) -> bool:
    """Check if solver is compressible."""
    return solver_type in (SolverType.EULER, SolverType.NAVIER_STOKES, SolverType.RANS)


def _is_viscous_solver(solver_type: SolverType) -> bool:
    """Check if solver includes viscous effects."""
    return solver_type in (
        SolverType.NAVIER_STOKES, SolverType.RANS,
        SolverType.INC_NAVIER_STOKES, SolverType.INC_RANS
    )


def _is_rans_solver(solver_type: SolverType) -> bool:
    """Check if solver is RANS."""
    return solver_type in (SolverType.RANS, SolverType.INC_RANS)


@dataclass
class FluidProperties:
    """Fluid properties for simulation.
    
    For nozzle flow with MARKER_INLET/OUTLET BCs, these serve as:
    - Initial conditions for the flow field
    - Reference state for non-dimensionalization
    - Freestream values (used for coefficient calculations)
    
    Set pressure/temperature close to expected average flow conditions
    for better convergence.
    """
    density: float = 1.225              # kg/m³ (computed from P, T if ideal gas)
    viscosity: float = 1.81e-5          # Pa·s (dynamic viscosity)
    temperature: float = 300.0          # K (freestream/reference temperature)
    pressure: float = 300000.0          # Pa (freestream/reference pressure - between inlet and outlet)
    gas_constant: float = 287.058       # J/(kg·K) for air
    gamma: float = 1.4                  # Ratio of specific heats
    specific_heat_cp: float = 1004.5    # J/(kg·K)
    prandtl_number: float = 0.72
    prandtl_turb: float = 0.9


@dataclass
class BoundaryCondition:
    """Boundary condition definition."""
    name: str
    boundary_type: BoundaryType
    
    # Inlet conditions (for MARKER_INLET) - De Laval nozzle defaults
    # For compressible flow: specify TOTAL (stagnation) conditions, NOT velocity
    # The solver computes velocity from pressure ratio automatically
    total_temperature: float = 300.0    # K (stagnation/total temperature)
    total_pressure: float = 500000.0    # Pa (5 bar stagnation pressure)
    flow_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    
    # Velocity inlet (only for incompressible solvers - not used for nozzle flow)
    velocity_magnitude: float = 100.0   # m/s (incompressible only)
    
    # Outlet conditions (for MARKER_OUTLET)
    # Back pressure controls shock location and flow regime
    # Lower pressure = more expansion, higher exit Mach
    static_pressure: float = 50000.0    # Pa (0.5 bar for supersonic expansion)
    
    # Wall conditions
    wall_temperature: float = 300.0     # K (for isothermal)
    heat_flux: float = 0.0              # W/m² (for heat flux BC)
    is_adiabatic: bool = True
    
    # Legacy compatibility fields (used by GUI)
    pressure_value: float = 0.0         # Alias for static_pressure
    turbulence_intensity: float = 0.05
    turbulence_length_scale: float = 0.1


@dataclass
class SolverSettings:
    """Solver configuration for SU2."""
    solver_type: SolverType = SolverType.EULER  # EULER for inviscid nozzle flow
    
    # Simulation mode
    is_transient: bool = False          # True for unsteady, False for steady-state
    
    # Flow conditions (de Laval nozzle defaults)
    mach_number: float = 0.3            # Subsonic inlet Mach
    reynolds_number: float = 1e6
    angle_of_attack: float = 0.0
    sideslip_angle: float = 0.0
    
    # Iteration settings
    max_iterations: int = 500           # Sufficient for steady convergence
    convergence_residual: float = -8    # Log10 of residual target (achievable)
    convergence_tolerance: float = 1e-6  # Legacy compatibility
    
    # Time stepping (used when is_transient=True)
    time_step: float = 1e-5             # Physical time step for unsteady
    end_time: float = 0.1              # Simulation end time (10ms for nozzle startup)
    inner_iterations: int = 20          # Inner iterations per time step (dual time-stepping)
    cfl_number: float = 5.0             # CFL for steady (lower for stability)
    cfl_adapt: bool = False
    cfl_min: float = 0.1
    cfl_max: float = 100.0
    max_courant: float = 5.0            # Legacy compatibility alias
    
    # Numerical schemes
    conv_num_method: str = "ROE"        # JST, LAX-FRIEDRICH, ROE, AUSM, HLLC, etc.
    muscl: bool = True
    slope_limiter: str = "VENKATAKRISHNAN"
    time_discre: str = "RUNGE-KUTTA_EXPLICIT" # EULER_IMPLICIT, EULER_EXPLICIT, RUNGE-KUTTA_EXPLICIT
    gradient_method: str = "GREEN_GAUSS"  # GREEN_GAUSS or WEIGHTED_LEAST_SQUARES
    
    # Linear solver
    linear_solver: str = "FGMRES"
    linear_solver_prec: str = "ILU"
    linear_solver_iter: int = 10
    linear_solver_error: float = 1e-6
    
    # Multigrid
    mglevel: int = 3
    mgcycle: str = "W_CYCLE"
    
    # Output settings
    output_frequency: int = 50           # Write every 50 iters (~10 files for 500 iters)
    write_interval: int = 50            # Legacy compatibility
    
    # Parallel processing
    n_processors: int = 6
    decomposition_method: str = "METIS" # SU2 uses METIS internally
    
    # PIMPLE-style settings (legacy compatibility, mapped to CFL)
    n_outer_correctors: int = 2
    n_correctors: int = 2
    n_non_orthogonal_correctors: int = 0
    adjust_time_step: bool = False


@dataclass
class TurbulenceModel:
    """Turbulence model settings (legacy-compatible interface)."""
    enabled: bool = True
    model_type: str = "kEpsilon"             # SA, SST, kOmegaSST, kEpsilon, SpalartAllmaras
    wall_functions: bool = True
    
    # Freestream turbulence
    freestream_turbulence_intensity: float = 0.05
    freestream_turb2lam_ratio: float = 10.0


class SimulationSetup:
    """Main simulation setup and SU2 case generation."""
    
    def __init__(self):
        self.fluid_properties = FluidProperties()
        self.boundary_conditions: Dict[str, BoundaryCondition] = {}
        self.solver_settings = SolverSettings()
        self.turbulence_model = TurbulenceModel()
        self.case_directory = ""
        self.mesh_data = None
        self.geometry = None
        self.config_filename = "config.cfg"
        self.mesh_filename = "mesh.su2"
        
    def add_boundary_condition(self, bc: BoundaryCondition):
        """Add or update boundary condition."""
        self.boundary_conditions[bc.name] = bc
        
    def remove_boundary_condition(self, name: str):
        """Remove boundary condition."""
        if name in self.boundary_conditions:
            del self.boundary_conditions[name]
            
    def get_boundary_condition(self, name: str) -> Optional[BoundaryCondition]:
        """Get boundary condition by name."""
        return self.boundary_conditions.get(name)
        
    def set_inlet_conditions(self, boundary_name: str, velocity: float = None,
                            total_pressure: float = None, total_temperature: float = None,
                            direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                            turbulence_intensity: float = 0.05):
        """Set inlet boundary conditions."""
        bc = BoundaryCondition(
            name=boundary_name,
            boundary_type=BoundaryType.INLET,
            velocity_magnitude=velocity or 0.0,
            total_pressure=total_pressure or self.fluid_properties.pressure,
            total_temperature=total_temperature or self.fluid_properties.temperature,
            flow_direction=direction,
            turbulence_intensity=turbulence_intensity
        )
        self.add_boundary_condition(bc)
        
    def set_outlet_conditions(self, boundary_name: str, pressure: float = None):
        """Set outlet boundary conditions."""
        static_p = pressure if pressure is not None else self.fluid_properties.pressure * 0.8
        bc = BoundaryCondition(
            name=boundary_name,
            boundary_type=BoundaryType.OUTLET,
            static_pressure=static_p,
            pressure_value=static_p
        )
        self.add_boundary_condition(bc)
        
    def set_wall_conditions(self, boundary_name: str, roughness: float = 0.0,
                           is_adiabatic: bool = True, wall_temperature: float = None):
        """Set wall boundary conditions."""
        bc = BoundaryCondition(
            name=boundary_name,
            boundary_type=BoundaryType.WALL,
            is_adiabatic=is_adiabatic,
            wall_temperature=wall_temperature or self.fluid_properties.temperature,
            velocity_magnitude=0.0  # No-slip
        )
        self.add_boundary_condition(bc)

    def set_symmetry_conditions(self, boundary_name: str):
        """Set symmetry boundary conditions."""
        bc = BoundaryCondition(
            name=boundary_name,
            boundary_type=BoundaryType.SYMMETRY
        )
        self.add_boundary_condition(bc)
        
    def generate_case_files(self, geometry, mesh_data: Dict = None):
        """Generate SU2 case files from geometry and mesh data.
        
        This method is called by the GUI.
        """
        if not self.case_directory:
            raise ValueError("Case directory not set")
            
        # Store geometry and mesh data
        self.geometry = geometry
        self.mesh_data = mesh_data
        
        # Generate the SU2 case
        return self.generate_su2_case(self.case_directory, mesh_data)
    
    def generate_su2_case(self, case_directory: str, mesh_data: Dict = None):
        """Generate complete SU2 case."""
        self.case_directory = case_directory
        self.mesh_data = mesh_data
        
        # Create directory
        os.makedirs(case_directory, exist_ok=True)
        
        # Generate mesh file
        if mesh_data:
            self._write_mesh_file()
        
        # Generate main configuration file
        self._write_config_file()
        
        return case_directory
    
    def _get_turbulence_model_type(self) -> str:
        """Map legacy turbulence model names to SU2 types."""
        if not self.turbulence_model.enabled:
            return "NONE"
        
        # Handle both enum and string types for model_type
        model_type = self.turbulence_model.model_type
        if isinstance(model_type, TurbulenceModelType):
            model = model_type.value.upper()
        else:
            model = str(model_type).upper()
        
        # Map OpenFOAM-style names to SU2
        mapping = {
            "KOMEGASST": "SST",
            "KOMEGA": "SST",
            "KEPSILON": "SST",  # SU2 doesn't have k-epsilon, use SST
            "SPALARTALLMARAS": "SA",
            "SA": "SA",
            "SST": "SST",
            "LAMINAR": "NONE",
        }
        
        return mapping.get(model, "SST")
    
    def _is_compressible(self) -> bool:
        """Check if current solver is compressible."""
        return _is_compressible_solver(self.solver_settings.solver_type)
    
    def _is_rans(self) -> bool:
        """Check if current solver is RANS."""
        return _is_rans_solver(self.solver_settings.solver_type) and self.turbulence_model.enabled
    
    def _write_mesh_file(self):
        """Write mesh in SU2 format."""
        if not self.mesh_data:
            return
            
        # No fallback: if converter is missing or fails, it should raise error
        from backend.meshing.su2_mesh_converter import SU2MeshConverter
        
        converter = SU2MeshConverter()
        converter.load_from_mesh_data(self.mesh_data)
        
        # Detect boundaries from geometry if not provided
        if not converter.boundaries:
            converter.detect_boundaries_from_geometry(self.geometry)
        
        mesh_path = os.path.join(self.case_directory, self.mesh_filename)
        converter.write_su2_mesh(mesh_path)
    
    def _write_config_file(self):
        """Write SU2 configuration file."""
        config_path = os.path.join(self.case_directory, self.config_filename)
        
        turb_model = self._get_turbulence_model_type()
        is_rans = turb_model != "NONE" and self._is_rans()
        is_compressible = self._is_compressible()
        
        lines = []
        lines.append("% ================== SU2 CONFIGURATION FILE ==================")
        lines.append("% Generated by Nozzle CFD Design Tool")
        lines.append("%")
        
        # Problem definition
        lines.append("")
        lines.append("% -------------------- PROBLEM DEFINITION --------------------")
        lines.append(f"SOLVER= {self.solver_settings.solver_type.value}")
        lines.append("MATH_PROBLEM= DIRECT")
        lines.append("RESTART_SOL= NO")
        
        # Time domain settings (steady vs transient)
        lines.append("")
        lines.append("% -------------------- TIME DOMAIN ----------------------------")
        if self.solver_settings.is_transient:
            lines.append("TIME_DOMAIN= YES")
            lines.append("TIME_MARCHING= DUAL_TIME_STEPPING-2ND_ORDER")
            lines.append(f"TIME_STEP= {self.solver_settings.time_step}")
            lines.append(f"MAX_TIME= {self.solver_settings.end_time}")
            # Calculate number of time steps from end_time and time_step
            time_iter = int(self.solver_settings.end_time / self.solver_settings.time_step) + 1
            lines.append(f"TIME_ITER= {max(time_iter, self.solver_settings.max_iterations)}")
            lines.append(f"INNER_ITER= {self.solver_settings.inner_iterations}")
        else:
            lines.append("TIME_DOMAIN= NO")
        
        # Fluid model
        lines.append("")
        lines.append("% -------------------- FLUID MODEL ----------------------------")
        if is_compressible:
            lines.append("FLUID_MODEL= IDEAL_GAS")
            lines.append(f"GAMMA_VALUE= {self.fluid_properties.gamma}")
            lines.append(f"GAS_CONSTANT= {self.fluid_properties.gas_constant}")
            lines.append("CRITICAL_TEMPERATURE= 131.0")
            lines.append("CRITICAL_PRESSURE= 3588550.0")
            lines.append("ACENTRIC_FACTOR= 0.035")
        else:
            lines.append("FLUID_MODEL= CONSTANT_DENSITY")
            lines.append("INC_DENSITY_MODEL= CONSTANT")
            lines.append(f"INC_DENSITY_INIT= {self.fluid_properties.density}")
        
        # Viscosity model
        if _is_viscous_solver(self.solver_settings.solver_type):
            lines.append("")
            lines.append("% -------------------- VISCOSITY MODEL -----------------------")
            lines.append("VISCOSITY_MODEL= SUTHERLAND")
            lines.append("MU_REF= 1.716e-5")
            lines.append("MU_T_REF= 273.15")
            lines.append("SUTHERLAND_CONSTANT= 110.4")
            lines.append(f"PRANDTL_LAM= {self.fluid_properties.prandtl_number}")
            lines.append(f"PRANDTL_TURB= {self.fluid_properties.prandtl_turb}")
        
        # Freestream conditions
        lines.append("")
        lines.append("% -------------------- FREE-STREAM DEFINITION -----------------")
        if is_compressible:
            lines.append(f"MACH_NUMBER= {self.solver_settings.mach_number}")
            lines.append(f"AOA= {self.solver_settings.angle_of_attack}")
            lines.append(f"SIDESLIP_ANGLE= {self.solver_settings.sideslip_angle}")
            lines.append(f"FREESTREAM_PRESSURE= {self.fluid_properties.pressure}")
            lines.append(f"FREESTREAM_TEMPERATURE= {self.fluid_properties.temperature}")
            # Reynolds number is REQUIRED for all viscous solvers (NAVIER_STOKES and RANS)
            if _is_viscous_solver(self.solver_settings.solver_type):
                lines.append(f"REYNOLDS_NUMBER= {self.solver_settings.reynolds_number}")
                lines.append("REYNOLDS_LENGTH= 1.0")
                # Use thermodynamic conditions (P, T) for initialization instead of 
                # computing density from Reynolds number (default REYNOLDS behavior)
                # This ensures FREESTREAM_PRESSURE is actually used for initialization
                lines.append("INIT_OPTION= TD_CONDITIONS")
        else:
            lines.append("INC_VELOCITY_INIT= (1.0, 0.0, 0.0)")
            lines.append(f"INC_TEMPERATURE_INIT= {self.fluid_properties.temperature}")
            lines.append("INC_NONDIM= DIMENSIONAL")
        
        # Turbulence model
        if is_rans:
            lines.append("")
            lines.append("% -------------------- TURBULENCE MODEL ----------------------")
            lines.append(f"KIND_TURB_MODEL= {turb_model}")
            lines.append(f"FREESTREAM_TURBULENCEINTENSITY= {self.turbulence_model.freestream_turbulence_intensity}")
            lines.append(f"FREESTREAM_TURB2LAMVISCRATIO= {self.turbulence_model.freestream_turb2lam_ratio}")
        
        # Reference values
        lines.append("")
        lines.append("% -------------------- REFERENCE VALUES ------------------------")
        lines.append("REF_ORIGIN_MOMENT_X= 0.0")
        lines.append("REF_ORIGIN_MOMENT_Y= 0.0")
        lines.append("REF_ORIGIN_MOMENT_Z= 0.0")
        lines.append("REF_LENGTH= 1.0")
        lines.append("REF_AREA= 1.0")
        lines.append("REF_DIMENSIONALIZATION= DIMENSIONAL")
        
        # Boundary markers
        lines.append("")
        lines.append("% -------------------- BOUNDARY MARKERS ------------------------")
        self._write_boundary_markers(lines)
        
        # Numerical method
        lines.append("")
        lines.append("% -------------------- NUMERICAL METHOD ------------------------")
        lines.append(f"NUM_METHOD_GRAD= {self.solver_settings.gradient_method}")
        lines.append(f"CFL_NUMBER= {self.solver_settings.cfl_number}")
        if self.solver_settings.cfl_adapt:
            lines.append("CFL_ADAPT= YES")
            lines.append(f"CFL_ADAPT_PARAM= (0.5, 2.0, {self.solver_settings.cfl_min}, {self.solver_settings.cfl_max})")
        else:
            lines.append("CFL_ADAPT= NO")
        lines.append("MAX_DELTA_TIME= 1E10")
        
        # Convective scheme
        lines.append("")
        lines.append("% -------------------- CONVECTIVE SCHEME -----------------------")
        lines.append(f"CONV_NUM_METHOD_FLOW= {self.solver_settings.conv_num_method}")
        # Centered schemes (JST, LAX-FRIEDRICH) don't use MUSCL reconstruction
        centered_schemes = ["JST", "LAX-FRIEDRICH", "LAX_FRIEDRICH"]
        use_muscl = self.solver_settings.muscl and self.solver_settings.conv_num_method.upper() not in centered_schemes
        lines.append(f"MUSCL_FLOW= {'YES' if use_muscl else 'NO'}")
        lines.append(f"SLOPE_LIMITER_FLOW= {self.solver_settings.slope_limiter}")
        lines.append(f"TIME_DISCRE_FLOW= {self.solver_settings.time_discre}")
        
        if is_rans:
            lines.append("")
            lines.append("CONV_NUM_METHOD_TURB= SCALAR_UPWIND")
            lines.append("MUSCL_TURB= NO")
            lines.append("SLOPE_LIMITER_TURB= VENKATAKRISHNAN")
            lines.append("TIME_DISCRE_TURB= EULER_IMPLICIT")
        
        # Linear solver
        lines.append("")
        lines.append("% -------------------- LINEAR SOLVER --------------------------")
        lines.append(f"LINEAR_SOLVER= {self.solver_settings.linear_solver}")
        lines.append(f"LINEAR_SOLVER_PREC= {self.solver_settings.linear_solver_prec}")
        lines.append(f"LINEAR_SOLVER_ERROR= {self.solver_settings.linear_solver_error}")
        lines.append(f"LINEAR_SOLVER_ITER= {self.solver_settings.linear_solver_iter}")
        
        # Multigrid
        lines.append("")
        lines.append("% -------------------- MULTIGRID ------------------------------")
        lines.append(f"MGLEVEL= {self.solver_settings.mglevel}")
        lines.append(f"MGCYCLE= {self.solver_settings.mgcycle}")
        lines.append("MG_PRE_SMOOTH= (1, 2, 3, 3)")
        lines.append("MG_POST_SMOOTH= (0, 0, 0, 0)")
        lines.append("MG_CORRECTION_SMOOTH= (0, 0, 0, 0)")
        lines.append("MG_DAMP_RESTRICTION= 0.75")
        lines.append("MG_DAMP_PROLONGATION= 0.75")
        
        # Convergence
        lines.append("")
        lines.append("% -------------------- CONVERGENCE CRITERIA -------------------")
        # ITER should only be used for steady-state simulations
        # For transient, TIME_ITER and INNER_ITER are used instead
        if not self.solver_settings.is_transient:
            lines.append(f"ITER= {self.solver_settings.max_iterations}")
        lines.append(f"CONV_RESIDUAL_MINVAL= {self.solver_settings.convergence_residual}")
        lines.append("CONV_STARTITER= 10")
        lines.append("CONV_CAUCHY_ELEMS= 100")
        lines.append("CONV_CAUCHY_EPS= 1E-6")
        
        # Input/Output
        lines.append("")
        lines.append("% -------------------- INPUT/OUTPUT ---------------------------")
        lines.append(f"MESH_FILENAME= {self.mesh_filename}")
        lines.append("MESH_FORMAT= SU2")
        lines.append("SOLUTION_FILENAME= restart_flow.dat")
        # Include RESTART in output files so restart files are written for multi-phase simulations
        lines.append("OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW, SURFACE_CSV)")
        lines.append("RESTART_FILENAME= restart_flow")
        # For unsteady simulations, overwrite restart file each time step (latest state for restart)
        if self.solver_settings.is_transient:
            lines.append("WRT_RESTART_OVERWRITE= YES")
        lines.append("TABULAR_FORMAT= CSV")
        lines.append("CONV_FILENAME= history")
        lines.append("HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF, FLOW_COEFF, FLOW_COEFF_SURF)")
        lines.append("VOLUME_FILENAME= flow")
        lines.append("SURFACE_FILENAME= surface_flow")
        lines.append(f"OUTPUT_WRT_FREQ= {self.solver_settings.output_frequency}")
        lines.append("SCREEN_OUTPUT= (INNER_ITER, WALL_TIME, RMS_DENSITY, RMS_MOMENTUM-X, RMS_ENERGY, LIFT, DRAG)")
        
        # Write config file
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            f.write('\n')
    
    def _write_boundary_markers(self, lines: List[str]):
        """Write boundary marker definitions to config lines."""
        
        # Collect markers by type
        inlet_markers = []
        outlet_markers = []
        wall_markers = []
        symmetry_markers = []
        farfield_markers = []
        
        for bc_name, bc in self.boundary_conditions.items():
            if bc.boundary_type == BoundaryType.INLET:
                inlet_markers.append(bc)
            elif bc.boundary_type == BoundaryType.OUTLET:
                outlet_markers.append(bc)
            elif bc.boundary_type == BoundaryType.WALL:
                wall_markers.append(bc)
            elif bc.boundary_type == BoundaryType.SYMMETRY:
                symmetry_markers.append(bc)
            elif bc.boundary_type == BoundaryType.FARFIELD:
                farfield_markers.append(bc)
        
        # Write inlet markers
        if inlet_markers:
            if self._is_compressible():
                # MARKER_INLET= (name, Ttotal, Ptotal, nx, ny, nz, ...)
                inlet_strs = []
                for bc in inlet_markers:
                    inlet_strs.append(f"{bc.name}, {bc.total_temperature}, {bc.total_pressure}, "
                                    f"{bc.flow_direction[0]}, {bc.flow_direction[1]}, {bc.flow_direction[2]}")
                lines.append(f"MARKER_INLET= ({', '.join(inlet_strs)})")
            else:
                # Incompressible: MARKER_INLET= (name, velocity, ...)
                inlet_strs = []
                for bc in inlet_markers:
                    inlet_strs.append(f"{bc.name}, {bc.velocity_magnitude}, "
                                    f"{bc.flow_direction[0]}, {bc.flow_direction[1]}, {bc.flow_direction[2]}")
                lines.append(f"MARKER_INLET= ({', '.join(inlet_strs)})")
        
        # Write outlet markers
        if outlet_markers:
            outlet_strs = [f"{bc.name}, {bc.static_pressure}" for bc in outlet_markers]
            lines.append(f"MARKER_OUTLET= ({', '.join(outlet_strs)})")
        
        # Write wall markers
        # For EULER (inviscid) solver, use MARKER_EULER (slip wall)
        # For viscous solvers (NAVIER_STOKES, RANS), use MARKER_HEATFLUX or MARKER_ISOTHERMAL
        is_viscous = _is_viscous_solver(self.solver_settings.solver_type)
        
        if wall_markers:
            if is_viscous:
                # Viscous solver: use heat flux or isothermal walls
                adiabatic_walls = [bc for bc in wall_markers if bc.is_adiabatic]
                isothermal_walls = [bc for bc in wall_markers if not bc.is_adiabatic]
                
                if adiabatic_walls:
                    heatflux_strs = [f"{bc.name}, 0.0" for bc in adiabatic_walls]
                    lines.append(f"MARKER_HEATFLUX= ({', '.join(heatflux_strs)})")
                
                if isothermal_walls:
                    iso_strs = [f"{bc.name}, {bc.wall_temperature}" for bc in isothermal_walls]
                    lines.append(f"MARKER_ISOTHERMAL= ({', '.join(iso_strs)})")
            else:
                # Inviscid solver (EULER): use slip wall (MARKER_EULER)
                wall_names = ", ".join(bc.name for bc in wall_markers)
                lines.append(f"MARKER_EULER= ({wall_names})")
        
        # Write symmetry markers
        if symmetry_markers:
            sym_names = ", ".join(bc.name for bc in symmetry_markers)
            lines.append(f"MARKER_SYM= ({sym_names})")
        
        # Write farfield markers
        if farfield_markers:
            ff_names = ", ".join(bc.name for bc in farfield_markers)
            lines.append(f"MARKER_FAR= ({ff_names})")
        
        # Monitoring markers - monitor wall for forces
        if wall_markers:
            lines.append(f"MARKER_MONITORING= ({wall_markers[0].name})")
        elif inlet_markers:
            lines.append(f"MARKER_MONITORING= ({inlet_markers[0].name})")
        
        # Analyze markers - for flow quantities at inlet/outlet
        analyze_markers = []
        if inlet_markers:
            analyze_markers.extend(bc.name for bc in inlet_markers)
        if outlet_markers:
            analyze_markers.extend(bc.name for bc in outlet_markers)
        if analyze_markers:
            lines.append(f"MARKER_ANALYZE= ({', '.join(analyze_markers)})")
            lines.append("MARKER_ANALYZE_AVERAGE= MASSFLUX")
        
        # Plotting markers
        all_markers = [bc.name for bc in self.boundary_conditions.values()]
        if all_markers:
            lines.append(f"MARKER_PLOTTING= ({', '.join(all_markers)})")
    
    def save_configuration(self, filename: str):
        """Save simulation configuration to JSON file."""
        config = {
            'fluid_properties': asdict(self.fluid_properties),
            'boundary_conditions': {
                name: {**asdict(bc), 'boundary_type': bc.boundary_type.value}
                for name, bc in self.boundary_conditions.items()
            },
            'solver_settings': {
                **asdict(self.solver_settings),
                'solver_type': self.solver_settings.solver_type.value
            },
            'turbulence_model': asdict(self.turbulence_model)
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
    def load_configuration(self, filename: str):
        """Load simulation configuration from JSON file."""
        with open(filename, 'r') as f:
            config = json.load(f)
            
        # Restore fluid properties
        self.fluid_properties = FluidProperties(**config['fluid_properties'])
        
        # Restore solver settings
        solver_data = config['solver_settings']
        solver_data['solver_type'] = SolverType(solver_data['solver_type'])
        self.solver_settings = SolverSettings(**solver_data)
        
        # Restore turbulence model
        self.turbulence_model = TurbulenceModel(**config['turbulence_model'])
        
        # Restore boundary conditions
        self.boundary_conditions.clear()
        for name, bc_data in config['boundary_conditions'].items():
            bc_data['boundary_type'] = BoundaryType(bc_data['boundary_type'])
            if 'flow_direction' in bc_data and isinstance(bc_data['flow_direction'], list):
                bc_data['flow_direction'] = tuple(bc_data['flow_direction'])
            bc = BoundaryCondition(**bc_data)
            self.boundary_conditions[name] = bc
