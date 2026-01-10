"""
Standard Values Loader for Nozzle CFD Design Tool

Loads default values from standard_values_gui.yml for use in the GUI.
This allows users to customize default values without modifying code.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def _load_standard_values() -> Dict[str, Any]:
    """Load standard values from standard_values_gui.yml file."""
    values = {}
    
    # Look for standard_values_gui.yml in common locations
    search_paths = [
        Path(__file__).parent.parent.parent / "standard_values_gui.yml",  # project root
        Path.cwd() / "standard_values_gui.yml",  # current working directory
        Path.home() / ".nozzle_cfd" / "standard_values_gui.yml",  # user home
    ]
    
    for values_path in search_paths:
        if values_path.exists():
            # Strict loading: propagate exceptions if file exists but fails to parse
            # User requirement: "dont downgrade to standards in case of error"
            with open(values_path, 'r') as f:
                values = yaml.safe_load(f) or {}
            print(f"Loaded standard values from: {values_path}")
            break
    
    return values


# Load values at module import time
_STANDARD_VALUES = _load_standard_values()


class StandardValues:
    """
    Container for standard/default values used throughout the GUI.
    
    Values can be customized via standard_values_gui.yml at the project root.
    
    Usage:
        from core.standard_values import StandardValues
        
        # Get a value with dot notation path
        inlet_pressure = StandardValues.get('boundary_conditions.inlet.total_pressure', 500000)
        
        # Or access pre-defined properties
        inlet_pressure = StandardValues.inlet_total_pressure
    """
    
    _values = _STANDARD_VALUES
    
    @classmethod
    def get(cls, path: str, default: Any = None) -> Any:
        """
        Get a value by dot-notation path.
        
        Args:
            path: Dot-separated path like 'boundary_conditions.inlet.total_pressure'
            default: Default value if path not found
            
        Returns:
            The value at the path, or default if not found
        """
        keys = path.split('.')
        value = cls._values
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    @classmethod
    def reload(cls):
        """Reload values from file (useful for runtime updates)."""
        cls._values = _load_standard_values()
    
    # -------------------------------------------------------------------------
    # Geometry Settings
    # -------------------------------------------------------------------------
    
    @property
    def interpolation_points(self) -> int:
        return self.get('geometry.interpolation_points', 100)
    
    @property
    def min_throat_ratio(self) -> float:
        return self.get('geometry.min_throat_ratio', 0.5)
    
    @property
    def max_divergence_angle(self) -> float:
        return self.get('geometry.max_divergence_angle', 20)
    
    # -------------------------------------------------------------------------
    # Mesh Settings
    # -------------------------------------------------------------------------
    
    @property
    def mesh_type(self) -> str:
        return self.get('mesh.mesh_type', 'hex')
    
    @property
    def global_element_size(self) -> float:
        return self.get('mesh.global_element_size', 0.02)
    
    @property
    def min_element_size(self) -> float:
        return self.get('mesh.min_element_size', 0.001)
    
    @property
    def max_element_size(self) -> float:
        return self.get('mesh.max_element_size', 0.05)
    
    @property
    def wall_element_size(self) -> float:
        return self.get('mesh.wall_element_size', 0.005)
    
    @property
    def nozzle_refinement_enabled(self) -> bool:
        return self.get('mesh.nozzle_refinement.enabled', True)
    
    @property
    def nozzle_refinement_cell_size(self) -> float:
        return self.get('mesh.nozzle_refinement.cell_size', 0.002)
    
    @property
    def nozzle_refinement_growth_rate(self) -> float:
        return self.get('mesh.nozzle_refinement.growth_rate', 1.15)
    
    @property
    def boundary_layer_enabled(self) -> bool:
        return self.get('mesh.boundary_layer.enabled', True)
    
    @property
    def num_boundary_layers(self) -> int:
        return self.get('mesh.boundary_layer.num_layers', 8)
    
    @property
    def first_layer_thickness(self) -> float:
        return self.get('mesh.boundary_layer.first_layer_thickness', 1e-5)
    
    @property
    def growth_ratio(self) -> float:
        return self.get('mesh.boundary_layer.growth_ratio', 1.2)
    
    @property
    def quality_threshold(self) -> float:
        return self.get('mesh.quality_threshold', 0.3)
    
    @property
    def smoothing_iterations(self) -> int:
        return self.get('mesh.smoothing_iterations', 5)
    
    @property
    def farfield_distance(self) -> float:
        return self.get('mesh.farfield_distance', 5.0)
    
    # -------------------------------------------------------------------------
    # Boundary Conditions
    # -------------------------------------------------------------------------
    
    @property
    def inlet_total_pressure(self) -> float:
        return self.get('boundary_conditions.inlet.total_pressure', 500000)
    
    @property
    def inlet_total_temperature(self) -> float:
        return self.get('boundary_conditions.inlet.total_temperature', 300)
    
    @property
    def inlet_velocity(self) -> float:
        return self.get('boundary_conditions.inlet.velocity_magnitude', 100.0)
    
    @property
    def outlet_static_pressure(self) -> float:
        return self.get('boundary_conditions.outlet.static_pressure', 50000)
    
    @property
    def wall_temperature(self) -> float:
        return self.get('boundary_conditions.wall.temperature', 300)
    
    @property
    def wall_is_adiabatic(self) -> bool:
        return self.get('boundary_conditions.wall.is_adiabatic', True)
    
    # -------------------------------------------------------------------------
    # Fluid Properties
    # -------------------------------------------------------------------------
    
    @property
    def gas_constant(self) -> float:
        return self.get('fluid.gas_constant', 287.058)
    
    @property
    def gamma(self) -> float:
        return self.get('fluid.gamma', 1.4)
    
    @property
    def reference_pressure(self) -> float:
        # Auto-calculate from outlet pressure if not explicitly set
        override = self.get('fluid.reference_pressure', None)
        if override is not None:
            return override
        # Initialize domain at outlet (back) pressure - physically motivated:
        # The domain starts at rest with back pressure, then inlet pressure
        # drives the flow and shocks form naturally where physics requires.
        return self.outlet_static_pressure
    
    @property
    def reference_temperature(self) -> float:
        # Auto-calculate from inlet temperature if not explicitly set
        override = self.get('fluid.reference_temperature', None)
        if override is not None:
            return override
        return self.inlet_total_temperature
    
    @property
    def reference_density(self) -> float:
        # Auto-calculate from reference P, T using ideal gas law if not explicitly set
        override = self.get('fluid.reference_density', None)
        if override is not None:
            return override
        # ρ = P / (R * T)
        return self.reference_pressure / (self.gas_constant * self.reference_temperature)
    
    # -------------------------------------------------------------------------
    # Initialization Settings
    # -------------------------------------------------------------------------
    
    @property
    def init_method(self) -> str:
        return self.get('initialization.method', 'outlet_pressure')
    
    @property
    def init_velocity(self) -> float:
        return self.get('initialization.velocity', 1.0)
    
    @property
    def init_temperature(self) -> float:
        return self.get('initialization.temperature', 300.0)
    
    @property
    def init_custom_pressure(self) -> float:
        return self.get('initialization.custom_pressure', 101325.0)
    
    @property
    def init_custom_density(self) -> float:
        return self.get('initialization.custom_density', 1.225)
    
    @property
    def init_pressure(self) -> float:
        """Get initialization pressure based on selected method."""
        method = self.init_method
        if method == 'outlet_pressure':
            return self.outlet_static_pressure
        elif method == 'inlet_pressure':
            return self.inlet_total_pressure
        elif method == 'average_pressure':
            return (self.inlet_total_pressure + self.outlet_static_pressure) / 2.0
        elif method == 'custom':
            return self.init_custom_pressure
        else:
            return self.outlet_static_pressure  # Default fallback
    
    # -------------------------------------------------------------------------
    # Solver Settings
    # -------------------------------------------------------------------------
    
    @property
    def solver_type(self) -> str:
        return self.get('solver.type', 'EULER')
    
    @property
    def is_transient(self) -> bool:
        return self.get('solver.is_transient', False)
    
    @property
    def mach_number(self) -> float:
        return self.get('solver.mach_number', 0.3)
    
    @property
    def reynolds_number(self) -> float:
        return self.get('solver.reynolds_number', 1e6)
    
    @property
    def max_iterations(self) -> int:
        return self.get('solver.max_iterations', 2000)
    
    @property
    def convergence_residual(self) -> float:
        return self.get('solver.convergence_residual', -6)
    
    @property
    def cfl_number(self) -> float:
        return self.get('solver.cfl_number', 5.0)
    
    @property
    def cfl_adapt(self) -> bool:
        return self.get('solver.cfl_adapt', False)
    
    @property
    def cfl_min(self) -> float:
        return self.get('solver.cfl_min', 0.1)
    
    @property
    def cfl_max(self) -> float:
        return self.get('solver.cfl_max', 100.0)
    
    @property
    def convective_scheme(self) -> str:
        return self.get('solver.convective_scheme', 'ROE')
    
    @property
    def muscl_reconstruction(self) -> bool:
        return self.get('solver.muscl_reconstruction', True)
    
    @property
    def slope_limiter(self) -> str:
        return self.get('solver.slope_limiter', 'VENKATAKRISHNAN')
    
    @property
    def time_discretization(self) -> str:
        return self.get('solver.time_discretization', 'EULER_IMPLICIT')
    
    @property
    def gradient_method(self) -> str:
        return self.get('solver.gradient_method', 'WEIGHTED_LEAST_SQUARES')
    
    # -------------------------------------------------------------------------
    # Linear Solver Settings
    # -------------------------------------------------------------------------
    
    @property
    def linear_solver(self) -> str:
        return self.get('solver.linear_solver', 'FGMRES')
    
    @property
    def linear_solver_preconditioner(self) -> str:
        return self.get('solver.linear_solver_preconditioner', 'ILU')
    
    @property
    def linear_solver_iterations(self) -> int:
        return self.get('solver.linear_solver_iterations', 10)
    
    @property
    def linear_solver_error(self) -> float:
        return self.get('solver.linear_solver_error', 1e-6)
    
    # -------------------------------------------------------------------------
    # Multigrid Settings
    # -------------------------------------------------------------------------
    
    @property
    def multigrid_levels(self) -> int:
        return self.get('solver.multigrid_levels', 3)
    
    @property
    def multigrid_cycle(self) -> str:
        return self.get('solver.multigrid_cycle', 'W_CYCLE')
    
    # -------------------------------------------------------------------------
    # Transient Settings
    # -------------------------------------------------------------------------
    
    @property
    def time_step(self) -> float:
        return self.get('transient.time_step', 1e-5)
    
    @property
    def end_time(self) -> float:
        return self.get('transient.end_time', 0.01)
    
    @property
    def inner_iterations(self) -> int:
        return self.get('transient.inner_iterations', 20)
    
    # Variable time step (multi-phase simulation)
    @property
    def variable_dt_enabled(self) -> bool:
        return self.get('transient.variable_dt.enabled', False)
    
    @property
    def phase1_dt(self) -> float:
        return self.get('transient.variable_dt.phase1_dt', 1e-7)
    
    @property
    def phase1_duration(self) -> float:
        return self.get('transient.variable_dt.phase1_duration', 0.001)
    
    @property
    def phase1_inner_iter(self) -> int:
        return self.get('transient.variable_dt.phase1_inner_iter', 50)
    
    @property
    def phase2_dt(self) -> float:
        return self.get('transient.variable_dt.phase2_dt', 1e-6)
    
    @property
    def phase2_duration(self) -> float:
        return self.get('transient.variable_dt.phase2_duration', 0.01)
    
    @property
    def phase2_inner_iter(self) -> int:
        return self.get('transient.variable_dt.phase2_inner_iter', 30)
    
    @property
    def phase3_dt(self) -> float:
        return self.get('transient.variable_dt.phase3_dt', 5e-6)
    
    @property
    def phase3_inner_iter(self) -> int:
        return self.get('transient.variable_dt.phase3_inner_iter', 20)
    
    # -------------------------------------------------------------------------
    # Turbulence Settings
    # -------------------------------------------------------------------------
    
    @property
    def turbulence_enabled(self) -> bool:
        return self.get('turbulence.enabled', False)
    
    @property
    def turbulence_model(self) -> str:
        return self.get('turbulence.model', 'SST')
    
    @property
    def turbulence_intensity(self) -> float:
        return self.get('turbulence.intensity', 0.05)
    
    # -------------------------------------------------------------------------
    # Parallel Settings
    # -------------------------------------------------------------------------
    
    @property
    def num_processors(self) -> int:
        return self.get('parallel.num_processors', 6)
    
    # -------------------------------------------------------------------------
    # Output Settings
    # -------------------------------------------------------------------------
    
    @property
    def output_frequency(self) -> int:
        return self.get('output.frequency', 100)
    
    # -------------------------------------------------------------------------
    # Postprocessing / Results Viewer Settings
    # -------------------------------------------------------------------------
    
    @property
    def postproc_default_field(self) -> str:
        return self.get('postprocessing.default_field', 'Pressure')
    
    @property
    def postproc_colormap(self) -> str:
        return self.get('postprocessing.colormap', 'viridis')
    
    @property
    def postproc_contour_levels(self) -> int:
        return self.get('postprocessing.contour_levels', 20)
    
    @property
    def postproc_show_mesh_edges(self) -> bool:
        return self.get('postprocessing.show_mesh_edges', False)
    
    @property
    def postproc_case_directories(self) -> list:
        return self.get('postprocessing.case_directories', ['./case', './case2'])
    
    @property
    def postproc_available_fields(self) -> list:
        return self.get('postprocessing.available_fields', 
                       ['Pressure', 'Velocity', 'Temperature', 'Mach', 'Density'])


# Create a singleton instance for easy access
DEFAULTS = StandardValues()
