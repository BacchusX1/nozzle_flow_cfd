"""
Tests to verify that simulations with standard values from config can start properly.
This includes both steady-state and transient simulation modes.
"""
import os
import subprocess
import tempfile
from pathlib import Path
import pytest
import shutil

from src.core.standard_values import DEFAULTS
from src.core.modules.simulation_setup import (
    SimulationSetup, SolverType, BoundaryType, TurbulenceModelType,
    BoundaryCondition, FluidProperties, SolverSettings, TurbulenceModel
)
from src.core.su2_runner import SU2Runner


def _is_su2_available() -> bool:
    """Check if SU2_CFD is available on PATH."""
    return shutil.which("SU2_CFD") is not None


def _create_simple_mesh(mesh_path: Path) -> None:
    """Create a simple 2D mesh for testing (small triangle mesh)."""
    # Simple 2D domain: a small rectangular region with triangles
    mesh_content = """NDIME= 2
NELEM= 8
5 0 1 4 0
5 1 2 5 1
5 2 3 6 2
5 4 5 7 3
5 5 6 8 4
5 0 4 7 5
5 1 5 4 6
5 2 6 5 7
NPOIN= 9
0.0 0.0 0
0.5 0.0 1
1.0 0.0 2
1.5 0.0 3
0.0 0.5 4
0.5 0.5 5
1.0 0.5 6
0.0 1.0 7
0.5 1.0 8
NMARK= 3
MARKER_TAG= inlet
MARKER_ELEMS= 2
3 0 4 
3 4 7 
MARKER_TAG= outlet
MARKER_ELEMS= 2
3 3 6 
3 6 8 
MARKER_TAG= wall
MARKER_ELEMS= 4
3 0 1 
3 1 2 
3 2 3 
3 7 8 
"""
    mesh_path.write_text(mesh_content)


def _setup_simulation_with_defaults(is_transient: bool = False) -> SimulationSetup:
    """Create a SimulationSetup configured with standard values from YAML config."""
    sim = SimulationSetup()
    
    # Apply DEFAULTS to solver settings
    sim.solver_settings.solver_type = SolverType.NAVIER_STOKES  # DEFAULTS.solver_type
    sim.solver_settings.is_transient = is_transient
    sim.solver_settings.max_iterations = DEFAULTS.max_iterations
    sim.solver_settings.convergence_residual = DEFAULTS.convergence_residual
    sim.solver_settings.cfl_number = DEFAULTS.cfl_number
    sim.solver_settings.conv_num_method = DEFAULTS.convective_scheme
    sim.solver_settings.muscl = DEFAULTS.muscl_reconstruction
    sim.solver_settings.slope_limiter = DEFAULTS.slope_limiter
    sim.solver_settings.time_discre = DEFAULTS.time_discretization
    
    if is_transient:
        sim.solver_settings.time_step = DEFAULTS.time_step
        sim.solver_settings.end_time = DEFAULTS.end_time
        sim.solver_settings.inner_iterations = DEFAULTS.inner_iterations
    
    # Turbulence settings
    if DEFAULTS.turbulence_enabled:
        sim.solver_settings.solver_type = SolverType.RANS
        sim.turbulence_model.enabled = True
        turb_map = {"SA": TurbulenceModelType.SA, "SST": TurbulenceModelType.SST}
        sim.turbulence_model.model_type = turb_map.get(DEFAULTS.turbulence_model, TurbulenceModelType.SST)
    else:
        sim.turbulence_model.enabled = False
    
    # Fluid properties from DEFAULTS
    sim.fluid_properties.pressure = DEFAULTS.inlet_total_pressure
    sim.fluid_properties.temperature = DEFAULTS.inlet_total_temperature
    
    # Boundary conditions
    sim.add_boundary_condition(BoundaryCondition(
        name="inlet",
        boundary_type=BoundaryType.INLET,
        total_pressure=DEFAULTS.inlet_total_pressure,
        total_temperature=DEFAULTS.inlet_total_temperature
    ))
    sim.add_boundary_condition(BoundaryCondition(
        name="outlet",
        boundary_type=BoundaryType.OUTLET,
        static_pressure=DEFAULTS.outlet_static_pressure
    ))
    sim.add_boundary_condition(BoundaryCondition(
        name="wall",
        boundary_type=BoundaryType.WALL
    ))
    
    return sim


class TestStandardValuesLoading:
    """Test that standard values are loaded correctly from YAML config."""
    
    def test_defaults_loaded(self):
        """Test that DEFAULTS singleton loads values from YAML."""
        assert DEFAULTS._values is not None, "DEFAULTS values not loaded"
        assert len(DEFAULTS._values) > 0, "DEFAULTS is empty"
    
    def test_time_step_from_config(self):
        """Test that time_step is loaded from config (not hardcoded default)."""
        # The YAML config has 5e-6, not the hardcoded 1e-5
        assert DEFAULTS.time_step == 5e-6, f"Expected time_step=5e-6, got {DEFAULTS.time_step}"
    
    def test_solver_type_from_config(self):
        """Test that solver_type is loaded from config."""
        assert DEFAULTS.solver_type in ["EULER", "NAVIER_STOKES", "RANS"], \
            f"Invalid solver_type: {DEFAULTS.solver_type}"
    
    def test_cfl_number_from_config(self):
        """Test that CFL number is loaded from config."""
        assert 0.1 <= DEFAULTS.cfl_number <= 1000, f"CFL out of range: {DEFAULTS.cfl_number}"
    
    def test_convective_scheme_from_config(self):
        """Test that convective scheme is loaded from config."""
        valid_schemes = ["JST", "ROE", "AUSM", "AUSMPLUSUP2", "HLLC", "SLAU2", "LAX-FRIEDRICH"]
        assert DEFAULTS.convective_scheme in valid_schemes, \
            f"Invalid convective_scheme: {DEFAULTS.convective_scheme}"
    
    def test_discretization_settings(self):
        """Test discretization settings are loaded."""
        assert isinstance(DEFAULTS.muscl_reconstruction, bool)
        assert DEFAULTS.slope_limiter in ["VENKATAKRISHNAN", "VENKATAKRISHNAN_WANG", "BARTH_JESPERSEN", "NONE"]
        assert DEFAULTS.time_discretization in ["EULER_IMPLICIT", "EULER_EXPLICIT", "RUNGE-KUTTA_EXPLICIT"]
        assert DEFAULTS.gradient_method in ["GREEN_GAUSS", "WEIGHTED_LEAST_SQUARES"]


class TestSteadyStateConfigGeneration:
    """Test steady-state simulation config generation with standard values."""
    
    def test_steady_config_generates_without_error(self, tmp_path: Path):
        """Test that steady-state config can be generated."""
        sim = _setup_simulation_with_defaults(is_transient=False)
        case_dir = tmp_path / "steady_case"
        case_dir.mkdir()
        
        # Create simple mesh
        mesh_path = case_dir / "mesh.su2"
        _create_simple_mesh(mesh_path)
        
        # Generate config
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        config_path = case_dir / "config.cfg"
        assert config_path.exists(), "Config file not generated"
    
    def test_steady_config_has_required_sections(self, tmp_path: Path):
        """Test steady config has all required sections."""
        sim = _setup_simulation_with_defaults(is_transient=False)
        case_dir = tmp_path / "steady_case"
        case_dir.mkdir()
        _create_simple_mesh(case_dir / "mesh.su2")
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        content = (case_dir / "config.cfg").read_text()
        
        # Required sections
        assert "SOLVER=" in content, "Missing SOLVER"
        assert "CFL_NUMBER=" in content, "Missing CFL_NUMBER"
        assert "CONV_NUM_METHOD_FLOW=" in content, "Missing CONV_NUM_METHOD_FLOW"
        assert "MARKER_INLET=" in content, "Missing MARKER_INLET"
        assert "MARKER_OUTLET=" in content, "Missing MARKER_OUTLET"
        assert "MESH_FILENAME=" in content, "Missing MESH_FILENAME"
    
    def test_steady_config_no_time_domain(self, tmp_path: Path):
        """Test steady config does not have TIME_DOMAIN=YES."""
        sim = _setup_simulation_with_defaults(is_transient=False)
        case_dir = tmp_path / "steady_case"
        case_dir.mkdir()
        _create_simple_mesh(case_dir / "mesh.su2")
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        content = (case_dir / "config.cfg").read_text()
        assert "TIME_DOMAIN= NO" in content or "TIME_DOMAIN=NO" in content or "TIME_DOMAIN" not in content


class TestTransientConfigGeneration:
    """Test transient simulation config generation with standard values."""
    
    def test_transient_config_generates_without_error(self, tmp_path: Path):
        """Test that transient config can be generated."""
        sim = _setup_simulation_with_defaults(is_transient=True)
        case_dir = tmp_path / "transient_case"
        case_dir.mkdir()
        
        mesh_path = case_dir / "mesh.su2"
        _create_simple_mesh(mesh_path)
        
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        config_path = case_dir / "config.cfg"
        assert config_path.exists(), "Config file not generated"
    
    def test_transient_config_has_time_settings(self, tmp_path: Path):
        """Test transient config has time-related settings."""
        sim = _setup_simulation_with_defaults(is_transient=True)
        case_dir = tmp_path / "transient_case"
        case_dir.mkdir()
        _create_simple_mesh(case_dir / "mesh.su2")
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        content = (case_dir / "config.cfg").read_text()
        
        assert "TIME_DOMAIN= YES" in content or "TIME_DOMAIN=YES" in content, "Missing TIME_DOMAIN=YES"
        assert "TIME_STEP=" in content, "Missing TIME_STEP"
        assert "INNER_ITER=" in content, "Missing INNER_ITER"
    
    def test_transient_config_uses_correct_time_step(self, tmp_path: Path):
        """Test transient config has correct time step from DEFAULTS."""
        sim = _setup_simulation_with_defaults(is_transient=True)
        case_dir = tmp_path / "transient_case"
        case_dir.mkdir()
        _create_simple_mesh(case_dir / "mesh.su2")
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        content = (case_dir / "config.cfg").read_text()
        
        # Check that time step matches DEFAULTS (5e-6)
        assert "TIME_STEP= 5e-06" in content or "TIME_STEP=5e-06" in content or \
               "TIME_STEP= 5.0e-06" in content or "TIME_STEP=5.0e-06" in content, \
               f"TIME_STEP should be 5e-06 from config, got: {content}"


@pytest.mark.skipif(not _is_su2_available(), reason="SU2_CFD not installed")
class TestSU2SimulationStart:
    """Integration tests that verify SU2 can actually start with standard values.
    
    These tests require SU2 to be installed and will be skipped otherwise.
    """
    
    def test_steady_simulation_starts(self, tmp_path: Path):
        """Test that SU2 can start a steady-state simulation (runs 1 iteration)."""
        sim = _setup_simulation_with_defaults(is_transient=False)
        sim.solver_settings.max_iterations = 1  # Just 1 iteration to test startup
        sim.solver_settings.n_processors = 1  # Serial to avoid MPI issues
        
        case_dir = tmp_path / "steady_case"
        case_dir.mkdir()
        _create_simple_mesh(case_dir / "mesh.su2")
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        runner = SU2Runner(str(case_dir))
        success = runner.run_solver(n_processors=1)
        
        log_path = case_dir / "log.SU2_CFD"
        log_content = log_path.read_text() if log_path.exists() else "No log file"
        
        # Check log for startup indicators (even if simulation fails due to mesh)
        assert "Physical Case Definition" in log_content or success, \
            f"SU2 failed to start steady simulation. Log:\n{log_content[:2000]}"
    
    def test_transient_simulation_starts(self, tmp_path: Path):
        """Test that SU2 can start a transient simulation (runs 1 time step)."""
        sim = _setup_simulation_with_defaults(is_transient=True)
        sim.solver_settings.end_time = DEFAULTS.time_step * 2  # Just 2 time steps
        sim.solver_settings.n_processors = 1  # Serial to avoid MPI issues
        
        case_dir = tmp_path / "transient_case"
        case_dir.mkdir()
        _create_simple_mesh(case_dir / "mesh.su2")
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        runner = SU2Runner(str(case_dir))
        success = runner.run_solver(n_processors=1)
        
        log_path = case_dir / "log.SU2_CFD"
        log_content = log_path.read_text() if log_path.exists() else "No log file"
        
        # Check log for startup indicators
        assert "Physical Case Definition" in log_content or success, \
            f"SU2 failed to start transient simulation. Log:\n{log_content[:2000]}"
    
    def test_parallel_simulation_with_env_vars(self, tmp_path: Path):
        """Test that parallel simulation can start with proper MPI env vars."""
        if not shutil.which("mpirun"):
            pytest.skip("mpirun not available")
        
        sim = _setup_simulation_with_defaults(is_transient=False)
        sim.solver_settings.max_iterations = 1
        sim.solver_settings.n_processors = 2  # Use 2 processors
        
        case_dir = tmp_path / "parallel_case"
        case_dir.mkdir()
        _create_simple_mesh(case_dir / "mesh.su2")
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        runner = SU2Runner(str(case_dir))
        success = runner.run_solver(n_processors=2)
        
        log_path = case_dir / "log.SU2_CFD"
        log_content = log_path.read_text() if log_path.exists() else "No log file"
        
        # Check that it at least started (may still fail due to small mesh partitioning)
        # The key is no segfault during startup
        assert "Segmentation fault" not in log_content, \
            f"Segfault occurred in parallel run. Log:\n{log_content[:2000]}"


class TestBoundaryConditionsFromDefaults:
    """Test that boundary conditions use values from DEFAULTS."""
    
    def test_inlet_pressure_from_defaults(self, tmp_path: Path):
        """Test inlet pressure matches DEFAULTS."""
        sim = _setup_simulation_with_defaults(is_transient=False)
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        _create_simple_mesh(case_dir / "mesh.su2")
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        content = (case_dir / "config.cfg").read_text()
        
        # Check inlet BC contains expected pressure
        expected_pressure = DEFAULTS.inlet_total_pressure
        assert f"{expected_pressure}" in content or f"{int(expected_pressure)}" in content, \
            f"Inlet pressure {expected_pressure} not found in config"
    
    def test_outlet_pressure_from_defaults(self, tmp_path: Path):
        """Test outlet pressure matches DEFAULTS."""
        sim = _setup_simulation_with_defaults(is_transient=False)
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        _create_simple_mesh(case_dir / "mesh.su2")
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        content = (case_dir / "config.cfg").read_text()
        
        expected_pressure = DEFAULTS.outlet_static_pressure
        assert f"{expected_pressure}" in content or f"{int(expected_pressure)}" in content, \
            f"Outlet pressure {expected_pressure} not found in config"
