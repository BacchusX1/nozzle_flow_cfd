"""Tests for SU2 simulation setup generation."""
from pathlib import Path
import pytest

from backend.simulation.simulation_setup import (
    SimulationSetup, SolverType, BoundaryType, TurbulenceModelType,
    BoundaryCondition, FluidProperties, SolverSettings, TurbulenceModel
)


def _make_min_case(sim: SimulationSetup, case_dir: Path) -> None:
    """Create minimal SU2 case for testing."""
    sim.add_boundary_condition(BoundaryCondition(
        name="inlet",
        boundary_type=BoundaryType.INLET,
        total_pressure=101325.0,
        total_temperature=300.0
    ))
    sim.add_boundary_condition(BoundaryCondition(
        name="outlet",
        boundary_type=BoundaryType.OUTLET,
        static_pressure=80000.0
    ))
    sim.add_boundary_condition(BoundaryCondition(
        name="wall",
        boundary_type=BoundaryType.WALL
    ))
    sim.generate_su2_case(str(case_dir), mesh_data=None)


def test_config_uses_selected_solver(tmp_path: Path):
    """Test that config file uses selected solver type."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.RANS
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    assert "SOLVER=" in content
    assert "RANS" in content


@pytest.mark.parametrize(
    "solver_type,expected",
    [
        (SolverType.EULER, "EULER"),
        (SolverType.NAVIER_STOKES, "NAVIER_STOKES"),
        (SolverType.RANS, "RANS"),
        (SolverType.INC_EULER, "INC_EULER"),
        (SolverType.INC_NAVIER_STOKES, "INC_NAVIER_STOKES"),
        (SolverType.INC_RANS, "INC_RANS"),
    ],
)
def test_solver_type_in_config(tmp_path: Path, solver_type: SolverType, expected: str):
    """Test each solver type is correctly written to config."""
    case_dir = tmp_path / f"case_{solver_type.value}"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = solver_type
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    assert f"SOLVER= {expected}" in content


def test_sst_turbulence_model(tmp_path: Path):
    """Test SST turbulence model configuration."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.RANS
    sim.turbulence_model.enabled = True
    sim.turbulence_model.model_type = TurbulenceModelType.SST
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    assert "KIND_TURB_MODEL= SST" in content


def test_sa_turbulence_model(tmp_path: Path):
    """Test Spalart-Allmaras turbulence model configuration."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.RANS
    sim.turbulence_model.enabled = True
    sim.turbulence_model.model_type = TurbulenceModelType.SA
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    assert "KIND_TURB_MODEL= SA" in content


def test_laminar_no_turbulence(tmp_path: Path):
    """Test laminar flow has no turbulence model."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.NAVIER_STOKES
    sim.turbulence_model.enabled = False
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    # Should not have turbulence model or be NONE
    assert "KIND_TURB_MODEL= NONE" in content or "KIND_TURB_MODEL" not in content


def test_fluid_properties_in_config(tmp_path: Path):
    """Test that fluid properties are written to config."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.fluid_properties.pressure = 101325.0
    sim.fluid_properties.temperature = 288.15
    sim.fluid_properties.density = 1.225
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    assert "FREESTREAM_PRESSURE" in content
    assert "FREESTREAM_TEMPERATURE" in content


def test_mach_number_in_config(tmp_path: Path):
    """Test Mach number setting."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.mach_number = 0.8
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    assert "MACH_NUMBER= 0.8" in content


def test_boundary_markers_in_config(tmp_path: Path):
    """Test that boundary markers are written correctly."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    # Check boundary markers are present
    assert "MARKER" in content
    assert "inlet" in content.lower() or "MARKER_INLET" in content
    assert "outlet" in content.lower() or "MARKER_OUTLET" in content
    assert "wall" in content.lower() or "MARKER_HEATFLUX" in content


def test_convergence_criteria(tmp_path: Path):
    """Test convergence criteria settings."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.residual_reduction = 6
    sim.solver_settings.max_iterations = 5000
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    assert "CONV_RESIDUAL_MINVAL" in content or "CONV_CAUCHY_EPS" in content
    assert "ITER=" in content or "EXT_ITER" in content


def test_numerical_scheme_settings(tmp_path: Path):
    """Test numerical scheme configurations."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.cfl_number = 10.0
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    assert "CFL_NUMBER= 10.0" in content


def test_output_settings(tmp_path: Path):
    """Test output file configuration."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    # Should have output file settings
    assert "OUTPUT" in content or "RESTART" in content or "CONV_FILENAME" in content


def test_mesh_file_created_with_data(tmp_path: Path):
    """Test mesh file is created when mesh data provided."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    
    # Simple mesh data
    mesh_data = {
        "nodes": [(0, 0), (1, 0), (1, 1), (0, 1)],
        "elements": [(0, 1, 2), (0, 2, 3)],
        "boundaries": {
            "inlet": [(0, 1)],
            "outlet": [(2, 3)],
            "wall": [(1, 2), (3, 0)]
        }
    }
    
    sim.add_boundary_condition(BoundaryCondition(
        name="inlet",
        boundary_type=BoundaryType.INLET,
        total_pressure=101325.0
    ))
    sim.add_boundary_condition(BoundaryCondition(
        name="outlet",
        boundary_type=BoundaryType.OUTLET,
        static_pressure=80000.0
    ))
    
    sim.generate_su2_case(str(case_dir), mesh_data=mesh_data)
    
    mesh_file = case_dir / "mesh.su2"
    assert mesh_file.exists()
    
    content = mesh_file.read_text(encoding="utf-8", errors="ignore")
    assert "NDIME=" in content
    assert "NPOIN=" in content
    assert "NELEM=" in content


def test_incompressible_settings(tmp_path: Path):
    """Test incompressible solver settings."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.INC_NAVIER_STOKES
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    assert "INC_NAVIER_STOKES" in content
    # Should have incompressible-specific settings
    assert "INC_" in content


def test_config_file_is_valid_format(tmp_path: Path):
    """Test that config file has valid SU2 format."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    _make_min_case(sim, case_dir)
    
    content = (case_dir / "config.cfg").read_text(encoding="utf-8", errors="ignore")
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        # Each non-comment, non-empty line should have = 
        assert '=' in line, f"Invalid config line: {line}"
