"""Tests for SU2 runner module."""
from pathlib import Path
import pytest

from src.core.su2_runner import SU2Runner


def _write_min_case(case_dir: Path) -> None:
    """Write minimal SU2 case files for testing."""
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Write minimal config file
    config_content = """% SU2 Configuration File
SOLVER= EULER
MATH_PROBLEM= DIRECT
MACH_NUMBER= 0.8
AOA= 1.25
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 288.15
MESH_FILENAME= mesh.su2
MARKER_EULER= ( wall )
MARKER_FAR= ( farfield )
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 25.0
ITER= 250
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW)
"""
    (case_dir / "config.cfg").write_text(config_content, encoding="utf-8")
    
    # Write minimal mesh file
    mesh_content = """% SU2 Mesh file
NDIME= 2
NELEM= 2
5 0 1 2 0
5 0 2 3 1
NPOIN= 4
0.0 0.0 0
1.0 0.0 1
1.0 1.0 2
0.0 1.0 3
NMARK= 1
MARKER_TAG= wall
MARKER_ELEMS= 1
3 0 1 0
"""
    (case_dir / "mesh.su2").write_text(mesh_content, encoding="utf-8")


def test_get_solver_from_config(tmp_path: Path):
    """Test reading solver type from config."""
    case_dir = tmp_path / "case"
    _write_min_case(case_dir)
    
    runner = SU2Runner(str(case_dir))
    assert runner.get_solver_from_config() == "EULER"


def test_validate_case_valid(tmp_path: Path):
    """Test validation of a valid case."""
    case_dir = tmp_path / "case"
    _write_min_case(case_dir)
    
    runner = SU2Runner(str(case_dir))
    valid, message = runner.validate_case()
    assert valid
    assert "OK" in message or message == ""


def test_validate_case_missing_config(tmp_path: Path):
    """Test validation fails with missing config."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    
    runner = SU2Runner(str(case_dir))
    valid, message = runner.validate_case()
    assert not valid
    assert "config" in message.lower() or "missing" in message.lower()


def test_validate_case_missing_mesh(tmp_path: Path):
    """Test validation fails with missing mesh."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    
    # Only config, no mesh
    config_content = """SOLVER= EULER
MESH_FILENAME= mesh.su2
"""
    (case_dir / "config.cfg").write_text(config_content, encoding="utf-8")
    
    runner = SU2Runner(str(case_dir))
    valid, message = runner.validate_case()
    assert not valid
    assert "mesh" in message.lower()


def test_run_solver_does_not_execute_when_invalid(tmp_path: Path):
    """Test that run_solver returns False for invalid case."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    
    runner = SU2Runner(str(case_dir))
    result = runner.run_solver(n_processors=1)
    assert result is False


def test_run_command_missing_executable(tmp_path: Path):
    """Test that run_command handles missing executable."""
    case_dir = tmp_path / "case"
    _write_min_case(case_dir)
    
    runner = SU2Runner(str(case_dir))
    ok = runner.run_command("DEFINITELY_NOT_A_REAL_EXECUTABLE_12345", log_file="log.txt")
    assert ok is False
    # Check that the log file contains an error message
    log_path = case_dir / "log.txt"
    if log_path.exists():
        content = log_path.read_text()
        assert "not found" in content.lower() or "error" in content.lower()


def test_mesh_filename_parsing(tmp_path: Path):
    """Test mesh filename parsing from config."""
    case_dir = tmp_path / "case"
    _write_min_case(case_dir)
    
    runner = SU2Runner(str(case_dir))
    # Test the private _get_mesh_filename method
    mesh_filename = runner._get_mesh_filename()
    
    assert mesh_filename == "mesh.su2"
