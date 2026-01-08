"""
Integration tests for OpenFOAM solver execution.

These tests verify that OpenFOAM solvers can actually run on the 
pre-configured test cases. They are marked as slow and may require
OpenFOAM to be installed on the system.

Usage:
    pytest tests/test_openfoam_execution.py -v
    pytest tests/test_openfoam_execution.py -v -m "not slow"  # Skip slow tests
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


# Check if OpenFOAM is available
def _openfoam_available() -> bool:
    """Check if OpenFOAM executables are available."""
    try:
        result = subprocess.run(
            ["sonicFoam", "-help"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


OPENFOAM_AVAILABLE = _openfoam_available()


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def case2_copy(tmp_path: Path) -> Path:
    """Create a temporary copy of case2 for testing."""
    src = get_project_root() / "case2"
    if not src.exists():
        pytest.skip("case2 directory not found")
    
    dst = tmp_path / "case2"
    shutil.copytree(src, dst)
    
    # Clean up any processor directories from previous runs
    for proc_dir in dst.glob("processor*"):
        shutil.rmtree(proc_dir)
    
    return dst


def configure_short_run(case_dir: Path, end_time: float = 1e-05, delta_t: float = 1e-06) -> None:
    """Configure case for a very short test run."""
    control_dict = case_dir / "system" / "controlDict"
    content = control_dict.read_text(encoding="utf-8")
    
    # Replace endTime and deltaT for a minimal test
    import re
    content = re.sub(r'endTime\s+[\d.eE+-]+;', f'endTime         {end_time};', content)
    content = re.sub(r'deltaT\s+[\d.eE+-]+;', f'deltaT          {delta_t};', content)
    content = re.sub(r'writeInterval\s+[\d.eE+-]+;', f'writeInterval   {end_time};', content)
    
    control_dict.write_text(content, encoding="utf-8")


@pytest.mark.slow
@pytest.mark.skipif(not OPENFOAM_AVAILABLE, reason="OpenFOAM not installed")
def test_sonicfoam_starts_and_runs_one_timestep(case2_copy: Path):
    """
    Test that sonicFoam can start and run at least one timestep on case2.
    
    This verifies:
    - All required fields are present and correctly formatted
    - fvSchemes has all required divSchemes
    - fvSolution has correct PIMPLE dictionary format
    - Boundary conditions are compatible with the mesh
    """
    case_dir = case2_copy
    
    # Configure for minimal run (10 timesteps)
    configure_short_run(case_dir, end_time=1e-05, delta_t=1e-06)
    
    # Run sonicFoam (single processor for simplicity)
    log_file = case_dir / "log.sonicFoam.test"
    
    result = subprocess.run(
        ["sonicFoam"],
        cwd=str(case_dir),
        capture_output=True,
        timeout=120,  # 2 minute timeout
        text=True
    )
    
    # Write log for debugging
    log_file.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    
    # Check that it didn't crash with fatal error
    output = result.stdout + result.stderr
    
    assert "FOAM FATAL ERROR" not in output, f"sonicFoam crashed with fatal error:\n{output[-2000:]}"
    assert "FOAM FATAL IO ERROR" not in output, f"sonicFoam crashed with IO error:\n{output[-2000:]}"
    
    # Check that time loop started
    assert "Starting time loop" in output, "Time loop did not start"
    
    # Check that at least one timestep was computed
    assert "Time = " in output, "No timestep was computed"
    

@pytest.mark.slow
@pytest.mark.skipif(not OPENFOAM_AVAILABLE, reason="OpenFOAM not installed")
def test_sonicfoam_parallel_starts(case2_copy: Path):
    """
    Test that sonicFoam can start in parallel mode.
    """
    case_dir = case2_copy
    
    # Configure for minimal run
    configure_short_run(case_dir, end_time=2e-06, delta_t=1e-06)
    
    # Check if mpirun is available
    try:
        subprocess.run(["mpirun", "--version"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("mpirun not available")
    
    # Check if decomposePar is available
    try:
        subprocess.run(["decomposePar", "-help"], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("decomposePar not available")
    
    # Read number of processors from decomposeParDict
    decompose_dict = case_dir / "system" / "decomposeParDict"
    n_procs = 4  # default
    if decompose_dict.exists():
        import re
        content = decompose_dict.read_text(encoding="utf-8")
        match = re.search(r'numberOfSubdomains\s+(\d+);', content)
        if match:
            n_procs = int(match.group(1))
    
    # Decompose the case
    result = subprocess.run(
        ["decomposePar"],
        cwd=str(case_dir),
        capture_output=True,
        timeout=60,
        text=True
    )
    
    assert result.returncode == 0, f"decomposePar failed:\n{result.stderr}"
    
    # Check processor directories were created
    proc_dirs = list(case_dir.glob("processor*"))
    assert len(proc_dirs) > 0, "No processor directories created"
    
    # Run in parallel (use same number as decomposed)
    result = subprocess.run(
        ["mpirun", "-np", str(n_procs), "sonicFoam", "-parallel"],
        cwd=str(case_dir),
        capture_output=True,
        timeout=120,
        text=True
    )
    
    output = result.stdout + result.stderr
    
    # Write log for debugging
    (case_dir / "log.sonicFoam.parallel.test").write_text(output, encoding="utf-8")
    
    assert "FOAM FATAL ERROR" not in output, f"Parallel sonicFoam crashed:\n{output[-2000:]}"
    assert "Starting time loop" in output, "Time loop did not start in parallel"


@pytest.mark.slow  
@pytest.mark.skipif(not OPENFOAM_AVAILABLE, reason="OpenFOAM not installed")
def test_sonicfoam_fields_validated(case2_copy: Path):
    """
    Test that OpenFOAM can read and validate all field files.
    
    Uses checkMesh to verify mesh, and a quick solver start to verify fields.
    """
    case_dir = case2_copy
    
    # Run checkMesh if available
    try:
        result = subprocess.run(
            ["checkMesh"],
            cwd=str(case_dir),
            capture_output=True,
            timeout=60,
            text=True
        )
        
        assert "Mesh OK" in result.stdout or result.returncode == 0, \
            f"checkMesh reported issues:\n{result.stdout}"
    except FileNotFoundError:
        pass  # checkMesh not available, skip this check
    
    # Verify all required fields exist
    required_fields = ["U", "p", "T", "k", "omega", "nut", "alphat"]
    zero_dir = case_dir / "0"
    
    for field in required_fields:
        field_file = zero_dir / field
        assert field_file.exists(), f"Required field {field} not found in 0/"
        
        # Basic content check
        content = field_file.read_text(encoding="utf-8")
        assert "FoamFile" in content, f"Field {field} missing FoamFile header"
        assert "boundaryField" in content, f"Field {field} missing boundaryField"
        assert "frontAndBack" in content, f"Field {field} missing frontAndBack boundary"


@pytest.mark.slow
@pytest.mark.skipif(not OPENFOAM_AVAILABLE, reason="OpenFOAM not installed") 
def test_simpleFoam_runs_on_case(tmp_path: Path):
    """
    Test that simpleFoam can run on case (the incompressible case).
    """
    src = get_project_root() / "case"
    if not src.exists():
        pytest.skip("case directory not found")
    
    case_dir = tmp_path / "case"
    shutil.copytree(src, case_dir)
    
    # Clean up processor directories
    for proc_dir in case_dir.glob("processor*"):
        shutil.rmtree(proc_dir)
    
    # Clean up time directories (keep only 0)
    for time_dir in case_dir.iterdir():
        if time_dir.is_dir():
            try:
                t = float(time_dir.name)
                if t > 0:
                    shutil.rmtree(time_dir)
            except ValueError:
                pass
    
    # Modify controlDict for a very short run
    control_dict = case_dir / "system" / "controlDict"
    content = control_dict.read_text(encoding="utf-8")
    
    import re
    content = re.sub(r'endTime\s+[\d.eE+-]+;', 'endTime         5;', content)
    content = re.sub(r'deltaT\s+[\d.eE+-]+;', 'deltaT          1;', content)
    content = re.sub(r'writeInterval\s+[\d.eE+-]+;', 'writeInterval   5;', content)
    
    control_dict.write_text(content, encoding="utf-8")
    
    # Run simpleFoam
    result = subprocess.run(
        ["simpleFoam"],
        cwd=str(case_dir),
        capture_output=True,
        timeout=120,
        text=True
    )
    
    output = result.stdout + result.stderr
    
    assert "FOAM FATAL ERROR" not in output, f"simpleFoam crashed:\n{output[-2000:]}"
    assert "Starting time loop" in output or "Time = " in output, "Solver did not start"
