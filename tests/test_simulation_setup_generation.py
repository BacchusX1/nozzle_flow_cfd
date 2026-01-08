from pathlib import Path

import pytest

from core.modules.simulation_setup import SimulationSetup, SolverType


def _make_min_case(sim: SimulationSetup, case_dir: Path) -> None:
    sim.set_inlet_conditions("inlet", velocity=10.0)
    sim.set_outlet_conditions("outlet", pressure=80000.0)
    sim.set_wall_conditions("wall")
    sim.generate_openfoam_case(str(case_dir), mesh_data=None)


def test_control_dict_uses_selected_solver(tmp_path: Path):
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.RHOSIMPLE_FOAM
    _make_min_case(sim, case_dir)

    content = (case_dir / "system" / "controlDict").read_text(encoding="utf-8", errors="ignore")
    assert "application" in content
    assert "rhoSimpleFoam" in content


@pytest.mark.parametrize(
    "solver_type,expected_ddt",
    [
        (SolverType.SIMPLE_FOAM, "steadyState"),
        (SolverType.RHOSIMPLE_FOAM, "steadyState"),
        (SolverType.SONIC_FOAM, "Euler"),
    ],
)
def test_fv_schemes_ddt_matches_solver(tmp_path: Path, solver_type: SolverType, expected_ddt: str):
    case_dir = tmp_path / f"case_{solver_type.value}"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = solver_type
    _make_min_case(sim, case_dir)

    content = (case_dir / "system" / "fvSchemes").read_text(encoding="utf-8", errors="ignore")
    assert "ddtSchemes" in content
    assert f"default         {expected_ddt};" in content


def test_komega_sst_writes_k_and_omega_fields(tmp_path: Path):
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.turbulence_model.enabled = True
    sim.turbulence_model.model_type = "kOmegaSST"
    _make_min_case(sim, case_dir)

    fields = {p.name for p in (case_dir / "0").iterdir()}
    assert "k" in fields
    assert "omega" in fields
    assert "nut" in fields
    assert "epsilon" not in fields

    fv_solution = (case_dir / "system" / "fvSolution").read_text(encoding="utf-8", errors="ignore")
    assert "omega" in fv_solution
    assert "epsilon" not in fv_solution


def test_laminar_does_not_write_turbulence_fields(tmp_path: Path):
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.turbulence_model.enabled = False
    _make_min_case(sim, case_dir)

    fields = {p.name for p in (case_dir / "0").iterdir()}
    assert "U" in fields
    assert "p" in fields
    assert "k" not in fields
    assert "epsilon" not in fields
    assert "omega" not in fields
    assert "nut" not in fields


def test_pressure_internal_field_uses_fluid_pressure(tmp_path: Path):
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.fluid_properties.pressure = 123456.0
    _make_min_case(sim, case_dir)

    p_file = (case_dir / "0" / "p").read_text(encoding="utf-8", errors="ignore")
    assert "internalField" in p_file
    assert "uniform 123456.0" in p_file


def test_sonic_foam_time_step_settings(tmp_path: Path):
    """Test that sonicFoam uses correct time stepping from solver settings."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.SONIC_FOAM
    sim.solver_settings.time_step = 1e-7
    sim.solver_settings.end_time = 0.005
    sim.solver_settings.max_courant = 0.8
    sim.solver_settings.adjust_time_step = True
    _make_min_case(sim, case_dir)

    content = (case_dir / "system" / "controlDict").read_text(encoding="utf-8", errors="ignore")
    assert "sonicFoam" in content
    assert "deltaT" in content
    assert "1e-07" in content
    assert "endTime" in content
    assert "0.005" in content
    assert "adjustTimeStep  yes" in content
    assert "maxCo           0.8" in content


def test_sonic_foam_pimple_iterations(tmp_path: Path):
    """Test that sonicFoam uses PIMPLE iteration settings from solver settings."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.SONIC_FOAM
    sim.solver_settings.n_outer_correctors = 3
    sim.solver_settings.n_correctors = 4
    sim.solver_settings.n_non_orthogonal_correctors = 1
    _make_min_case(sim, case_dir)

    content = (case_dir / "system" / "fvSolution").read_text(encoding="utf-8", errors="ignore")
    assert "PIMPLE" in content
    assert "nOuterCorrectors    3" in content
    assert "nCorrectors         4" in content
    assert "nNonOrthogonalCorrectors 1" in content


def test_sonic_foam_pressure_dimensions(tmp_path: Path):
    """Test that sonicFoam uses compressible pressure dimensions."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.SONIC_FOAM
    _make_min_case(sim, case_dir)

    p_file = (case_dir / "0" / "p").read_text(encoding="utf-8", errors="ignore")
    # Compressible pressure dimensions: [1 -1 -2 0 0 0 0] (Pa)
    assert "[1 -1 -2 0 0 0 0]" in p_file


def test_sonic_foam_generates_alphat(tmp_path: Path):
    """Test that sonicFoam generates alphat field for turbulent compressible flow."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.SONIC_FOAM
    sim.turbulence_model.enabled = True
    sim.turbulence_model.model_type = "kOmegaSST"
    _make_min_case(sim, case_dir)

    fields = {p.name for p in (case_dir / "0").iterdir()}
    assert "alphat" in fields
    assert "T" in fields

    alphat_content = (case_dir / "0" / "alphat").read_text(encoding="utf-8", errors="ignore")
    assert "alphat" in alphat_content
    assert "compressible::alphatWallFunction" in alphat_content


def test_sonic_foam_fvschemes_compressible_divergence(tmp_path: Path):
    """Test that sonicFoam generates correct compressible divergence schemes."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.SONIC_FOAM
    _make_min_case(sim, case_dir)

    content = (case_dir / "system" / "fvSchemes").read_text(encoding="utf-8", errors="ignore")
    # Compressible-specific schemes
    assert "div(phiv,p)" in content
    assert "div(((rho*nuEff)*dev2(T(grad(U)))))" in content
    assert "div(phi,e)" in content


def test_sonic_foam_fvsolution_rho_solver(tmp_path: Path):
    """Test that sonicFoam generates rho solver entries."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.SONIC_FOAM
    _make_min_case(sim, case_dir)

    content = (case_dir / "system" / "fvSolution").read_text(encoding="utf-8", errors="ignore")
    assert "rho" in content
    assert "rhoFinal" in content
    assert "diagonal" in content
    # Final solvers for PIMPLE
    assert "pFinal" in content


def test_steady_solver_uses_iteration_as_time(tmp_path: Path):
    """Test that steady-state solvers use iteration count as pseudo-time."""
    case_dir = tmp_path / "case"
    sim = SimulationSetup()
    sim.solver_settings.solver_type = SolverType.SIMPLE_FOAM
    sim.solver_settings.max_iterations = 500
    sim.solver_settings.time_step = 1.0
    sim.solver_settings.end_time = 500.0
    _make_min_case(sim, case_dir)

    content = (case_dir / "system" / "controlDict").read_text(encoding="utf-8", errors="ignore")
    assert "simpleFoam" in content
    assert "SIMPLE" in (case_dir / "system" / "fvSolution").read_text(encoding="utf-8", errors="ignore")
