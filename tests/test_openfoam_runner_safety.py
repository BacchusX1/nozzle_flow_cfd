from pathlib import Path

from core.openfoam_runner import OpenFOAMRunner


def _write_min_case(case_dir: Path) -> None:
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant").mkdir(parents=True)

    (case_dir / "system" / "controlDict").write_text(
        "application     simpleFoam;\n",
        encoding="utf-8",
    )
    (case_dir / "system" / "fvSchemes").write_text("ddtSchemes{}\n", encoding="utf-8")
    (case_dir / "system" / "fvSolution").write_text("solvers{}\n", encoding="utf-8")
    (case_dir / "constant" / "transportProperties").write_text("transportModel Newtonian;\n", encoding="utf-8")


def test_get_application_from_control_dict(tmp_path: Path):
    case_dir = tmp_path / "case"
    _write_min_case(case_dir)

    runner = OpenFOAMRunner(str(case_dir))
    assert runner.get_application_from_control_dict() == "simpleFoam"


def test_validate_case_missing_files(tmp_path: Path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    runner = OpenFOAMRunner(str(case_dir))
    ok, msg = runner.validate_case()
    assert not ok
    assert "Missing required case files" in msg


def test_run_solver_does_not_execute_when_case_invalid(tmp_path: Path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    runner = OpenFOAMRunner(str(case_dir))
    assert runner.run_solver("simpleFoam", n_processors=1) is False


def test_run_command_does_not_use_missing_executable(tmp_path: Path):
    case_dir = tmp_path / "case"
    _write_min_case(case_dir)

    runner = OpenFOAMRunner(str(case_dir))
    ok = runner.run_command("definitelyNotARealExecutable123", "log.test")
    assert ok is False
    log = (case_dir / "log.test").read_text(encoding="utf-8", errors="ignore")
    assert "Executable not found" in log
