"""
Integration tests for de Laval nozzle CFD simulation workflow.

These tests run REAL SU2 simulations - they require SU2 to be installed.
Run these tests with the su2_env conda environment activated.

Usage:
    conda activate su2_env
    pytest tests/test_laval_nozzle_integration.py -v

These tests verify:
1. Geometry loading from template
2. Mesh generation with Gmsh
3. SU2 case file generation
4. Actual SU2_CFD execution (limited iterations)
5. Simulation results validation
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# Skip all tests in this module if SU2 is not installed
SU2_AVAILABLE = shutil.which("SU2_CFD") is not None

pytestmark = pytest.mark.skipif(
    not SU2_AVAILABLE,
    reason="SU2_CFD not found in PATH. Install with: conda activate su2_env"
)


@pytest.fixture()
def suppress_message_boxes(monkeypatch):
    """Prevent modal QMessageBox calls from blocking tests."""
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)


@pytest.fixture()
def gui(qapp, suppress_message_boxes):
    """Create GUI instance with stubbed post-processing tab."""
    from src import frontend
    from PySide6.QtWidgets import QWidget

    # Stub the heavy post-processing tab
    def _stub_post_tab(self):
        return QWidget()

    frontend.NozzleDesignGUI.create_postprocessing_tab = _stub_post_tab

    window = frontend.NozzleDesignGUI()
    window.show()
    qapp.processEvents()
    yield window
    window.close()
    qapp.processEvents()


def _find_button(window, text: str):
    """Find a button by its text."""
    from PySide6.QtWidgets import QPushButton

    matches = [b for b in window.findChildren(QPushButton) if b.text() == text]
    assert matches, f"Button not found: {text}"
    assert len(matches) == 1, f"Multiple buttons found for text={text}: {len(matches)}"
    return matches[0]


def _click(window, qapp, button_text: str):
    """Click a button by its text."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    btn = _find_button(window, button_text)
    assert btn.isEnabled(), f"Button disabled: {button_text}"
    QTest.mouseClick(btn, Qt.LeftButton)
    qapp.processEvents()


def _modify_config_for_quick_test(config_path: Path, max_iter: int = 10):
    """Modify SU2 config for quick testing (few iterations)."""
    content = config_path.read_text()
    
    # Replace iteration count for quick testing
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith('ITER=') or line.strip().startswith('ITER '):
            new_lines.append(f'ITER= {max_iter}')
        elif line.strip().startswith('TIME_ITER=') or line.strip().startswith('TIME_ITER '):
            new_lines.append(f'TIME_ITER= {max_iter}')
        else:
            new_lines.append(line)
    
    # Ensure ITER is set if not found
    if not any('ITER=' in l for l in new_lines):
        new_lines.insert(0, f'ITER= {max_iter}')
    
    config_path.write_text('\n'.join(new_lines))


class TestDeLavalSteadyIntegration:
    """Integration tests for steady-state de Laval nozzle simulation."""

    def test_full_workflow_steady_euler(self, gui, qapp, tmp_path: Path):
        """
        Test complete workflow: de Laval geometry -> mesh -> steady EULER simulation.
        
        This test runs a real SU2 simulation with EULER solver (inviscid).
        """
        # Step 1: Load de Laval nozzle geometry template
        gui.load_template("de_laval")
        assert len(gui.geometry.elements) > 0, "De Laval template should have geometry elements"

        # Step 2: Generate mesh (uses real Gmsh)
        _click(gui, qapp, " Generate Mesh")
        assert gui.current_mesh_data is not None, "Mesh data should be generated"
        
        # Verify mesh has reasonable number of elements
        mesh_stats = gui.current_mesh_data.get("stats", {})
        num_elements = mesh_stats.get("num_elements", 0)
        assert num_elements > 100, f"Mesh should have >100 elements, got {num_elements}"

        # Step 3: Switch to Simulation tab and configure steady EULER
        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        case_dir = tmp_path / "steady_euler_case"
        gui.case_directory.setText(str(case_dir))

        gui.simulation_mode.setCurrentText("Steady")
        qapp.processEvents()

        gui.solver_type.setCurrentText("EULER")
        qapp.processEvents()

        # Set reasonable flow conditions for supersonic nozzle
        gui.inlet_pressure.setValue(300000)  # 3 bar
        gui.temperature.setValue(300)        # 300 K
        gui.n_processors.setValue(1)

        # Step 4: Setup case files
        _click(gui, qapp, "⚙️ Generate Case Files")

        # Verify case structure
        assert gui.current_case_directory is not None
        cd = Path(gui.current_case_directory)
        assert (cd / "config.cfg").exists(), "Config file should exist"
        assert (cd / "mesh.su2").exists(), "Mesh file should exist"

        # Verify config content
        config = (cd / "config.cfg").read_text()
        assert "SOLVER= EULER" in config

        # Step 5: Modify config for quick test (only 10 iterations)
        _modify_config_for_quick_test(cd / "config.cfg", max_iter=10)

        # Step 6: Run actual SU2 simulation
        from src.core.su2_runner import SU2Runner
        
        runner = SU2Runner(str(cd))
        valid, msg = runner.validate_case()
        assert valid, f"Case validation failed: {msg}"

        # Run SU2_CFD (this actually runs the solver!)
        success = runner.run_solver(n_processors=1)
        
        # Check log for any errors
        log_path = cd / "log.SU2_CFD"
        if log_path.exists():
            log_content = log_path.read_text()
            # Print last 20 lines for debugging
            print("\n--- SU2 Log (last 20 lines) ---")
            print('\n'.join(log_content.split('\n')[-20:]))
        
        assert success, f"SU2_CFD failed. Check {log_path}"

        # Verify output files were created
        # SU2 creates restart and history files
        output_files = list(cd.glob("*.csv")) + list(cd.glob("*.dat")) + list(cd.glob("restart*"))
        assert len(output_files) > 0, "SU2 should create output files"

    def test_full_workflow_steady_rans(self, gui, qapp, tmp_path: Path):
        """
        Test complete workflow: de Laval geometry -> mesh -> steady RANS simulation.
        
        This test runs a real SU2 simulation with RANS solver (SST turbulence).
        """
        # Step 1: Load de Laval nozzle geometry template
        gui.load_template("de_laval")
        assert len(gui.geometry.elements) > 0

        # Step 2: Generate mesh
        _click(gui, qapp, " Generate Mesh")
        assert gui.current_mesh_data is not None

        # Step 3: Configure steady RANS
        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        case_dir = tmp_path / "steady_rans_case"
        gui.case_directory.setText(str(case_dir))

        gui.simulation_mode.setCurrentText("Steady")
        qapp.processEvents()

        gui.solver_type.setCurrentText("RANS")
        qapp.processEvents()

        # Verify turbulence model is visible and set to SST
        assert gui.turbulence_model.isVisible()
        gui.turbulence_model.setCurrentText("SST")

        gui.inlet_pressure.setValue(300000)
        gui.temperature.setValue(300)
        gui.n_processors.setValue(1)

        # Step 4: Setup case files
        _click(gui, qapp, "⚙️ Generate Case Files")

        cd = Path(gui.current_case_directory)
        assert (cd / "config.cfg").exists()
        assert (cd / "mesh.su2").exists()

        config = (cd / "config.cfg").read_text()
        assert "SOLVER= RANS" in config
        assert "KIND_TURB_MODEL= SST" in config

        # Step 5: Modify for quick test
        _modify_config_for_quick_test(cd / "config.cfg", max_iter=10)

        # Step 6: Run SU2
        from src.core.su2_runner import SU2Runner
        
        runner = SU2Runner(str(cd))
        success = runner.run_solver(n_processors=1)
        
        log_path = cd / "log.SU2_CFD"
        if log_path.exists():
            log_content = log_path.read_text()
            print("\n--- SU2 Log (last 20 lines) ---")
            print('\n'.join(log_content.split('\n')[-20:]))

        assert success, f"SU2_CFD RANS failed. Check {log_path}"


class TestDeLavalTransientIntegration:
    """Integration tests for transient de Laval nozzle simulation."""

    def test_full_workflow_transient_euler(self, gui, qapp, tmp_path: Path):
        """
        Test complete workflow: de Laval geometry -> mesh -> transient EULER simulation.
        """
        # Step 1: Load geometry
        gui.load_template("de_laval")
        assert len(gui.geometry.elements) > 0

        # Step 2: Generate mesh
        _click(gui, qapp, " Generate Mesh")
        assert gui.current_mesh_data is not None

        # Step 3: Configure transient EULER
        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        case_dir = tmp_path / "transient_euler_case"
        gui.case_directory.setText(str(case_dir))

        gui.simulation_mode.setCurrentText("Transient")
        qapp.processEvents()

        gui.solver_type.setCurrentText("EULER")
        qapp.processEvents()

        # Set time stepping parameters
        gui.time_step.setValue(1e-6)
        gui.end_time.setValue(1e-5)  # Very short for testing

        gui.inlet_pressure.setValue(300000)
        gui.temperature.setValue(300)
        gui.n_processors.setValue(1)

        # Step 4: Setup case files
        _click(gui, qapp, "⚙️ Generate Case Files")

        cd = Path(gui.current_case_directory)
        assert (cd / "config.cfg").exists()

        config = (cd / "config.cfg").read_text()
        assert "SOLVER= EULER" in config
        assert "TIME_DOMAIN= YES" in config

        # Step 5: Modify for quick test
        _modify_config_for_quick_test(cd / "config.cfg", max_iter=5)

        # Step 6: Run SU2
        from src.core.su2_runner import SU2Runner
        
        runner = SU2Runner(str(cd))
        success = runner.run_solver(n_processors=1)

        log_path = cd / "log.SU2_CFD"
        if log_path.exists():
            log_content = log_path.read_text()
            print("\n--- SU2 Log (last 20 lines) ---")
            print('\n'.join(log_content.split('\n')[-20:]))

        assert success, f"SU2_CFD transient failed. Check {log_path}"

    def test_full_workflow_transient_rans(self, gui, qapp, tmp_path: Path):
        """
        Test complete workflow: de Laval geometry -> mesh -> transient RANS (URANS).
        """
        # Step 1: Load geometry
        gui.load_template("de_laval")

        # Step 2: Generate mesh
        _click(gui, qapp, " Generate Mesh")
        assert gui.current_mesh_data is not None

        # Step 3: Configure transient RANS
        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        case_dir = tmp_path / "transient_rans_case"
        gui.case_directory.setText(str(case_dir))

        gui.simulation_mode.setCurrentText("Transient")
        qapp.processEvents()

        gui.solver_type.setCurrentText("RANS")
        qapp.processEvents()

        gui.turbulence_model.setCurrentText("SST")
        gui.time_step.setValue(1e-6)
        gui.end_time.setValue(1e-5)

        gui.inlet_pressure.setValue(300000)
        gui.temperature.setValue(300)
        gui.n_processors.setValue(1)

        # Step 4: Setup case files
        _click(gui, qapp, "⚙️ Generate Case Files")

        cd = Path(gui.current_case_directory)
        config = (cd / "config.cfg").read_text()
        assert "SOLVER= RANS" in config
        assert "TIME_DOMAIN= YES" in config
        assert "KIND_TURB_MODEL= SST" in config

        # Step 5: Modify for quick test
        _modify_config_for_quick_test(cd / "config.cfg", max_iter=5)

        # Step 6: Run SU2
        from src.core.su2_runner import SU2Runner
        
        runner = SU2Runner(str(cd))
        success = runner.run_solver(n_processors=1)

        log_path = cd / "log.SU2_CFD"
        if log_path.exists():
            log_content = log_path.read_text()
            print("\n--- SU2 Log (last 20 lines) ---")
            print('\n'.join(log_content.split('\n')[-20:]))

        assert success, f"SU2_CFD URANS failed. Check {log_path}"


class TestSU2RunnerIntegration:
    """Integration tests for SU2Runner with real SU2 executable."""

    def test_su2_cfd_executable_exists(self):
        """Verify SU2_CFD executable is available."""
        su2_path = shutil.which("SU2_CFD")
        assert su2_path is not None, "SU2_CFD should be in PATH"
        assert os.path.isfile(su2_path), "SU2_CFD should be a file"

    def test_su2_cfd_can_show_help(self):
        """Verify SU2_CFD can be executed and shows help."""
        import subprocess
        
        result = subprocess.run(
            ["SU2_CFD", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # SU2_CFD --help should succeed
        assert result.returncode == 0, f"SU2_CFD --help failed: {result.stderr}"
        assert "SU2" in result.stdout or "configfile" in result.stdout.lower()

    def test_runner_detects_missing_mesh(self, tmp_path: Path):
        """Test that SU2Runner correctly validates missing mesh."""
        from src.core.su2_runner import SU2Runner
        
        case_dir = tmp_path / "incomplete_case"
        case_dir.mkdir()
        
        # Create config without mesh
        config = """% SU2 Config
SOLVER= EULER
MESH_FILENAME= mesh.su2
ITER= 10
"""
        (case_dir / "config.cfg").write_text(config)
        
        runner = SU2Runner(str(case_dir))
        valid, msg = runner.validate_case()
        
        assert not valid, "Should fail validation without mesh"
        assert "mesh" in msg.lower()

    def test_runner_validates_complete_case(self, tmp_path: Path):
        """Test that SU2Runner validates a complete case."""
        from src.core.su2_runner import SU2Runner
        
        case_dir = tmp_path / "complete_case"
        case_dir.mkdir()
        
        # Create minimal valid config and mesh
        config = """% SU2 Config
SOLVER= EULER
MATH_PROBLEM= DIRECT
MACH_NUMBER= 2.0
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 300.0
MESH_FILENAME= mesh.su2
MARKER_EULER= ( wall )
ITER= 10
"""
        (case_dir / "config.cfg").write_text(config)
        
        # Minimal mesh (single triangle)
        mesh = """% SU2 Mesh
NDIME= 2
NELEM= 1
5 0 1 2 0
NPOIN= 3
0.0 0.0 0
1.0 0.0 1
0.5 1.0 2
NMARK= 1
MARKER_TAG= wall
MARKER_ELEMS= 3
3 0 1 0
3 1 2 1
3 2 0 2
"""
        (case_dir / "mesh.su2").write_text(mesh)
        
        runner = SU2Runner(str(case_dir))
        valid, msg = runner.validate_case()
        
        assert valid, f"Should pass validation: {msg}"


class TestParallelExecution:
    """Test parallel SU2 execution with MPI."""

    @pytest.mark.skipif(
        shutil.which("mpirun") is None,
        reason="mpirun not available"
    )
    def test_parallel_euler_simulation(self, gui, qapp, tmp_path: Path):
        """Test parallel SU2 execution with 2 processors."""
        gui.load_template("de_laval")
        _click(gui, qapp, " Generate Mesh")

        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        case_dir = tmp_path / "parallel_case"
        gui.case_directory.setText(str(case_dir))

        gui.simulation_mode.setCurrentText("Steady")
        gui.solver_type.setCurrentText("EULER")
        gui.inlet_pressure.setValue(300000)
        gui.temperature.setValue(300)
        gui.n_processors.setValue(2)  # Parallel!
        qapp.processEvents()

        _click(gui, qapp, "⚙️ Generate Case Files")

        cd = Path(gui.current_case_directory)
        _modify_config_for_quick_test(cd / "config.cfg", max_iter=10)

        from src.core.su2_runner import SU2Runner
        
        runner = SU2Runner(str(cd))
        success = runner.run_solver(n_processors=2)

        log_path = cd / "log.SU2_CFD"
        if log_path.exists():
            log_content = log_path.read_text()
            print("\n--- SU2 Parallel Log (last 20 lines) ---")
            print('\n'.join(log_content.split('\n')[-20:]))

        assert success, f"Parallel SU2_CFD failed. Check {log_path}"
