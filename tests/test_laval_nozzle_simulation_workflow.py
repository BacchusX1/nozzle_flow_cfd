"""
End-to-end GUI workflow tests for de Laval nozzle simulation.

These tests verify the complete workflow from geometry setup through
mesh generation to simulation start for both steady and transient modes.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
    from frontend import frontend
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


@pytest.fixture()
def mock_mesh_generator(monkeypatch):
    """Mock mesh generator for fast deterministic meshing."""
    from backend.meshing import mesh_generator as mg

    mesh_data = {
        "nodes": [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)],
        "elements": [(0, 1, 2, 3)],
        "stats": {"num_nodes": 4, "num_elements": 1, "element_type": "quad", "mesh_quality": 0.9},
    }

    class DummyGenerator:
        def __init__(self):
            self._stats = mesh_data["stats"]

        def generate_mesh(self, geometry, params=None):
            return mesh_data

        def get_mesh_statistics(self):
            return dict(self._stats)

        def analyze_mesh_quality(self, mesh_data_in):
            return {
                "num_elements": 1,
                "num_nodes": 4,
                "min_quality": 0.9,
                "avg_quality": 0.9,
                "max_aspect_ratio": 1.0,
                "skewness": 0.0,
                "orthogonality": 1.0,
            }

    monkeypatch.setattr(mg, "AdvancedMeshGenerator", DummyGenerator)
    return mesh_data


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


class TestDeLavalSteadySimulation:
    """Test steady-state simulation workflow for de Laval nozzle."""

    def test_setup_mesh_and_case_files_for_steady_rans(
        self, gui, qapp, mock_mesh_generator, tmp_path: Path
    ):
        """Test: Load de Laval template -> Generate mesh -> Setup steady RANS case."""
        # Step 1: Load de Laval nozzle geometry template
        gui.load_template("de_laval")
        assert len(gui.geometry.elements) > 0, "De Laval template should have geometry elements"

        # Step 2: Generate mesh
        _click(gui, qapp, " Generate Mesh")
        assert gui.current_mesh_data is not None, "Mesh data should be generated"

        # Step 3: Switch to Simulation tab and configure steady RANS
        gui.tab_widget.setCurrentIndex(2)  # Simulation tab
        qapp.processEvents()

        case_dir = tmp_path / "steady_case"
        gui.case_directory.setText(str(case_dir))

        # Set steady mode
        gui.simulation_mode.setCurrentText("Steady")
        qapp.processEvents()

        # Set RANS solver
        gui.solver_type.setCurrentText("RANS")
        qapp.processEvents()

        # Set flow conditions
        gui.inlet_pressure.setValue(300000)  # 3 bar for supersonic nozzle
        gui.temperature.setValue(300)
        gui.n_processors.setValue(1)

        # Step 4: Setup case files
        _click(gui, qapp, "⚙️ Generate Case Files")

        # Verify case structure
        assert gui.current_case_directory is not None
        cd = Path(gui.current_case_directory)
        assert (cd / "config.cfg").exists(), "Config file should exist"

        config = (cd / "config.cfg").read_text(encoding="utf-8", errors="ignore")
        assert "SOLVER= RANS" in config, "Config should specify RANS solver"
        assert "KIND_TURB_MODEL" in config, "RANS solver should have turbulence model"
        # Steady state should NOT have TIME_DOMAIN or should be NO
        assert "TIME_DOMAIN= NO" in config or "TIME_DOMAIN" not in config

    def test_run_steady_simulation_calls_su2_runner(
        self, gui, qapp, mock_mesh_generator, monkeypatch, tmp_path: Path
    ):
        """Test: Complete steady simulation workflow invokes SU2Runner correctly."""
        # Setup
        gui.load_template("de_laval")
        _click(gui, qapp, " Generate Mesh")

        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        case_dir = tmp_path / "steady_run_case"
        gui.case_directory.setText(str(case_dir))
        gui.simulation_mode.setCurrentText("Steady")
        gui.solver_type.setCurrentText("RANS")
        gui.n_processors.setValue(1)
        qapp.processEvents()

        _click(gui, qapp, "⚙️ Generate Case Files")

        # Mock SU2Runner - must patch where it's imported (core.su2_runner)
        calls = {"run_solver": 0, "n_processors": None}

        original_su2runner = None
        
        # Import the actual module to get the original class
        from core import su2_runner as su2_runner_module
        original_su2runner = su2_runner_module.SU2Runner

        class MockSU2Runner(original_su2runner):
            def validate_case(self):
                return True, "Valid"
            
            def run_solver(self, n_processors: int = 1):
                calls["run_solver"] += 1
                calls["n_processors"] = n_processors
                return True

        monkeypatch.setattr(su2_runner_module, "SU2Runner", MockSU2Runner)

        # Run simulation
        _click(gui, qapp, "▶️ Run Simulation")

        assert calls["run_solver"] == 1, "run_solver should be called once"
        assert calls["n_processors"] == 1, "Should use 1 processor"

    def test_steady_euler_solver_configuration(
        self, gui, qapp, mock_mesh_generator, tmp_path: Path
    ):
        """Test steady EULER solver creates correct config."""
        gui.load_template("de_laval")
        _click(gui, qapp, " Generate Mesh")

        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        case_dir = tmp_path / "euler_case"
        gui.case_directory.setText(str(case_dir))
        gui.simulation_mode.setCurrentText("Steady")
        gui.solver_type.setCurrentText("EULER")
        qapp.processEvents()

        _click(gui, qapp, "⚙️ Generate Case Files")

        cd = Path(gui.current_case_directory)
        config = (cd / "config.cfg").read_text(encoding="utf-8", errors="ignore")
        
        assert "SOLVER= EULER" in config, "Config should specify EULER solver"
        # EULER should not have turbulence model or should be NONE
        assert "KIND_TURB_MODEL= NONE" in config or "KIND_TURB_MODEL" not in config


class TestDeLavalTransientSimulation:
    """Test transient simulation workflow for de Laval nozzle."""

    def test_setup_mesh_and_case_files_for_transient_rans(
        self, gui, qapp, mock_mesh_generator, tmp_path: Path
    ):
        """Test: Load de Laval template -> Generate mesh -> Setup transient RANS case."""
        # Step 1: Load de Laval nozzle geometry template
        gui.load_template("de_laval")
        assert len(gui.geometry.elements) > 0

        # Step 2: Generate mesh
        _click(gui, qapp, " Generate Mesh")
        assert gui.current_mesh_data is not None

        # Step 3: Switch to Simulation tab and configure transient RANS
        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        case_dir = tmp_path / "transient_case"
        gui.case_directory.setText(str(case_dir))

        # Set transient mode
        gui.simulation_mode.setCurrentText("Transient")
        qapp.processEvents()

        # Set RANS solver (URANS)
        gui.solver_type.setCurrentText("RANS")
        qapp.processEvents()

        # Set flow conditions
        gui.inlet_pressure.setValue(300000)
        gui.temperature.setValue(300)
        gui.n_processors.setValue(1)

        # Set transient-specific settings
        gui.time_step.setValue(1e-6)
        gui.end_time.setValue(0.001)

        # Step 4: Setup case files
        _click(gui, qapp, "⚙️ Generate Case Files")

        # Verify case structure
        assert gui.current_case_directory is not None
        cd = Path(gui.current_case_directory)
        assert (cd / "config.cfg").exists()

        config = (cd / "config.cfg").read_text(encoding="utf-8", errors="ignore")
        assert "SOLVER= RANS" in config
        # Transient simulation should have TIME_DOMAIN= YES
        assert "TIME_DOMAIN= YES" in config, "Transient config should have TIME_DOMAIN= YES"

    def test_run_transient_simulation_calls_su2_runner(
        self, gui, qapp, mock_mesh_generator, monkeypatch, tmp_path: Path
    ):
        """Test: Complete transient simulation workflow invokes SU2Runner correctly."""
        # Setup
        gui.load_template("de_laval")
        _click(gui, qapp, " Generate Mesh")

        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        case_dir = tmp_path / "transient_run_case"
        gui.case_directory.setText(str(case_dir))
        gui.simulation_mode.setCurrentText("Transient")
        gui.solver_type.setCurrentText("RANS")
        gui.n_processors.setValue(2)
        gui.time_step.setValue(1e-6)
        gui.end_time.setValue(0.001)
        qapp.processEvents()

        _click(gui, qapp, "⚙️ Generate Case Files")

        # Mock SU2Runner - must patch where it's imported (core.su2_runner)
        calls = {"run_solver": 0, "n_processors": None}

        from core import su2_runner as su2_runner_module
        original_su2runner = su2_runner_module.SU2Runner

        class MockSU2Runner(original_su2runner):
            def validate_case(self):
                return True, "Valid"
            
            def run_solver(self, n_processors: int = 1):
                calls["run_solver"] += 1
                calls["n_processors"] = n_processors
                return True

        monkeypatch.setattr(su2_runner_module, "SU2Runner", MockSU2Runner)

        # Run simulation
        _click(gui, qapp, "▶️ Run Simulation")

        assert calls["run_solver"] == 1, "run_solver should be called once"
        assert calls["n_processors"] == 2, "Should use 2 processors"

    def test_transient_mode_shows_time_controls(self, gui, qapp):
        """Test that transient mode shows time-stepping UI controls."""
        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        # Switch to transient using the radio button
        gui.transient_radio.setChecked(True)
        qapp.processEvents()

        # Transient group should be visible (not hidden), steady group should be hidden
        assert not gui.transient_group.isHidden(), "Transient group should be visible in transient mode"
        assert gui.steady_group.isHidden(), "Steady group should be hidden in transient mode"

    def test_steady_mode_shows_cfl_control(self, gui, qapp):
        """Test that steady mode shows CFL control and hides time controls."""
        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        # Ensure steady mode using the radio button
        gui.steady_radio.setChecked(True)
        qapp.processEvents()

        # Steady group should be visible, transient group should be hidden
        assert not gui.steady_group.isHidden(), "Steady group should be visible in steady mode"
        assert gui.transient_group.isHidden(), "Transient group should be hidden in steady mode"


class TestSU2ExecutableValidation:
    """Test validation of SU2 executable availability."""

    def test_run_simulation_fails_gracefully_without_su2(
        self, gui, qapp, mock_mesh_generator, tmp_path: Path
    ):
        """Test that simulation fails gracefully when SU2_CFD is not available."""
        gui.load_template("de_laval")
        _click(gui, qapp, " Generate Mesh")

        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        case_dir = tmp_path / "no_su2_case"
        gui.case_directory.setText(str(case_dir))
        gui.simulation_mode.setCurrentText("Steady")
        gui.solver_type.setCurrentText("RANS")
        qapp.processEvents()

        _click(gui, qapp, "⚙️ Generate Case Files")

        # Don't mock SU2Runner - let it try to find the real executable
        # The run should fail gracefully with an error message in the log
        
        from backend.simulation.su2_runner import SU2Runner
        runner = SU2Runner(str(case_dir))
        
        # Manually check executable availability
        import shutil
        su2_available = shutil.which("SU2_CFD") is not None
        
        if not su2_available:
            # The runner should detect missing executable
            result = runner.run_solver(n_processors=1)
            assert result is False, "run_solver should return False when SU2_CFD not found"
            
            # Check log file for error message
            log_path = case_dir / "log.SU2_CFD"
            if log_path.exists():
                log_content = log_path.read_text()
                assert "not found" in log_content.lower() or "error" in log_content.lower()

    def test_su2_runner_validates_case_before_running(
        self, gui, qapp, mock_mesh_generator, tmp_path: Path
    ):
        """Test that SU2Runner validates case files before attempting to run."""
        from backend.simulation.su2_runner import SU2Runner
        
        # Create case without mesh file
        case_dir = tmp_path / "invalid_case"
        case_dir.mkdir()
        
        config_content = """% SU2 Configuration
SOLVER= RANS
MESH_FILENAME= mesh.su2
"""
        (case_dir / "config.cfg").write_text(config_content)
        # Note: mesh.su2 does NOT exist
        
        runner = SU2Runner(str(case_dir))
        valid, message = runner.validate_case()
        
        assert not valid, "Validation should fail when mesh is missing"
        assert "mesh" in message.lower(), "Error message should mention missing mesh"


class TestTurbulenceModelVisibility:
    """Test turbulence model UI visibility based on solver type."""

    def test_rans_solver_shows_turbulence_model(self, gui, qapp):
        """Test that RANS solver shows turbulence model dropdown."""
        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        gui.solver_type.setCurrentText("RANS")
        qapp.processEvents()

        assert gui.turbulence_model.isVisible(), "Turbulence model should be visible for RANS"
        assert gui.lbl_turbulence.isVisible(), "Turbulence label should be visible for RANS"

    def test_euler_solver_hides_turbulence_model(self, gui, qapp):
        """Test that EULER solver hides turbulence model dropdown."""
        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        gui.solver_type.setCurrentText("EULER")
        qapp.processEvents()

        assert not gui.turbulence_model.isVisible(), "Turbulence model should be hidden for EULER"
        assert not gui.lbl_turbulence.isVisible(), "Turbulence label should be hidden for EULER"

    def test_navier_stokes_solver_hides_turbulence_model(self, gui, qapp):
        """Test that NAVIER_STOKES (laminar) solver hides turbulence model."""
        gui.tab_widget.setCurrentIndex(2)
        qapp.processEvents()

        gui.solver_type.setCurrentText("NAVIER_STOKES")
        qapp.processEvents()

        assert not gui.turbulence_model.isVisible(), "Turbulence model should be hidden for laminar"
