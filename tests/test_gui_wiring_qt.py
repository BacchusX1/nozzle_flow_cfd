from __future__ import annotations

from pathlib import Path

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
    import frontend
    from PySide6.QtWidgets import QWidget

    # We are intentionally not testing Post-processing yet; avoid
    # constructing heavy Matplotlib canvases during window initialization.
    def _stub_post_tab(self):
        return QWidget()

    frontend.NozzleDesignGUI.create_postprocessing_tab = _stub_post_tab

    window = frontend.NozzleDesignGUI()
    window.show()
    # Let Qt process show/paint events
    qapp.processEvents()
    yield window
    window.close()
    qapp.processEvents()


def _find_button(window, text: str):
    from PySide6.QtWidgets import QPushButton

    matches = [b for b in window.findChildren(QPushButton) if b.text() == text]
    assert matches, f"Button not found: {text}"
    assert len(matches) == 1, f"Multiple buttons found for text={text}: {len(matches)}"
    return matches[0]


def _click(window, qapp, button_text: str):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    btn = _find_button(window, button_text)
    assert btn.isEnabled(), f"Button disabled: {button_text}"
    QTest.mouseClick(btn, Qt.LeftButton)
    qapp.processEvents()


def test_gui_has_expected_tabs(gui):
    assert hasattr(gui, "tab_widget")
    tab = gui.tab_widget

    # Post-processing tab exists, but we won't test its behavior yet.
    labels = [tab.tabText(i) for i in range(tab.count())]
    # App currently uses these labels in the modern tab widget.
    assert "Geometry Design" in labels
    assert "Mesh Generation" in labels
    assert "Simulation" in labels
    assert "Results" in labels


def test_generate_mesh_button_wires_mesh_data(gui, qapp, monkeypatch, tmp_path: Path):
    # Ensure we have a valid geometry without manual drawing.
    gui.load_template("converging")

    # Patch the mesher to be deterministic and fast.
    from core.modules import mesh_generator as mg

    mesh_data = {
        "nodes": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
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

    _click(gui, qapp, " Generate Mesh")

    assert gui.current_mesh_data is not None
    assert (gui.current_mesh_data.get("nodes") or gui.current_mesh_data.get("vertices"))

    # Smoke-check stats text updated
    assert hasattr(gui, "mesh_stats")
    text = gui.mesh_stats.toPlainText()
    assert "Total elements" in text
    assert "Total nodes" in text


def test_geometry_clear_undo_validate_finish(gui, qapp):
    # Start with a populated geometry
    gui.load_template("converging")
    initial = len(gui.geometry.elements)
    assert initial > 0

    # Validate should not crash and should keep geometry
    _click(gui, qapp, "Validate")
    assert len(gui.geometry.elements) == initial

    # Undo removes one element
    _click(gui, qapp, "Undo")
    assert len(gui.geometry.elements) == initial - 1

    # Finish Geo switches to mesh tab
    _click(gui, qapp, "Finish Geo")
    assert gui.tab_widget.currentIndex() == 1

    # Back to geometry tab and clear
    gui.tab_widget.setCurrentIndex(0)
    qapp.processEvents()
    _click(gui, qapp, "Clear")
    assert len(gui.geometry.elements) == 0


def test_mesh_analyze_and_export_buttons(gui, qapp, monkeypatch, tmp_path: Path):
    gui.load_template("converging")

    # Patch the mesher to be deterministic and fast.
    from core.modules import mesh_generator as mg

    mesh_data = {
        "nodes": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        "elements": [(0, 1, 2, 3)],
        "stats": {"num_nodes": 4, "num_elements": 1, "element_type": "quad", "mesh_quality": 0.9},
    }

    calls = {"analyze": 0}

    class DummyGenerator:
        def __init__(self):
            self._stats = mesh_data["stats"]

        def generate_mesh(self, geometry, params=None):
            return mesh_data

        def get_mesh_statistics(self):
            return dict(self._stats)

        def analyze_mesh_quality(self, mesh_data_in):
            calls["analyze"] += 1
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

    # Generate mesh first
    _click(gui, qapp, " Generate Mesh")
    assert gui.current_mesh_data is not None

    # Analyze mesh quality
    _click(gui, qapp, "[Chart] Analyze Mesh Quality")
    assert calls["analyze"] == 1

    # Export mesh via QFileDialog patch
    from PySide6.QtWidgets import QFileDialog

    out_path = tmp_path / "mesh_out.msh"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_path), "MSH Files (*.msh)"))

    _click(gui, qapp, "[Save] Export Mesh")

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8", errors="ignore")
    assert "$MeshFormat" in content


def test_setup_case_files_button_generates_case(gui, qapp, tmp_path: Path):
    gui.load_template("converging")

    case_dir = tmp_path / "case"
    gui.case_directory.setText(str(case_dir))

    # Pick a non-default solver and laminar to ensure wiring works.
    gui.solver_type.setCurrentText("sonicFoam")
    gui.turbulence_model.setCurrentText("laminar")
    gui.inlet_pressure.setValue(150000)
    gui.temperature.setValue(350)

    _click(gui, qapp, "[Settings] Setup Case Files")

    assert gui.current_case_directory
    cd = Path(gui.current_case_directory)
    assert (cd / "system" / "controlDict").exists()
    assert (cd / "system" / "fvSchemes").exists()
    assert (cd / "system" / "fvSolution").exists()

    control_dict = (cd / "system" / "controlDict").read_text(encoding="utf-8", errors="ignore")
    assert "application" in control_dict
    assert "sonicFoam" in control_dict

    turb_props = (cd / "constant" / "turbulenceProperties").read_text(encoding="utf-8", errors="ignore")
    assert "simulationType laminar" in turb_props


def test_run_simulation_aborts_on_solver_mismatch(gui, qapp, monkeypatch, tmp_path: Path):
    gui.load_template("converging")

    case_dir = tmp_path / "case"
    gui.case_directory.setText(str(case_dir))
    gui.solver_type.setCurrentText("simpleFoam")
    _click(gui, qapp, "[Settings] Setup Case Files")

    # Change UI solver without regenerating case -> should abort before blockMesh.
    gui.solver_type.setCurrentText("sonicFoam")

    from core.openfoam_runner import OpenFOAMRunner

    def _boom(*a, **k):
        raise AssertionError("block_mesh should not be called on solver mismatch")

    monkeypatch.setattr(OpenFOAMRunner, "block_mesh", _boom)

    _click(gui, qapp, "[Run] Run Simulation")

    # Ensure log mentions mismatch
    assert "mismatch" in gui.simulation_log.toPlainText().lower()


def test_run_simulation_calls_runner_methods(gui, qapp, monkeypatch, tmp_path: Path):
    gui.load_template("converging")

    case_dir = tmp_path / "case"
    gui.case_directory.setText(str(case_dir))
    gui.solver_type.setCurrentText("simpleFoam")
    gui.n_processors.setValue(1)

    _click(gui, qapp, "[Settings] Setup Case Files")

    from core.openfoam_runner import OpenFOAMRunner

    calls = {"block": 0, "run": 0}

    def fake_block(self):
        calls["block"] += 1
        return True

    def fake_run(self, solver: str = "simpleFoam", n_processors: int = 1):
        calls["run"] += 1
        # Should use configured solver (from controlDict)
        assert solver == "simpleFoam"
        assert n_processors == 1
        return True

    monkeypatch.setattr(OpenFOAMRunner, "block_mesh", fake_block)
    monkeypatch.setattr(OpenFOAMRunner, "run_solver", fake_run)
    monkeypatch.setattr(OpenFOAMRunner, "decompose_par", lambda self: True)
    monkeypatch.setattr(OpenFOAMRunner, "reconstruct_par", lambda self: True)

    _click(gui, qapp, "[Run] Run Simulation")

    assert calls["block"] == 1
    assert calls["run"] == 1


def test_solver_type_visibility_toggle(gui, qapp):
    """Test that transient solver controls visibility changes based on solver type."""
    # Start with default steady-state solver
    gui.solver_type.setCurrentText("simpleFoam")
    gui._on_solver_type_changed("simpleFoam")  # Explicitly call the slot
    qapp.processEvents()
    
    # Transient controls should be hidden for steady-state solvers
    # Use isHidden() which checks the widget's own hidden flag, not parent visibility
    assert gui.time_step.isHidden()
    assert gui.end_time.isHidden()
    assert gui.n_outer_correctors.isHidden()
    assert gui.n_correctors.isHidden()
    assert gui.max_courant.isHidden()
    
    # Switch to transient compressible solver
    gui.solver_type.setCurrentText("sonicFoam")
    gui._on_solver_type_changed("sonicFoam")  # Explicitly call the slot
    qapp.processEvents()
    
    # Transient controls should NOT be hidden
    assert not gui.time_step.isHidden()
    assert not gui.end_time.isHidden()
    assert not gui.n_outer_correctors.isHidden()
    assert not gui.n_correctors.isHidden()
    # Max Courant should NOT be hidden for compressible solvers
    assert not gui.max_courant.isHidden()
    
    # Switch back to steady-state
    gui.solver_type.setCurrentText("rhoSimpleFoam")
    gui._on_solver_type_changed("rhoSimpleFoam")  # Explicitly call the slot
    qapp.processEvents()
    
    # Transient controls should be hidden again
    assert gui.time_step.isHidden()
    assert gui.n_outer_correctors.isHidden()


def test_sonic_foam_settings_wired_to_case(gui, qapp, tmp_path: Path):
    """Test that sonicFoam time step and PIMPLE settings are correctly wired to case files."""
    gui.load_template("converging")

    case_dir = tmp_path / "case"
    gui.case_directory.setText(str(case_dir))

    # Configure sonicFoam with specific settings
    gui.solver_type.setCurrentText("sonicFoam")
    qapp.processEvents()
    
    gui.time_step.setValue(5e-8)
    gui.end_time.setValue(0.002)
    gui.max_courant.setValue(0.6)
    gui.n_outer_correctors.setValue(3)
    gui.n_correctors.setValue(4)
    gui.convergence_tolerance.setValue(1e-7)

    _click(gui, qapp, "[Settings] Setup Case Files")

    cd = Path(gui.current_case_directory)
    
    # Check controlDict has correct time settings
    control_dict = (cd / "system" / "controlDict").read_text(encoding="utf-8", errors="ignore")
    assert "sonicFoam" in control_dict
    assert "5e-08" in control_dict  # deltaT
    assert "0.002" in control_dict  # endTime
    assert "maxCo           0.6" in control_dict
    assert "adjustTimeStep  yes" in control_dict
    
    # Check fvSolution has correct PIMPLE settings
    fv_solution = (cd / "system" / "fvSolution").read_text(encoding="utf-8", errors="ignore")
    assert "PIMPLE" in fv_solution
    assert "nOuterCorrectors    3" in fv_solution
    assert "nCorrectors         4" in fv_solution
    
    # Check pressure has compressible dimensions
    p_file = (cd / "0" / "p").read_text(encoding="utf-8", errors="ignore")
    assert "[1 -1 -2 0 0 0 0]" in p_file


def test_steady_solver_settings_wired_to_case(gui, qapp, tmp_path: Path):
    """Test that steady-state solver settings are correctly wired to case files."""
    gui.load_template("converging")

    case_dir = tmp_path / "case"
    gui.case_directory.setText(str(case_dir))

    # Configure simpleFoam with specific settings
    gui.solver_type.setCurrentText("simpleFoam")
    qapp.processEvents()
    
    gui.max_iterations.setValue(2000)
    gui.convergence_tolerance.setValue(1e-5)

    _click(gui, qapp, "[Settings] Setup Case Files")

    cd = Path(gui.current_case_directory)
    
    # Check controlDict
    control_dict = (cd / "system" / "controlDict").read_text(encoding="utf-8", errors="ignore")
    assert "simpleFoam" in control_dict
    assert "endTime         2000" in control_dict
    
    # Check fvSolution uses SIMPLE (not PIMPLE)
    fv_solution = (cd / "system" / "fvSolution").read_text(encoding="utf-8", errors="ignore")
    assert "SIMPLE" in fv_solution
    assert "PIMPLE" not in fv_solution
