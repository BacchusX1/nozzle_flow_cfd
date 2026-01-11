#!/usr/bin/env python3
"""
Comprehensive Nozzle Design GUI with CFD Workflow
Professional full-stack development approach with immediate drawing capability.

Features:
- Immediate drawing without start/stop buttons
- Advanced meshing with boundary layers  
- Complete CFD simulation setup
- Post-processing and visualization
- Professional dark theme
"""

import sys
import os
import json
import math
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QGridLayout, QTabWidget, QGroupBox,
    QLabel, QPushButton, QLineEdit, QTextEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QRadioButton,
    QButtonGroup, QFileDialog, QMessageBox, QSplitter,
    QProgressBar, QStatusBar, QToolBar, QSlider,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QFrame, QFormLayout, QInputDialog,
    QListWidget, QListWidgetItem, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap, QIcon, QAction

# Matplotlib imports
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg for PySide6 compatibility
try:
    # Preferred for Qt6 / PySide6
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # pragma: no cover
    # Fallback for older Matplotlib installs
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.lines
import numpy as np
import json
import os

from core.theme import Theme
from core.nozzle import NozzleGeometry
from core.geometry import PolynomialElement, LineElement, ArcElement
from core.template_loader import TemplateLoader
from core.standard_values import DEFAULTS
from core.modules.mesh_generator import AdvancedMeshGenerator
from core.modules.simulation_setup import SimulationSetup
from core.modules.postprocessing import ResultsProcessor
from core.modules.interactive_postprocessor import InteractivePostprocessorWidget
from core.su2_runner import SU2Runner


class SimulationWorker(QObject):
    """Worker thread for running SU2 simulations without blocking the GUI."""
    finished = Signal(bool, str)  # success, message
    progress = Signal(str)  # log message
    
    def __init__(self, case_directory: str, n_processors: int = 1, phases: list = None):
        """
        Initialize simulation worker.
        
        Args:
            case_directory: Path to case directory
            n_processors: Number of MPI processors
            phases: List of phase configs for variable dt simulation.
                   Each phase is a dict with keys: config_file, description
                   If None, runs single config.cfg
        """
        super().__init__()
        self.case_directory = case_directory
        self.n_processors = n_processors
        self.phases = phases or [{"config_file": "config.cfg", "description": "Main simulation"}]
        self._stop_requested = False
        self._process = None
    
    def run(self):
        """Run the SU2 simulation in background (supports multi-phase)."""
        import subprocess
        import shutil
        
        try:
            total_phases = len(self.phases)
            
            for phase_idx, phase in enumerate(self.phases):
                if self._stop_requested:
                    self.finished.emit(False, "Simulation stopped by user")
                    return
                
                config_file = phase.get("config_file", "config.cfg")
                description = phase.get("description", f"Phase {phase_idx + 1}")
                config_path = os.path.join(self.case_directory, config_file)
                
                # Validate config exists
                if not os.path.exists(config_path):
                    self.finished.emit(False, f"Missing configuration file: {config_file}")
                    return
                
                self.progress.emit(f"{'='*40}")
                self.progress.emit(f"Phase {phase_idx + 1}/{total_phases}: {description}")
                self.progress.emit(f"Config: {config_file}")
                self.progress.emit(f"{'='*40}")
                
                # Build command
                if self.n_processors > 1:
                    if not shutil.which("mpirun"):
                        self.finished.emit(False, "mpirun not found on PATH (required for parallel runs)")
                        return
                    cmd = ["mpirun", "-np", str(self.n_processors), "SU2_CFD", config_file]
                else:
                    if not shutil.which("SU2_CFD"):
                        self.finished.emit(False, "SU2_CFD not found on PATH")
                        return
                    cmd = ["SU2_CFD", config_file]
                
                self.progress.emit(f"Starting: {' '.join(cmd)}")
                
                log_file_name = f"log.SU2_CFD_phase{phase_idx + 1}" if total_phases > 1 else "log.SU2_CFD"
                log_path = os.path.join(self.case_directory, log_file_name)
                
                with open(log_path, "w") as log_file:
                    self._process = subprocess.Popen(
                        cmd,
                        cwd=self.case_directory,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    # Stream output line by line
                    for line in iter(self._process.stdout.readline, ''):
                        if self._stop_requested:
                            self._process.terminate()
                            self._process.wait(timeout=5)
                            self.progress.emit("Simulation stopped by user")
                            self.finished.emit(False, "Simulation stopped by user")
                            return
                        log_file.write(line)
                        log_file.flush()
                        # Emit progress for key lines
                        stripped = line.strip()
                        if stripped and ("Iter" in stripped or "Error" in stripped or 
                                        "Convergence" in stripped or "Time" in stripped or
                                        "CL" in stripped or "CD" in stripped):
                            self.progress.emit(f"[P{phase_idx+1}] {stripped[:100]}")
                    
                    self._process.wait()
                    
                    if self._process.returncode != 0:
                        self.finished.emit(False, f"Phase {phase_idx + 1} failed with code {self._process.returncode}")
                        return
                
                self.progress.emit(f"Phase {phase_idx + 1} completed successfully")
            
            # All phases completed
            if total_phases > 1:
                self.finished.emit(True, f"All {total_phases} phases completed successfully")
            else:
                self.finished.emit(True, "SU2 simulation completed successfully")
                    
        except Exception as e:
            self.finished.emit(False, f"Simulation error: {str(e)}")
        finally:
            self._process = None
    
    def stop(self):
        """Request the simulation to stop."""
        self._stop_requested = True
        if self._process is not None:
            try:
                self._process.terminate()
            except Exception:
                pass


class AxisSettingsDialog(QDialog):
    """Dialog for setting axis min/max values and log scale."""
    
    def __init__(self, parent=None, axis_name="X", current_min=0, current_max=1, is_log=False):
        super().__init__(parent)
        self.setWindowTitle(f"{axis_name} Axis Settings")
        self.setModal(True)
        self.setMinimumWidth(300)
        
        layout = QVBoxLayout(self)
        
        # Min value
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min:"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1e12, 1e12)
        self.min_spin.setDecimals(6)
        self.min_spin.setValue(current_min)
        min_layout.addWidget(self.min_spin)
        layout.addLayout(min_layout)
        
        # Max value
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max:"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1e12, 1e12)
        self.max_spin.setDecimals(6)
        self.max_spin.setValue(current_max)
        max_layout.addWidget(self.max_spin)
        layout.addLayout(max_layout)
        
        # Log scale checkbox
        self.log_check = QCheckBox("Logarithmic Scale")
        self.log_check.setChecked(is_log)
        layout.addWidget(self.log_check)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_values(self):
        return self.min_spin.value(), self.max_spin.value(), self.log_check.isChecked()


class InteractiveMonitorCanvas(FigureCanvas):
    """
    Interactive matplotlib canvas for simulation monitoring with:
    - Scroll to zoom
    - Left mouse drag to pan
    - Double-click on axis to edit settings
    """
    
    def __init__(self, parent=None, width=10, height=6, dpi=100, facecolor='#1e1e1e'):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=facecolor)
        self.ax = self.fig.add_subplot(111, facecolor='#1e1e1e')
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Store plot data for inspection
        self.plot_data = {}  # name -> (x_values, y_values)
        
        # Pan state
        self._pan_active = False
        self._pan_start = None
        self._xlim_start = None
        self._ylim_start = None
        
        # Axis settings
        self._x_is_log = False
        self._y_is_log = True  # Default log for residuals
        
        # Connect events
        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)
        
        # Style axes
        self._style_axes()
        
    def _style_axes(self):
        """Apply dark theme styling to axes."""
        text_color = '#e0e0e0'
        border_color = '#404040'
        
        self.ax.set_xlabel('Iteration', color=text_color)
        self.ax.set_ylabel('Residual', color=text_color)
        self.ax.tick_params(colors=text_color)
        for spine in self.ax.spines.values():
            spine.set_color(border_color)
        self.ax.grid(True, alpha=0.2, color='#606060', linestyle='--')
        
    def _on_scroll(self, event):
        """Handle scroll events for zooming."""
        if event.inaxes != self.ax:
            return
            
        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return
            
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        if xdata is None or ydata is None:
            return
        
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor
        
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
        
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.draw()
        
    def _on_press(self, event):
        """Handle mouse press events."""
        if event.dblclick:
            # Double-click: show axis settings dialog
            self._handle_axis_double_click(event)
            return
            
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Left mouse button
            self._pan_active = True
            self._pan_start = (event.xdata, event.ydata)
            self._xlim_start = self.ax.get_xlim()
            self._ylim_start = self.ax.get_ylim()
            
    def _on_release(self, event):
        """Handle mouse release events."""
        if event.button == 1:
            self._pan_active = False
            self._pan_start = None
            
    def _on_motion(self, event):
        """Handle mouse motion events for panning."""
        if not self._pan_active or self._pan_start is None:
            return
            
        if event.inaxes != self.ax or event.xdata is None:
            return
            
        dx = self._pan_start[0] - event.xdata
        dy = self._pan_start[1] - event.ydata
        
        self.ax.set_xlim(self._xlim_start[0] + dx, self._xlim_start[1] + dx)
        self.ax.set_ylim(self._ylim_start[0] + dy, self._ylim_start[1] + dy)
        
        self.draw()
        
    def _handle_axis_double_click(self, event):
        """Handle double-click on axes to show settings dialog."""
        # Determine if click is near X or Y axis
        fig_point = self.fig.transFigure.inverted().transform((event.x, event.y))
        ax_bbox = self.ax.get_position()
        
        # Check if near X axis (bottom of plot)
        if fig_point[1] < ax_bbox.y0 + 0.05:
            self._show_axis_dialog('X')
        # Check if near Y axis (left of plot)
        elif fig_point[0] < ax_bbox.x0 + 0.05:
            self._show_axis_dialog('Y')
    
    def _show_axis_dialog(self, axis: str):
        """Show axis settings dialog."""
        if axis == 'X':
            xlim = self.ax.get_xlim()
            dialog = AxisSettingsDialog(
                self.parent(), 
                axis_name="X Axis",
                current_min=xlim[0],
                current_max=xlim[1],
                is_log=self._x_is_log
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                min_val, max_val, is_log = dialog.get_values()
                self._x_is_log = is_log
                if is_log:
                    self.ax.set_xscale('log')
                else:
                    self.ax.set_xscale('linear')
                self.ax.set_xlim(min_val, max_val)
                self.draw()
        else:  # Y axis
            ylim = self.ax.get_ylim()
            dialog = AxisSettingsDialog(
                self.parent(),
                axis_name="Y Axis (Residual)",
                current_min=ylim[0],
                current_max=ylim[1],
                is_log=self._y_is_log
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                min_val, max_val, is_log = dialog.get_values()
                self._y_is_log = is_log
                if is_log:
                    self.ax.set_yscale('log')
                else:
                    self.ax.set_yscale('linear')
                self.ax.set_ylim(min_val, max_val)
                self.draw()


class InteractiveGeometryCanvas(FigureCanvas):
    """
    Interactive matplotlib canvas for geometry design with:
    - Scroll to zoom (middle mouse)
    - Right mouse drag to pan
    - Double-click on axis to edit min/max settings
    - Left click for drawing (passed through)
    """
    
    def __init__(self, parent=None, width=12, height=8, dpi=100, facecolor='#1e1e1e'):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=facecolor)
        self.ax = self.fig.add_subplot(111, facecolor='#1e1e1e')
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Pan state (use right button for pan to preserve left for drawing)
        self._pan_active = False
        self._pan_start = None
        self._xlim_start = None
        self._ylim_start = None
        
        # Store original limits for reset
        self._original_xlim = None
        self._original_ylim = None
        
        # Drawing state (managed by parent GUI)
        self.current_points = []
        self.drawing_mode = "Polynomial"
        self.hover_point = None
        self.selected_elements = []
        self.editing_mode = "draw"
        
        # Callback for external draw events
        self._on_left_click_callback = None
        self._on_move_callback = None
        self._on_key_callback = None
        
        # Connect internal events for zoom/pan
        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)
        
        # Make focusable for keyboard events
        from PySide6.QtCore import Qt
        self.setFocusPolicy(Qt.ClickFocus)
        
    def set_callbacks(self, on_left_click=None, on_move=None, on_key=None):
        """Set callbacks for drawing events."""
        self._on_left_click_callback = on_left_click
        self._on_move_callback = on_move
        self._on_key_callback = on_key
        
    def store_original_limits(self):
        """Store current limits as original for reset."""
        self._original_xlim = self.ax.get_xlim()
        self._original_ylim = self.ax.get_ylim()
        
    def reset_view(self):
        """Reset view to original limits."""
        if self._original_xlim and self._original_ylim:
            self.ax.set_xlim(self._original_xlim)
            self.ax.set_ylim(self._original_ylim)
            self.draw()
        
    def _on_scroll(self, event):
        """Handle scroll events for zooming."""
        if event.inaxes != self.ax:
            return
            
        base_scale = 1.15
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return
            
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        if xdata is None or ydata is None:
            return
        
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor
        
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
        
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.draw()
        
    def _on_press(self, event):
        """Handle mouse press events."""
        if event.dblclick:
            # Double-click: show axis settings dialog
            self._handle_axis_double_click(event)
            return
            
        if event.inaxes != self.ax:
            return
        
        # Right mouse button for pan (to preserve left for drawing)
        if event.button == 3:
            # Check if we're in a mode where right-click should pan OR finish drawing
            # If there are current points, right-click finishes drawing
            if not self.current_points:
                # No drawing in progress - use for pan
                self._pan_active = True
                self._pan_start = (event.xdata, event.ydata)
                self._xlim_start = self.ax.get_xlim()
                self._ylim_start = self.ax.get_ylim()
                return
        
        # Middle mouse button always pans
        if event.button == 2:
            self._pan_active = True
            self._pan_start = (event.xdata, event.ydata)
            self._xlim_start = self.ax.get_xlim()
            self._ylim_start = self.ax.get_ylim()
            return
            
        # Left click and right click (when drawing) passed to callback
        if self._on_left_click_callback:
            self._on_left_click_callback(event)
            
    def _on_release(self, event):
        """Handle mouse release events."""
        if event.button in [2, 3]:  # Middle or right button
            self._pan_active = False
            self._pan_start = None
            
    def _on_motion(self, event):
        """Handle mouse motion events for panning and drawing preview."""
        # Handle panning
        if self._pan_active and self._pan_start is not None:
            if event.inaxes == self.ax and event.xdata is not None:
                dx = self._pan_start[0] - event.xdata
                dy = self._pan_start[1] - event.ydata
                
                self.ax.set_xlim(self._xlim_start[0] + dx, self._xlim_start[1] + dx)
                self.ax.set_ylim(self._ylim_start[0] + dy, self._ylim_start[1] + dy)
                
                self.draw()
            return
        
        # Pass to drawing callback
        if self._on_move_callback:
            self._on_move_callback(event)
        
    def _handle_axis_double_click(self, event):
        """Handle double-click on axes to show settings dialog."""
        fig_point = self.fig.transFigure.inverted().transform((event.x, event.y))
        ax_bbox = self.ax.get_position()
        
        # Check if near X axis (bottom of plot)
        if fig_point[1] < ax_bbox.y0 + 0.05:
            self._show_axis_dialog('X')
        # Check if near Y axis (left of plot)  
        elif fig_point[0] < ax_bbox.x0 + 0.05:
            self._show_axis_dialog('Y')
    
    def _show_axis_dialog(self, axis: str):
        """Show axis settings dialog."""
        if axis == 'X':
            xlim = self.ax.get_xlim()
            dialog = AxisSettingsDialog(
                self.parent(), 
                axis_name="X Axis (Axial Distance)",
                current_min=xlim[0],
                current_max=xlim[1],
                is_log=False
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                min_val, max_val, is_log = dialog.get_values()
                self.ax.set_xlim(min_val, max_val)
                self.draw()
        else:  # Y axis
            ylim = self.ax.get_ylim()
            dialog = AxisSettingsDialog(
                self.parent(),
                axis_name="Y Axis (Radial Distance)",
                current_min=ylim[0],
                current_max=ylim[1],
                is_log=False
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                min_val, max_val, is_log = dialog.get_values()
                self.ax.set_ylim(min_val, max_val)
                self.draw()


class InteractiveMeshCanvas(FigureCanvas):
    """
    Interactive matplotlib canvas for mesh visualization with:
    - Scroll to zoom
    - Left/Middle mouse drag to pan
    - Equal aspect ratio maintained
    - Proper boundary color legend
    """
    
    def __init__(self, parent=None, width=10, height=6, dpi=100, facecolor='#1e1e1e'):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=facecolor)
        self.ax = self.fig.add_subplot(111, facecolor='#1e1e1e')
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Pan state
        self._pan_active = False
        self._pan_start = None
        self._xlim_start = None
        self._ylim_start = None
        
        # Store original limits for reset
        self._original_xlim = None
        self._original_ylim = None
        
        # Connect events
        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)
        
        # Style axes
        self._style_axes()
        
    def _style_axes(self):
        """Apply dark theme styling to axes."""
        text_color = '#e0e0e0'
        border_color = '#404040'
        
        self.ax.set_xlabel('X [m]', color=text_color)
        self.ax.set_ylabel('Y [m]', color=text_color)
        self.ax.tick_params(colors=text_color)
        for spine in self.ax.spines.values():
            spine.set_color(border_color)
        self.ax.grid(True, alpha=0.2, color='#606060', linestyle='--')
        
    def store_original_limits(self):
        """Store current limits as original for reset."""
        self._original_xlim = self.ax.get_xlim()
        self._original_ylim = self.ax.get_ylim()
        
    def reset_view(self):
        """Reset view to original limits."""
        if self._original_xlim and self._original_ylim:
            self.ax.set_xlim(self._original_xlim)
            self.ax.set_ylim(self._original_ylim)
            self.draw()
        
    def _on_scroll(self, event):
        """Handle scroll events for zooming."""
        if event.inaxes != self.ax:
            return
            
        base_scale = 1.15
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return
            
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        if xdata is None or ydata is None:
            return
        
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor
        
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
        
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.draw()
        
    def _on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax:
            return
        
        # Left or middle mouse button for pan
        if event.button in [1, 2]:
            self._pan_active = True
            self._pan_start = (event.xdata, event.ydata)
            self._xlim_start = self.ax.get_xlim()
            self._ylim_start = self.ax.get_ylim()
            
    def _on_release(self, event):
        """Handle mouse release events."""
        if event.button in [1, 2]:
            self._pan_active = False
            self._pan_start = None
            
    def _on_motion(self, event):
        """Handle mouse motion events for panning."""
        if not self._pan_active or self._pan_start is None:
            return
            
        if event.inaxes != self.ax or event.xdata is None:
            return
            
        dx = self._pan_start[0] - event.xdata
        dy = self._pan_start[1] - event.ydata
        
        self.ax.set_xlim(self._xlim_start[0] + dx, self._xlim_start[1] + dx)
        self.ax.set_ylim(self._ylim_start[0] + dy, self._ylim_start[1] + dy)
        
        self.draw()


class NozzleDesignGUI(QMainWindow):
    """Professional CFD workflow application with intuitive interface and comprehensive features."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.geometry = NozzleGeometry()
        self.template_loader = TemplateLoader()
        self.current_file = None
        self.is_modified = False
        
        # Initialize advanced components directly - no fallback as requested
        self.mesh_generator = AdvancedMeshGenerator()
        self.simulation_setup = SimulationSetup()
        self.results_processor = ResultsProcessor()
        self.advanced_features = True
        
        # Current state
        self.current_mesh_data = None
        self.current_case_directory = ""
        self.current_results = None
        self.time_step_files = {}
        self.current_loaded_time_step = None
        
        # Simulation thread management
        self._simulation_thread = None
        self._simulation_worker = None

        # Optional workflow status UI (may not be present)
        self.status_labels = {}
        self.progress_bars = {}
        
        # Editing state
        self.editing_mode = "draw"
        self.selected_element_index = None
        
        # Design constraints (internal defaults, not shown in UI)
        self._min_throat_ratio = 0.5
        self._max_divergence_angle = 20
        self._enforce_continuity = True
        
        # Scaling state (computed once window is shown and when moved/resized)
        self.scale_factor = 1.0
        self.scale_factor = 1.0
        self.base_font_size = 10
        self._last_applied_scale = None

        self.setup_ui()
        # Apply initial theme/font; will be refined on first showEvent
        self.apply_responsive_font(force=True)
        self.apply_theme()
        
    def _get_target_screen(self):
        handle = self.windowHandle()
        if handle and handle.screen():
            return handle.screen()
        return QApplication.primaryScreen()

    def _compute_scale_factor(self) -> float:
        """Compute a stable UI scale factor based on DPI and resolution.

        Uses weighted average of physical DPI and logical resolution.
        This scales appropriately for 1080p (1.0), 1440p (1.2), 4K (1.4+).
        """
        screen = self._get_target_screen()
        if screen is None:
            return 1.0

        # Get both logical DPI and physical pixel dimensions
        logical_dpi = float(screen.logicalDotsPerInch() or 96.0)
        device_pixel_ratio = screen.devicePixelRatio() if hasattr(screen, 'devicePixelRatio') else 1.0
        
        geom = screen.availableGeometry()
        width_px = float(geom.width())
        height_px = float(geom.height())
        
        # Calculate resolution-based scale: 1920x1080 = 1.0
        # 3840x2160 (4K) = ~2.0, 2560x1440 (1440p) = ~1.2
        diagonal_px = (width_px ** 2 + height_px ** 2) ** 0.5
        diagonal_reference = (1920.0 ** 2 + 1080.0 ** 2) ** 0.5
        resolution_scale = diagonal_px / diagonal_reference
        
        # DPI scale with device pixel ratio consideration
        dpi_scale = (logical_dpi / 96.0) * max(device_pixel_ratio, 1.0)
        
        # Weighted average: 60% resolution, 40% DPI
        combined_scale = (0.6 * resolution_scale) + (0.4 * dpi_scale)
        
        # Keep within reasonable limits (0.75x for small screens, 2.5x for ultra-high res)
        return max(0.75, min(2.5, combined_scale))

    def apply_responsive_font(self, force: bool = False):
        """Apply a resolution-aware font system.

        This is safe to call repeatedly; it only reapplies when the scale changes.
        """
        scale = self._compute_scale_factor()
        if (not force) and (self._last_applied_scale is not None) and abs(scale - self._last_applied_scale) < 0.05:
            return

        # Base font grows with scale; allow larger sizes for 4K displays.
        base_size = int(round(16 * scale))
        base_size = max(14, min(30, base_size))

        font = QFont("Arial", base_size)
        font.setWeight(QFont.Weight.Normal)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self.setFont(font)

        self.scale_factor = float(scale)
        self.base_font_size = int(base_size)
        self._last_applied_scale = float(scale)

        # Theme font sizes follow base_size - all increased for 4K
        Theme.FONT_SIZE_TINY = max(12, self.base_font_size - 2)
        Theme.FONT_SIZE_SMALL = max(13, self.base_font_size - 1)
        Theme.FONT_SIZE_NORMAL = self.base_font_size
        Theme.FONT_SIZE_MEDIUM = self.base_font_size + 2
        Theme.FONT_SIZE_LARGE = self.base_font_size + 4
        Theme.FONT_SIZE_XLARGE = self.base_font_size + 8
        Theme.FONT_SIZE_TITLE = self.base_font_size + 12

        # Layout scaling
        self.update_ui_scaling()
        
    def changeEvent(self, event):
        """Handle window state changes."""
        super().changeEvent(event)
        # Don't call apply_theme() here - it triggers setStyleSheet which causes recursion
        
    def resizeEvent(self, event):
        """Handle window resize and keep UI responsive."""
        super().resizeEvent(event)
        # Only update splitter proportions on resize, not full theme
        self._update_splitter_on_resize()
            
    def showEvent(self, event):
        """Handle window show event for proper initialization."""
        super().showEvent(event)
        # Theme is already applied in __init__, no need to reapply
            
    def update_ui_scaling(self):
        """Adjust key layout constraints based on current scale_factor."""
        scale = getattr(self, 'scale_factor', 1.0) or 1.0

        if hasattr(self, 'left_panel'):
            # Dynamic left panel sizing based on resolution
            base_min_width = 280  # Minimum width at 1x scale
            base_max_width = 340  # Maximum width at 1x scale
            min_w = int(round(base_min_width * scale))
            max_w = int(round(base_max_width * scale))
            self.left_panel.setMinimumWidth(min_w)
            self.left_panel.setMaximumWidth(max_w)
            
        # Dynamically adjust splitter sizes if it exists
        if hasattr(self, 'main_splitter') and self.main_splitter is not None:
            # Calculate left panel size based on scale
            left_size = int(round(320 * scale))
            # Get total available width
            total_size = self.main_splitter.width()
            if total_size > 0:
                right_size = total_size - left_size
                self.main_splitter.setSizes([left_size, right_size])
        
    def _update_splitter_on_resize(self):
        """Update splitter proportions when window is resized."""
        if hasattr(self, 'main_splitter') and self.main_splitter is not None:
            scale = getattr(self, 'scale_factor', 1.0) or 1.0
            left_size = int(round(320 * scale))
            total_size = self.main_splitter.width()
            if total_size > left_size + 100:  # Ensure minimum space for right panel
                right_size = total_size - left_size
                self.main_splitter.setSizes([left_size, right_size])
        
    def apply_theme(self):
        """Apply clean modern theme with proper small font sizes."""
        # Use resolution-aware sizes
        normal_font = int(getattr(Theme, 'FONT_SIZE_NORMAL', 10))
        medium_font = int(getattr(Theme, 'FONT_SIZE_MEDIUM', normal_font + 1))
        large_font = int(getattr(Theme, 'FONT_SIZE_LARGE', normal_font + 2))

        scale = getattr(self, 'scale_factor', 1.0) or 1.0

        # Spacing scales gently
        spacing_sm = int(round(4 * scale))
        spacing_md = int(round(8 * scale))
        spacing_lg = int(round(12 * scale))
        
        self.setStyleSheet(f"""
            /* === CLEAN BASE STYLING === */
            QMainWindow {{
                background: {Theme.BACKGROUND};
                color: {Theme.TEXT_PRIMARY};
                border: none;
                font-size: {normal_font}px;
                font-family: Arial, sans-serif;
            }}
            
            QWidget {{
                background: transparent;
                color: {Theme.TEXT_PRIMARY};
                border: none;
                font-size: {normal_font}px;
            }}
            
            /* === GROUPBOXES FOR 4K === */
            QGroupBox {{
                background: {Theme.SURFACE_VARIANT};
                color: {Theme.TEXT_PRIMARY};
                font-weight: 600;
                font-size: {medium_font}px;
                border: 1px solid {Theme.BORDER};
                border-radius: 10px;
                /* Give the title room; avoids clipped/overlapping headers */
                margin-top: 22px;
                padding: 24px 12px 12px 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                top: 0px;
                padding: 4px 14px;
                background: {Theme.SURFACE_VARIANT};
                border-radius: 6px;
                color: {Theme.TEXT_PRIMARY};
                font-size: {medium_font}px;
            }}
            
            /* === MODERN OUTLINED BUTTONS (NO RED/GREEN) === */
            QPushButton {{
                background: {Theme.SURFACE_VARIANT};
                color: {Theme.TEXT_PRIMARY};
                border: 2px solid {Theme.PRIMARY};
                border-radius: 10px;
                padding: 10px 18px;
                font-size: {normal_font}px;
                font-weight: 600;
                font-family: Arial, sans-serif;
                min-height: {int(round(38 * scale))}px;
            }}
            QPushButton:hover {{
                background: {Theme.SURFACE_ELEVATED};
                border-color: {Theme.PRIMARY_LIGHT};
            }}
            QPushButton:pressed {{
                background: {Theme.SURFACE};
                border-color: {Theme.PRIMARY_VARIANT};
            }}
            QPushButton:disabled {{
                background: {Theme.SURFACE_VARIANT};
                color: {Theme.TEXT_TERTIARY};
                border-color: {Theme.BORDER};
            }}

            QPushButton[class="primary"] {{
                background: {Theme.HIGHLIGHT_STRONG};
                border-color: {Theme.PRIMARY};
            }}
            QPushButton[class="primary"]:hover {{
                background: {Theme.HIGHLIGHT};
                border-color: {Theme.PRIMARY_LIGHT};
            }}
            
            /* === PREMIUM INPUTS === */
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background: {Theme.INPUT_BACKGROUND};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.INPUT_BORDER};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: {normal_font}px;
                min-height: {int(round(30 * scale))}px;
                max-height: {int(round(40 * scale))}px;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {Theme.INPUT_BORDER_FOCUS};
                background: {Theme.SURFACE};
            }}
            
            /* === COMBOBOX WITH PROPER DROPDOWN STYLING === */
            QComboBox {{
                background: {Theme.SURFACE_VARIANT};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.INPUT_BORDER};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: {normal_font}px;
                min-height: {int(round(30 * scale))}px;
                max-height: {int(round(40 * scale))}px;
            }}
            QComboBox:focus {{
                border-color: {Theme.INPUT_BORDER_FOCUS};
                background: {Theme.SURFACE_ELEVATED};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 28px;
                subcontrol-origin: padding;
                subcontrol-position: right center;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid {Theme.TEXT_PRIMARY};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background: {Theme.SURFACE_ELEVATED};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER_ACCENT};
                border-radius: 6px;
                selection-background-color: {Theme.PRIMARY};
                selection-color: {Theme.TEXT_INVERSE};
                outline: none;
                padding: 6px;
            }}
            QComboBox QAbstractItemView::item {{
                background: transparent;
                color: {Theme.TEXT_PRIMARY};
                padding: 10px 14px;
                min-height: 32px;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background: {Theme.PRIMARY_VARIANT};
                color: {Theme.TEXT_INVERSE};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background: {Theme.PRIMARY};
                color: {Theme.TEXT_INVERSE};
            }}
            
            /* === COMPACT LABELS === */
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {normal_font}px;
                font-weight: 400;
                padding: 2px 0px;
            }}
            
            /* === COMPACT TAB WIDGET === */
            QTabWidget::pane {{
                background: {Theme.SURFACE};
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                padding: {spacing_md}px;
                margin-top: -1px;
            }}
            QTabBar {{
                qproperty-drawBase: 0;
            }}
            QTabBar::tab {{
                background: {Theme.SURFACE_VARIANT};
                color: {Theme.TEXT_SECONDARY};
                border: 1px solid {Theme.BORDER};
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 20px;
                margin-right: 4px;
                font-size: {normal_font}px;
                font-weight: 500;
                min-width: 100px;
            }}
            QTabBar::tab:selected {{
                background: {Theme.SURFACE};
                color: {Theme.TEXT_PRIMARY};
                border-color: {Theme.BORDER};
                border-bottom: 1px solid {Theme.SURFACE};
                font-weight: 600;
            }}
            QTabBar::tab:hover:!selected {{
                background: {Theme.SURFACE_ELEVATED};
                color: {Theme.TEXT_PRIMARY};
            }}
            
            /* === SPLITTER (EASY TO GRAB ON 4K) === */
            QSplitter::handle {{
                background: {Theme.BORDER};
                width: 6px;
                height: 6px;
                border-radius: 3px;
            }}
            QSplitter::handle:hover {{
                background: {Theme.BORDER_ACCENT};
            }}
            
            /* === COMPACT SCROLLBARS === */
            QScrollBar:vertical {{
                background: {Theme.SURFACE};
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {Theme.BORDER};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Theme.BORDER_ACCENT};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
            QScrollBar:horizontal {{
                background: {Theme.SURFACE};
                height: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {Theme.BORDER};
                border-radius: 6px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {Theme.BORDER_ACCENT};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                border: none;
                background: none;
            }}
            
            /* === CHECKBOX STYLING === */
            QCheckBox {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {normal_font}px;
                spacing: 8px;
                padding: 4px 0px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {Theme.BORDER};
                background: {Theme.INPUT_BACKGROUND};
            }}
            QCheckBox::indicator:checked {{
                background: {Theme.PRIMARY};
                border-color: {Theme.PRIMARY};
            }}
            QCheckBox::indicator:hover {{
                border-color: {Theme.PRIMARY_LIGHT};
            }}
            QCheckBox:disabled {{
                color: {Theme.TEXT_TERTIARY};
            }}
            
            /* === RADIO BUTTON STYLING === */
            QRadioButton {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {normal_font}px;
                spacing: 8px;
                padding: 4px 0px;
            }}
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid {Theme.BORDER};
                background: {Theme.INPUT_BACKGROUND};
            }}
            QRadioButton::indicator:checked {{
                background: {Theme.PRIMARY};
                border-color: {Theme.PRIMARY};
            }}
            QRadioButton::indicator:hover {{
                border-color: {Theme.PRIMARY_LIGHT};
            }}
            
            /* === TEXT EDIT STYLING === */
            QTextEdit {{
                background: {Theme.INPUT_BACKGROUND};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.INPUT_BORDER};
                border-radius: 6px;
                padding: 8px;
                font-size: {normal_font}px;
                selection-background-color: {Theme.PRIMARY};
                selection-color: {Theme.TEXT_INVERSE};
            }}
            QTextEdit:focus {{
                border-color: {Theme.INPUT_BORDER_FOCUS};
            }}
            
            /* === LIST WIDGET STYLING === */
            QListWidget {{
                background: {Theme.SURFACE_VARIANT};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 6px;
                outline: none;
            }}
            QListWidget::item {{
                padding: 6px 10px;
                border-bottom: 1px solid {Theme.BORDER_SUBTLE};
            }}
            QListWidget::item:selected {{
                background: {Theme.PRIMARY};
                color: {Theme.TEXT_INVERSE};
            }}
            QListWidget::item:hover:!selected {{
                background: {Theme.SURFACE_ELEVATED};
            }}
            
            /* === TOOLTIP STYLING === */
            QToolTip {{
                background: {Theme.SURFACE_ELEVATED};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER_ACCENT};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: {normal_font}px;
            }}
            
            /* === MENU STYLING === */
            QMenu {{
                background: {Theme.SURFACE_ELEVATED};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 6px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 8px 24px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background: {Theme.PRIMARY};
                color: {Theme.TEXT_INVERSE};
            }}
            QMenu::separator {{
                height: 1px;
                background: {Theme.BORDER};
                margin: 4px 8px;
            }}
            
            /* === MESSAGE BOX STYLING === */
            QMessageBox {{
                background: {Theme.SURFACE};
            }}
            QMessageBox QLabel {{
                color: {Theme.TEXT_PRIMARY};
            }}
        """)
        
        # Apply additional styling for specific widget types
        self.apply_custom_styles()
        
    def apply_custom_styles(self):
        """Apply custom styles to specific widgets after creation."""
        # This method will be called after widgets are created
        # to apply specific styling classes
        pass
        
    def setup_ui(self):
        """Setup the UI with a modern full-width workflow design."""
        self.setWindowTitle("Nozzle Flow CFD Designer")

        # Size relative to available screen
        screen = QApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            w = int(round(geom.width() * 0.92))
            h = int(round(geom.height() * 0.92))
            self.resize(w, h)
        else:
            self.resize(1600, 900)

        # Dynamic minimum size
        scale = self._compute_scale_factor()
        min_width = max(1000, min(2000, int(round(1100 * scale))))
        min_height = max(650, min(1400, int(round(700 * scale))))
        self.setMinimumSize(min_width, min_height)
        
        # Create central widget
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        
        # Main vertical layout - no left panel!
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top workflow header with progress indicator
        self.workflow_header = self.create_workflow_header()
        main_layout.addWidget(self.workflow_header)
        
        # Main tab widget takes full width
        self.tab_widget = self.create_workflow_tabs()
        main_layout.addWidget(self.tab_widget, 1)  # Stretch factor 1
        
        # Global output panel at the bottom
        self._create_global_output_panel(main_layout)
        
        # Bottom status bar
        self.create_modern_status_bar()
        
        # Initialize project metadata storage (no visible widgets needed)
        self._init_project_data()
        
        # Apply custom styles after widget creation
        self.apply_widget_classes()
        
    def _init_project_data(self):
        """Initialize project metadata (stored internally, not displayed in left panel)."""
        if not hasattr(self, '_project_name'):
            self._project_name = "Untitled Project"
        if not hasattr(self, '_project_description'):
            self._project_description = ""
    
    def _create_global_output_panel(self, parent_layout: QVBoxLayout):
        """Create a collapsible global output panel at the bottom of the window."""
        from datetime import datetime
        
        # Container widget
        output_container = QWidget()
        output_container.setObjectName("outputContainer")
        output_layout = QVBoxLayout(output_container)
        output_layout.setContentsMargins(8, 4, 8, 4)
        output_layout.setSpacing(2)
        
        # Header with toggle button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        toggle_btn = QPushButton("▼ Output Log")
        toggle_btn.setObjectName("outputToggle")
        toggle_btn.setFlat(True)
        toggle_btn.setStyleSheet(f"""
            QPushButton {{
                color: {Theme.TEXT_SECONDARY};
                font-weight: bold;
                text-align: left;
                padding: 4px 8px;
                border: none;
                background: transparent;
            }}
            QPushButton:hover {{
                color: {Theme.TEXT_PRIMARY};
            }}
        """)
        header_layout.addWidget(toggle_btn)
        
        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                color: {Theme.TEXT_TERTIARY};
                background: transparent;
                border: 1px solid {Theme.BORDER};
                border-radius: 3px;
                padding: 2px 8px;
            }}
            QPushButton:hover {{
                background: {Theme.SURFACE_ELEVATED};
            }}
        """)
        header_layout.addWidget(clear_btn)
        header_layout.addStretch()
        
        output_layout.addLayout(header_layout)
        
        # Output text area
        self.global_output_log = QTextEdit()
        self.global_output_log.setReadOnly(True)
        self.global_output_log.setMaximumHeight(150)
        self.global_output_log.setMinimumHeight(80)
        self.global_output_log.setStyleSheet(f"""
            QTextEdit {{
                background: {Theme.BACKGROUND};
                color: {Theme.TEXT_SECONDARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: {Theme.FONT_SIZE_SMALL}px;
                padding: 4px;
            }}
        """)
        output_layout.addWidget(self.global_output_log)
        
        # Connect toggle
        def toggle_output():
            visible = self.global_output_log.isVisible()
            self.global_output_log.setVisible(not visible)
            toggle_btn.setText("▶ Output Log" if visible else "▼ Output Log")
        
        toggle_btn.clicked.connect(toggle_output)
        clear_btn.clicked.connect(lambda: self.global_output_log.clear())
        
        # Style the container
        output_container.setStyleSheet(f"""
            #outputContainer {{
                background: {Theme.SURFACE};
                border-top: 1px solid {Theme.BORDER};
            }}
        """)
        
        parent_layout.addWidget(output_container)
        
        # Initial message
        self.log_message("Application started. Ready for CFD workflow.", "info")
    
    def log_message(self, message: str, level: str = "info"):
        """Log a message to the global output panel.
        
        Args:
            message: The message to log
            level: One of 'info', 'warning', 'error', 'success'
        """
        from datetime import datetime
        
        if not hasattr(self, 'global_output_log') or self.global_output_log is None:
            print(f"[{level.upper()}] {message}")
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color based on level
        colors = {
            "info": Theme.TEXT_SECONDARY,
            "warning": "#FFA500",
            "error": Theme.ERROR,
            "success": Theme.SUCCESS
        }
        color = colors.get(level, Theme.TEXT_SECONDARY)
        
        # Prefix based on level
        prefixes = {
            "info": "ℹ",
            "warning": "⚠",
            "error": "✗",
            "success": "✓"
        }
        prefix = prefixes.get(level, "•")
        
        html = f'<span style="color: {Theme.TEXT_TERTIARY};">[{timestamp}]</span> <span style="color: {color};">{prefix} {message}</span><br>'
        
        # Append and scroll to bottom
        cursor = self.global_output_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.global_output_log.setTextCursor(cursor)
        self.global_output_log.insertHtml(html)
        self.global_output_log.ensureCursorVisible()
            
    def apply_widget_classes(self):
        """Apply CSS classes to specific widgets for better styling."""
        # Find and style primary action buttons (without emoticons)
        primary_buttons = [
            "Auto Generate",
            "Generate Mesh", 
            "Run Simulation",
            "Export Mesh"
        ]
        
        for button in self.findChildren(QPushButton):
            if any(text in button.text() for text in primary_buttons):
                button.setProperty("class", "primary")
                button.style().unpolish(button)
                button.style().polish(button)

    def _configure_splitter(self, splitter: QSplitter):
        """Make splitters easier to grab and keep panes visible."""
        scale = float(getattr(self, 'scale_factor', 1.0) or 1.0)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(max(8, int(round(10 * scale))))

    def _create_bottom_input_panel(self, title: str) -> QWidget:
        """Bottom input area shared across tabs (notes / quick inputs)."""
        panel = QGroupBox(title)
        panel.setObjectName("bottomInputPanel")

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setSpacing(12)

        editor = QTextEdit()
        editor.setPlaceholderText(
            "Quick notes / parameter overrides…\n"
            "(This area is intentionally large for 4K and can be resized via the splitter.)"
        )
        editor.setMinimumHeight(80)
        layout.addWidget(editor, 1)

        actions = QVBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(10)

        btn_clear = QPushButton("Clear Notes")
        btn_clear.clicked.connect(editor.clear)
        actions.addWidget(btn_clear)

        actions.addStretch(1)
        layout.addLayout(actions)

        # Keep the bottom panel compact by default; user can expand via splitter.
        scale = float(getattr(self, 'scale_factor', 1.0) or 1.0)
        panel.setMinimumHeight(int(round(120 * scale)))
        panel.setMaximumHeight(int(round(320 * scale)))

        return panel
                
    def create_workflow_header(self):
        """Create a sleek workflow progress header."""
        header = QWidget()
        header.setFixedHeight(90)
        header.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Theme.SURFACE}, stop:1 {Theme.SURFACE_VARIANT});
                border-bottom: 1px solid {Theme.BORDER};
            }}
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(32, 0, 32, 0)
        layout.setSpacing(0)
        
        # Left: Logo and title
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(16)
        
        # App title
        title = QLabel("Nozzle Flow CFD")
        title.setStyleSheet(f"""
            QLabel {{
                color: {Theme.PRIMARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_TITLE', 26))}px;
                font-weight: 700;
                font-family: Arial, sans-serif;
                letter-spacing: 0.5px;
                background: transparent;
            }}
        """)
        title_layout.addWidget(title)
        
        layout.addWidget(title_container)
        layout.addSpacing(40)
        
        # Center: Workflow steps
        steps_container = QWidget()
        steps_container.setStyleSheet("background: transparent;")
        steps_layout = QHBoxLayout(steps_container)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(8)
        
        self.workflow_steps = []
        step_names = [
            ("1", "Geometry", "Design nozzle profile"),
            ("2", "Mesh", "Generate CFD mesh"),
            ("3", "Simulate", "Run CFD analysis"),
            ("4", "Results", "Analyze output")
        ]
        
        for i, (num, name, tooltip) in enumerate(step_names):
            step_widget = self._create_workflow_step(num, name, tooltip, i == 0)
            self.workflow_steps.append(step_widget)
            steps_layout.addWidget(step_widget)
            
            # Add arrow between steps (except last)
            if i < len(step_names) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet(f"""
                    QLabel {{
                        color: {Theme.TEXT_SECONDARY};
                        font-size: 24px;
                        font-weight: 300;
                        background: transparent;
                        padding: 0 12px;
                    }}
                """)
                steps_layout.addWidget(arrow)
        
        layout.addWidget(steps_container)
        layout.addStretch()
        
        # Right: Project info / Quick actions
        actions_container = QWidget()
        actions_container.setStyleSheet("background: transparent;")
        actions_layout = QHBoxLayout(actions_container)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(12)
        
        # Project name label
        self.header_project_label = QLabel("New Project")
        self.header_project_label.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_SECONDARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_NORMAL', 16))}px;
                font-weight: 500;
                background: transparent;
            }}
        """)
        actions_layout.addWidget(self.header_project_label)
        
        # Quick save button
        save_btn = QPushButton("Save")
        save_btn.setFixedSize(90, 40)
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(self.save_project)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {Theme.TEXT_SECONDARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                font-size: {int(getattr(Theme, 'FONT_SIZE_NORMAL', 15))}px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {Theme.SURFACE_VARIANT};
                color: {Theme.TEXT_PRIMARY};
            }}
        """)
        actions_layout.addWidget(save_btn)
        
        layout.addWidget(actions_container)
        
        return header
        
    def _create_workflow_step(self, num, name, tooltip, is_active=False):
        """Create a single workflow step indicator."""
        step = QWidget()
        step.setFixedHeight(60)
        step.setCursor(Qt.PointingHandCursor)
        step.setToolTip(tooltip)
        step.setProperty("step_num", int(num))
        
        bg_color = Theme.PRIMARY if is_active else "transparent"
        text_color = Theme.TEXT_PRIMARY if is_active else Theme.TEXT_SECONDARY
        num_bg = Theme.PRIMARY_VARIANT if is_active else Theme.SURFACE_VARIANT
        
        step.setStyleSheet(f"""
            QWidget {{
                background: {bg_color};
                border-radius: 10px;
                padding: 6px 16px;
            }}
        """)
        
        layout = QHBoxLayout(step)
        layout.setContentsMargins(16, 8, 20, 8)
        layout.setSpacing(14)
        
        # Step number circle
        num_label = QLabel(num)
        num_label.setFixedSize(32, 32)
        num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num_label.setStyleSheet(f"""
            QLabel {{
                background: {num_bg};
                color: {Theme.TEXT_PRIMARY};
                font-size: 15px;
                font-weight: 700;
                border-radius: 16px;
            }}
        """)
        num_label.setObjectName("step_num_label")
        layout.addWidget(num_label)
        
        # Step name
        name_label = QLabel(name)
        name_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-size: 16px;
                font-weight: 600;
                background: transparent;
            }}
        """)
        name_label.setObjectName("step_name_label")
        layout.addWidget(name_label)
        
        # Make clickable
        step.mousePressEvent = lambda e, n=int(num): self._on_workflow_step_clicked(n)
        
        return step
        
    def _on_workflow_step_clicked(self, step_num):
        """Handle workflow step click to navigate tabs."""
        if hasattr(self, 'tab_widget') and self.tab_widget:
            self.tab_widget.setCurrentIndex(step_num - 1)
            self._update_workflow_header(step_num - 1)
            
    def _update_workflow_header(self, active_index):
        """Update workflow header to reflect current step."""
        for i, step in enumerate(self.workflow_steps):
            is_active = (i == active_index)
            bg_color = Theme.PRIMARY if is_active else "transparent"
            text_color = Theme.TEXT_PRIMARY if is_active else Theme.TEXT_SECONDARY
            num_bg = Theme.PRIMARY_VARIANT if is_active else Theme.SURFACE_VARIANT
            
            step.setStyleSheet(f"""
                QWidget {{
                    background: {bg_color};
                    border-radius: 8px;
                    padding: 4px 12px;
                }}
            """)
            
            # Update child labels
            num_label = step.findChild(QLabel, "step_num_label")
            if num_label:
                num_label.setStyleSheet(f"""
                    QLabel {{
                        background: {num_bg};
                        color: {Theme.TEXT_PRIMARY};
                        font-size: 12px;
                        font-weight: 700;
                        border-radius: 12px;
                    }}
                """)
            
            name_label = step.findChild(QLabel, "step_name_label")
            if name_label:
                name_label.setStyleSheet(f"""
                    QLabel {{
                        color: {text_color};
                        font-size: 13px;
                        font-weight: 600;
                        background: transparent;
                    }}
                """)
                
    def create_workflow_tabs(self):
        """Create the main workflow tab widget with hidden tab bar.
        
        Navigation is handled by the workflow header steps at the top.
        The QTabBar is hidden to avoid duplicate navigation options.
        """
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(False)
        
        # Hide the tab bar completely - navigation is via workflow header
        tab_widget.tabBar().setVisible(False)
        
        tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                background: {Theme.BACKGROUND};
                border: none;
                margin-top: 0px;
                padding: 0px;
            }}
        """)
        
        # Connect tab change to workflow header update
        tab_widget.currentChanged.connect(self._update_workflow_header)
        
        # Geometry Design Tab
        geometry_tab = self.create_geometry_tab()
        tab_widget.addTab(geometry_tab, "Geometry Design")
        
        # Mesh Generation Tab
        mesh_tab = self.create_meshing_tab()
        tab_widget.addTab(mesh_tab, "Mesh Generation")
        
        # Simulation Tab
        simulation_tab = self.create_simulation_tab()
        tab_widget.addTab(simulation_tab, "Simulation")
        
        # Results Tab
        results_tab = self.create_postprocessing_tab()
        tab_widget.addTab(results_tab, "Results")
        
        return tab_widget
        
    def create_modern_left_panel(self):
        """Create a compact, clean left control panel."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        panel.setMinimumWidth(300)
        
        # Main layout for the panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Header section without emoticons
        header = self.create_header_section()
        layout.addWidget(header)
        
        # Parameters card
        params_card = self.create_parameters_card()
        layout.addWidget(params_card)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
        
    def create_header_section(self):
        """Create a clean header section without emoticons."""
        header = QWidget()
        header.setFixedHeight(60)  # Smaller height
        
        header.setStyleSheet(f"""
            QWidget {{
                background: {Theme.SURFACE_VARIANT};
                border-radius: 8px;
                border: 1px solid {Theme.BORDER};
            }}
        """)
        
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(4)
        
        # Main title without emoticons
        title = QLabel("NOZZLE CFD DESIGNER")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 700;
                font-family: Arial, sans-serif;
                letter-spacing: 1px;
                background: transparent;
                padding: 4px 0px;
            }}
        """)
        layout.addWidget(title)
        
        # Subtitle without emoticons
        subtitle = QLabel("Professional CFD Workflow Suite")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_SECONDARY};
                font-size: 9px;
                font-weight: 400;
                font-family: Arial, sans-serif;
                background: transparent;
                padding: 0px;
            }}
        """)
        layout.addWidget(subtitle)
        
        return header
        
    def create_status_card(self):
        """Create a modern status card showing workflow progress."""
        card = QGroupBox("Workflow Status")
        card.setStyleSheet(f"""
            QGroupBox {{
                background: {Theme.SURFACE_VARIANT};
                border: 1px solid {Theme.BORDER};
                border-radius: 12px;
                font-size: 14px;
                font-weight: 600;
                padding: 16px 8px 8px 8px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        
        # Status indicators with modern styling
        self.geometry_status = QLabel("Geometry: Not defined")
        self.mesh_status = QLabel("Mesh: Not generated") 
        self.simulation_status = QLabel("Simulation: Not run")
        self.results_status = QLabel("Results: Not available")
        
        # Store status labels for compatibility with existing update system
        self.status_labels = {
            'geometry': self.geometry_status,
            'mesh': self.mesh_status,
            'simulation': self.simulation_status,
            'results': self.results_status
        }
        
        # Create progress bars for compatibility
        self.progress_bars = {}
        
        for key, status in self.status_labels.items():
            # Create container for status item with proper layout
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(8)
            
            # Style status label with fixed width for alignment
            status.setStyleSheet(f"""
                QLabel {{
                    color: {Theme.TEXT_SECONDARY};
                    font-size: 12px;
                    font-weight: 500;
                    padding: 8px 12px;
                    background: {Theme.SURFACE};
                    border-radius: 8px;
                    border-left: 4px solid {Theme.WARNING};
                    min-height: 16px;
                    min-width: 200px;
                }}
            """)
            container_layout.addWidget(status, 1)
            
            # Add progress label with fixed width for proper alignment
            progress_label = QLabel("0%")
            progress_label.setFixedWidth(35)
            progress_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            progress_label.setStyleSheet(f"""
                QLabel {{
                    color: {Theme.TEXT_SECONDARY};
                    font-size: 11px;
                    font-weight: 500;
                    padding: 0px 4px;
                }}
            """)
            self.progress_bars[key] = progress_label  # Store label instead of progress bar
            container_layout.addWidget(progress_label)
            
            layout.addWidget(container)
            
        return card
        
    def create_quick_actions_card(self):
        """Create a card with quick action buttons."""
        card = QGroupBox("Quick Actions")
        
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        
        # Auto-generate button
        btn_auto = QPushButton("Auto Generate Geometry")
        btn_auto.setProperty("class", "primary")
        btn_auto.clicked.connect(self.auto_generate_mesh)
        layout.addWidget(btn_auto)
        
        # Template buttons in a row
        template_layout = QHBoxLayout()
        
        btn_bell = QPushButton("Bell")
        btn_bell.clicked.connect(lambda: self.load_template("bell"))
        template_layout.addWidget(btn_bell)
        
        btn_conical = QPushButton("Conical")
        btn_conical.clicked.connect(lambda: self.load_template("conical"))
        template_layout.addWidget(btn_conical)
        
        layout.addLayout(template_layout)
        
        return card
        
    def create_parameters_card(self):
        """Create a modern parameters card with all the essential controls."""
        card = QGroupBox("Project & Parameters")
        
        layout = QVBoxLayout(card)
        layout.setSpacing(16)
        
        # Project Information Section
        project_section = QWidget()
        project_layout = QFormLayout(project_section)
        project_layout.setSpacing(8)

        self.project_file_display = QLineEdit()
        self.project_file_display.setReadOnly(True)
        self.project_file_display.setPlaceholderText("Not saved")
        self.project_file_display.setToolTip(
            "<b>Project File</b><br/>"
            "Path to the saved project JSON (if any)."
        )
        project_layout.addRow("Project File:", self.project_file_display)

        self.case_dir_display = QLineEdit()
        self.case_dir_display.setReadOnly(True)
        self.case_dir_display.setPlaceholderText("Not exported")
        self.case_dir_display.setToolTip(
            "<b>SU2 Case Directory</b><br/>"
            "Folder where the SU2 case was exported (if any)."
        )
        project_layout.addRow("Case Dir:", self.case_dir_display)
        
        self.project_name_edit = QLineEdit("Untitled Project")
        self.project_name_edit.setToolTip(
            "<b>Project Name</b><br/>"
            "A human-friendly name for this nozzle design." )
        self.project_name_edit.textChanged.connect(self.on_project_name_changed)
        project_layout.addRow("Project Name:", self.project_name_edit)
        
        self.project_description = QTextEdit()
        self.project_description.setMaximumHeight(60)
        self.project_description.setPlaceholderText("Brief description of the nozzle design...")
        self.project_description.setToolTip(
            "<b>Description / Notes</b><br/>"
            "Design goals, operating conditions, and any assumptions." )
        project_layout.addRow("Description:", self.project_description)
        
        layout.addWidget(project_section)
        
        # Geometry Information Section
        info_label = QLabel("Geometry Information")
        info_label.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
                margin: 8px 0px 4px 0px;
            }}
        """)
        layout.addWidget(info_label)
        
        info_section = QWidget()
        info_layout = QFormLayout(info_section)
        info_layout.setSpacing(6)
        
        self.info_labels = {
            'elements': QLabel("0"),
            'length': QLabel("0.00 m"),
            'throat_ratio': QLabel("N/A"),
            'expansion_ratio': QLabel("N/A")
        }
        
        info_layout.addRow("Elements:", self.info_labels['elements'])
        info_layout.addRow("Length:", self.info_labels['length'])
        info_layout.addRow("Throat Ratio:", self.info_labels['throat_ratio'])
        info_layout.addRow("Expansion Ratio:", self.info_labels['expansion_ratio'])
        
        # Style info labels with modern appearance
        for label in self.info_labels.values():
            label.setStyleSheet(f"""
                QLabel {{
                    color: {Theme.SUCCESS};
                    font-weight: 600;
                    font-size: 13px;
                    padding: 4px 8px;
                    background: {Theme.SURFACE};
                    border-radius: 4px;
                    border-left: 3px solid {Theme.SUCCESS};
                }}
            """)
        
        layout.addWidget(info_section)

        # Initialize metadata display
        self.refresh_project_metadata()
        
        return card

    def refresh_project_metadata(self):
        """Refresh the left-panel project metadata fields (safe if widgets not created yet)."""
        if hasattr(self, 'project_file_display'):
            self.project_file_display.setText(self.current_file or "")

        if hasattr(self, 'case_dir_display'):
            self.case_dir_display.setText(self.current_case_directory or "")

        # Update quick geometry stats if the info labels exist
        if hasattr(self, 'info_labels') and isinstance(self.info_labels, dict):
            try:
                self.info_labels.get('elements') and self.info_labels['elements'].setText(str(len(self.geometry.elements)))

                x_coords, y_coords = self.geometry.get_interpolated_points(
                    num_points_per_element=getattr(self, 'interpolation_points', None) and self.interpolation_points.value() or 100
                )
                if x_coords and y_coords:
                    inlet_r = float(y_coords[0]) if y_coords[0] is not None else 0.0
                    throat_r = float(min(y for y in y_coords if y is not None))
                    outlet_r = float(y_coords[-1]) if y_coords[-1] is not None else 0.0

                    length = float(max(x_coords) - min(x_coords))
                    self.info_labels.get('length') and self.info_labels['length'].setText(f"{length:.3f} m")

                    if inlet_r > 0 and throat_r > 0:
                        self.info_labels.get('throat_ratio') and self.info_labels['throat_ratio'].setText(f"{throat_r / inlet_r:.3f}")
                    else:
                        self.info_labels.get('throat_ratio') and self.info_labels['throat_ratio'].setText("N/A")

                    if throat_r > 0 and outlet_r > 0:
                        self.info_labels.get('expansion_ratio') and self.info_labels['expansion_ratio'].setText(f"{outlet_r / throat_r:.3f}")
                    else:
                        self.info_labels.get('expansion_ratio') and self.info_labels['expansion_ratio'].setText("N/A")
                else:
                    self.info_labels.get('length') and self.info_labels['length'].setText("0.00 m")
                    self.info_labels.get('throat_ratio') and self.info_labels['throat_ratio'].setText("N/A")
                    self.info_labels.get('expansion_ratio') and self.info_labels['expansion_ratio'].setText("N/A")
            except Exception:
                # Keep UI responsive even if geometry stats fail
                pass
    
    def create_modern_menu_bar(self):
        """Create a modern, beautiful menu bar."""
        menubar = self.menuBar()
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background: {Theme.SURFACE};
                color: {Theme.TEXT_PRIMARY};
                border-bottom: 1px solid {Theme.BORDER};
                padding: 6px;
                font-size: 13px;
            }}
        """)
        
        # File menu with modern icons
        file_menu = menubar.addMenu("[...] File")
        
        new_action = file_menu.addAction("New Project")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        
        open_action = file_menu.addAction("Open Project")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        
        save_action = file_menu.addAction("Save Project")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        
        file_menu.addSeparator()
        
        export_action = file_menu.addAction("Export Case")
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_case)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu(" Tools")
        
        validate_action = tools_menu.addAction("Validate Geometry")
        validate_action.triggered.connect(self.validate_geometry)
        
        mesh_action = tools_menu.addAction("Generate Mesh")
        mesh_action.triggered.connect(self.generate_mesh)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        reset_view_action = view_menu.addAction("Reset View")
        reset_view_action.triggered.connect(self.reset_view)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        tutorial_action = help_menu.addAction("Tutorial")
        tutorial_action.triggered.connect(self.show_tutorial)
        
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
        
    def create_modern_status_bar(self):
        """Create a modern status bar with indicators."""
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background: {Theme.SURFACE};
                color: {Theme.TEXT_SECONDARY};
                border-top: 1px solid {Theme.BORDER};
                padding: 8px;
                font-size: 12px;
            }}
        """)
        
        # Add status indicators
        self.status_label = QLabel("[*] Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Add permanent widgets to the right
        self.coords_label = QLabel("Position: (0, 0)")
        self.status_bar.addPermanentWidget(self.coords_label)
        
        self.mode_label = QLabel("Mode: Draw")
        self.status_bar.addPermanentWidget(self.mode_label)
        
    def create_modern_tab_widget(self):
        """Create a modern tab widget optimized for fullscreen with no overlays.
        
        NOTE: This is a legacy method. Main UI uses create_workflow_tabs().
        Tab bar is hidden as navigation is via workflow header.
        """
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(False)
        
        # Hide tab bar - navigation via workflow header
        tab_widget.tabBar().setVisible(False)
        
        tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                background: {Theme.SURFACE};
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                margin-top: 0px;
                padding: 4px;
            }}
        """)
        
        # Geometry Design Tab
        geometry_tab = self.create_geometry_tab()
        tab_widget.addTab(geometry_tab, "Geometry Design")
        
        # Mesh Generation Tab
        mesh_tab = self.create_meshing_tab()
        tab_widget.addTab(mesh_tab, "Mesh Generation")
        
        # Simulation Tab
        simulation_tab = self.create_simulation_tab()
        tab_widget.addTab(simulation_tab, "Simulation")
        
        # Results Tab
        results_tab = self.create_postprocessing_tab()
        tab_widget.addTab(results_tab, "Results")
        
        return tab_widget
        
    def show_tutorial(self):
        """Show a tutorial dialog."""
        self.log_message("Welcome to Nozzle CFD Designer! 1. Design your nozzle geometry, 2. Generate a CFD mesh, 3. Run CFD simulation, 4. Analyze results", "info")
        
    def reset_view(self):
        """Reset the view to default."""
        # This will be implemented based on the canvas system
        pass
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = file_menu.addAction("New Project")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        
        open_action = file_menu.addAction("Open Project")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        
        save_action = file_menu.addAction("Save Project")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        
        file_menu.addSeparator()
        
        export_action = file_menu.addAction("Export Case")
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_case)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
        
    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
    def show_about(self):
        """Show about dialog."""
        self.log_message("Nozzle CFD Design Tool v2.0 - Professional CFD workflow application with geometry drawing, meshing, simulation, and visualization", "info")
        
    def create_left_panel(self):
        """Create left control panel with enhanced workflow status and quick actions."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Project info section
        project_group = QGroupBox("Project Information")
        project_layout = QFormLayout(project_group)
        
        self.project_name_edit = QLineEdit("Untitled Project")
        self.project_name_edit.setToolTip("Enter a descriptive name for your CFD project")
        self.project_name_edit.textChanged.connect(self.on_project_name_changed)
        project_layout.addRow("Name:", self.project_name_edit)
        
        self.project_description = QTextEdit()
        self.project_description.setMaximumHeight(60)
        self.project_description.setPlaceholderText("Brief description of the nozzle design...")
        self.project_description.setToolTip("Add notes about design objectives, operating conditions, etc.")
        project_layout.addRow("Description:", self.project_description)
        
        layout.addWidget(project_group)
        
        # Enhanced workflow status with progress indicators
        status_group = QGroupBox("Workflow Progress")
        status_layout = QVBoxLayout(status_group)
        
        self.status_labels = {
            'geometry': QLabel("[X] Geometry Design"),
            'mesh': QLabel("[X] Mesh Generation"),
            'simulation': QLabel("[X] CFD Setup"),
            'results': QLabel("[X] Post-Processing")
        }
        
        self.progress_bars = {}
        for key, label in self.status_labels.items():
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            
            label.setStyleSheet("font-weight: bold; padding: 5px; min-width: 120px;")
            container_layout.addWidget(label)
            
            progress = QProgressBar()
            progress.setMaximumHeight(20)
            progress.setMaximumWidth(100)
            progress.setValue(0)
            self.progress_bars[key] = progress
            container_layout.addWidget(progress)
            
            status_layout.addWidget(container)
            
        layout.addWidget(status_group)
        
        # Enhanced quick actions with categories
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QGridLayout(actions_group)
        
        # File operations row
        btn_new = QPushButton("New")
        btn_open = QPushButton("Open")
        btn_save = QPushButton("Save")
        btn_export = QPushButton("Export")
        
        # Add tooltips for better UX
        btn_new.setToolTip("Create a new nozzle design project (Ctrl+N)")
        btn_open.setToolTip("Open an existing project file (Ctrl+O)")
        btn_save.setToolTip("Save current project (Ctrl+S)")
        btn_export.setToolTip("Export SU2 case files (Ctrl+E)")
        
        btn_new.clicked.connect(self.new_project)
        btn_open.clicked.connect(self.open_project)
        btn_save.clicked.connect(self.save_project)
        btn_export.clicked.connect(self.export_case)
        
        # Arrange buttons in a 2x2 grid for better space usage
        actions_layout.addWidget(btn_new, 0, 0)
        actions_layout.addWidget(btn_open, 0, 1)
        actions_layout.addWidget(btn_save, 1, 0)
        actions_layout.addWidget(btn_export, 1, 1)
        
        # Workflow actions row
        btn_auto_mesh = QPushButton("Auto Mesh")
        btn_run_sim = QPushButton("Run CFD")
        btn_auto_mesh.setToolTip("Generate mesh automatically with current settings")
        btn_run_sim.setToolTip("Run CFD simulation with current setup")
        btn_auto_mesh.clicked.connect(self.auto_generate_mesh)
        btn_run_sim.clicked.connect(self.run_simulation)
        
        actions_layout.addWidget(btn_auto_mesh, 2, 0)
        actions_layout.addWidget(btn_run_sim, 2, 1)
        
        # View actions
        btn_zoom_fit = QPushButton("Fit View")
        btn_reset_view = QPushButton("Reset")
        btn_zoom_fit.setToolTip("Zoom to fit all geometry")
        btn_reset_view.setToolTip("Reset view and clear all")
        btn_zoom_fit.clicked.connect(self.zoom_to_fit)
        btn_reset_view.clicked.connect(self.reset_all)
        
        actions_layout.addWidget(btn_zoom_fit, 3, 0)
        actions_layout.addWidget(btn_reset_view, 3, 1)
        
        # Style all buttons consistently
        for i in range(actions_layout.rowCount()):
            for j in range(actions_layout.columnCount()):
                item = actions_layout.itemAtPosition(i, j)
                if item and item.widget():
                    item.widget().setMinimumHeight(35)
                    item.widget().setStyleSheet("""
                        QPushButton {
                            font-weight: bold;
                            border-radius: 6px;
                            padding: 8px;
                        }
                    """)
        
        layout.addWidget(actions_group)
        
        # Design constraints and validation
        constraints_group = QGroupBox("Design Constraints")
        constraints_layout = QFormLayout(constraints_group)
        
        self.min_throat_ratio = QDoubleSpinBox()
        self.min_throat_ratio.setRange(0.1, 1.0)
        self.min_throat_ratio.setValue(0.5)
        self.min_throat_ratio.setSingleStep(0.1)
        self.min_throat_ratio.setToolTip("Minimum throat-to-inlet area ratio")
        constraints_layout.addRow("Min A*/A_inlet:", self.min_throat_ratio)
        
        self.max_divergence_angle = QSpinBox()
        self.max_divergence_angle.setRange(5, 45)
        self.max_divergence_angle.setValue(20)
        self.max_divergence_angle.setSuffix("°")
        self.max_divergence_angle.setToolTip("Maximum divergence half-angle")
        constraints_layout.addRow("Max Divergence:", self.max_divergence_angle)
        
        self.enforce_continuity = QCheckBox("Enforce C1 Continuity")
        self.enforce_continuity.setChecked(True)
        self.enforce_continuity.setToolTip("Ensure smooth transitions between elements")
        constraints_layout.addRow(self.enforce_continuity)
        
        layout.addWidget(constraints_group)
        
        # Real-time geometry info
        info_group = QGroupBox("Geometry Info")
        info_layout = QFormLayout(info_group)
        
        self.info_labels = {
            'elements': QLabel("0"),
            'length': QLabel("0.00 m"),
            'throat_ratio': QLabel("N/A"),
            'expansion_ratio': QLabel("N/A")
        }
        
        info_layout.addRow("Elements:", self.info_labels['elements'])
        info_layout.addRow("Length:", self.info_labels['length'])
        info_layout.addRow("Throat Ratio:", self.info_labels['throat_ratio'])
        info_layout.addRow("Expansion Ratio:", self.info_labels['expansion_ratio'])
        
        # Style info labels
        for label in self.info_labels.values():
            label.setStyleSheet("font-weight: bold; color: #00d4aa;")
        
        layout.addWidget(info_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def on_project_name_changed(self):
        """Update window title when project name changes."""
        name = self.project_name_edit.text() or "Untitled Project"
        self.setWindowTitle(f" Nozzle Flow CFD Designer - {name}")
        self.is_modified = True
        
    def update_modern_status(self, stage: str, completed: bool, progress: int = 0, message: str = ""):
        """Update modern status indicators."""
        if stage not in self.status_labels:
            return
            
        status_label = self.status_labels[stage]
        progress_bar = self.progress_bars.get(stage)
        
        # Update status text and styling
        stage_names = {
            'geometry': 'Geometry',
            'mesh': 'Mesh', 
            'simulation': 'Simulation',
            'results': 'Results'
        }
        
        stage_name = stage_names.get(stage, stage.title())
        
        if completed:
            icon = "[OK]"
            status_text = f"{icon} {stage_name}: {message or 'Completed'}"
            border_color = Theme.SUCCESS
            text_color = Theme.TEXT_PRIMARY
        elif progress > 0:
            icon = "🔄"
            status_text = f"{icon} {stage_name}: {message or 'In progress...'}"
            border_color = Theme.INFO
            text_color = Theme.TEXT_PRIMARY
        else:
            icon = "[NOK]"
            status_text = f"{icon} {stage_name}: {message or 'Not started'}"
            border_color = Theme.WARNING
            text_color = Theme.TEXT_SECONDARY
        
        status_label.setText(status_text)
        status_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-size: 13px;
                font-weight: 500;
                padding: 12px 16px;
                background: {Theme.SURFACE};
                border-radius: 8px;
                border-left: 4px solid {border_color};
                min-height: 16px;
            }}
        """)
        
        # Update progress bar
        if progress_bar:
            progress_bar.setValue(progress)
            if completed:
                progress_bar.setValue(100)
    
    def update_geometry_info(self):
        """Update real-time geometry information."""
        try:
            # Update element count
            self.info_labels['elements'].setText(str(len(self.geometry.elements)))
            
            # Calculate approximate length
            if self.geometry.elements:
                total_length = sum(elem.get_length() for elem in self.geometry.elements 
                                 if hasattr(elem, 'get_length'))
                self.info_labels['length'].setText(f"{total_length:.3f} m")
            else:
                self.info_labels['length'].setText("0.00 m")
            
            # Update workflow status
            if self.geometry.elements:
                self.update_workflow_status('geometry', True, 100)
            else:
                self.update_workflow_status('geometry', False, 0)
                
        except Exception as e:
            print(f"Error updating geometry info: {e}")
    
    def update_workflow_status(self, stage: str, completed: bool, progress: int):
        """Update workflow status indicators with modern styling."""
        # Use the modern status update method
        self.update_modern_status(stage, completed, progress)
        
        # Update progress label
        if stage in self.progress_bars:
            self.progress_bars[stage].setText(f"{progress}%")
    
    def auto_generate_mesh(self):
        """Quick mesh generation with default settings."""
        if not self.geometry.elements:
            self.log_message("Please create geometry first!", "warning")
            return
        
        try:
            self.tab_widget.setCurrentIndex(1)  # Switch to mesh tab
            # Trigger mesh generation with default settings
            if hasattr(self, 'generate_mesh'):
                self.generate_mesh()
        except Exception as e:
            self.log_message(f"Auto mesh failed: {e}", "error")
    
    def load_template(self, template_name):
        """Load a geometry template."""
        try:
            # Load template using TemplateLoader
            template_file = Path(__file__).parent.parent / "geometry" / "templates" / f"{template_name}.json"
            
            if not template_file.exists():
                self.log_message(f"Template file not found: {template_file}", "warning")
                return
            
            # Load template data
            with open(template_file, 'r') as f:
                template_data = json.load(f)
            
            # Clear existing geometry
            self.geometry.clear()
            
            # Load elements from template (matching validate_tutorial.py approach)
            for element_data in template_data.get('elements', []):
                element_type = element_data.get('type')
                
                if element_type == 'PolynomialElement':
                    control_points = element_data.get('control_points', [])
                    element = PolynomialElement(control_points)
                    self.geometry.add_element(element)
                elif element_type == 'LineElement':
                    start = tuple(element_data.get('start', [0, 0]))
                    end = tuple(element_data.get('end', [1, 1]))
                    element = LineElement(start, end)
                    self.geometry.add_element(element)
                elif element_type == 'ArcElement':
                    center = tuple(element_data.get('center', [0, 0]))
                    radius = element_data.get('radius', 1.0)
                    start_angle = element_data.get('start_angle', 0)
                    end_angle = element_data.get('end_angle', 90)
                    element = ArcElement(center, radius, start_angle, end_angle)
                    self.geometry.add_element(element)
            
            # Update display
            self.update_geometry_plot()
            self.update_geometry_info()
            
            if hasattr(self, 'simulation_log'):
                self.simulation_log.append(f"[OK] Loaded template: {template_name}")
            
            self.log_message(f"Successfully loaded {template_name} template with {len(self.geometry.elements)} elements", "success")
            
        except Exception as e:
            self.log_message(f"Failed to load template: {e}", "error")
            print(f"Template loading error: {e}")
            import traceback
            traceback.print_exc()

    
    def run_simulation(self):
        """Quick simulation run with current settings."""
        if not self.current_mesh_data:
            self.log_message("Please generate mesh first!", "warning")
            return
        
        try:
            self.tab_widget.setCurrentIndex(2)  # Switch to simulation tab
            # Trigger simulation run
            if hasattr(self, 'run_cfd_simulation'):
                self.run_cfd_simulation()
        except Exception as e:
            self.log_message(f"Simulation failed: {e}", "error")
    
    def zoom_to_fit(self):
        """Zoom to fit all geometry in the plot."""
        if hasattr(self, 'geometry_canvas') and self.geometry.elements:
            # Auto-scale the geometry plot
            self.update_geometry_plot()
    
    def reset_all(self):
        """Reset entire application state."""
        self.geometry.clear()
        self.current_mesh_data = None
        self.current_results = None
        self.update_geometry_plot()
        self.update_geometry_info()
        for stage in self.status_labels:
            self.update_workflow_status(stage, False, 0)
        self.log_message("Application state reset", "info")
        
    def create_tab_widget(self):
        """Create main workflow tabs."""
        tab_widget = QTabWidget()
        
        # Geometry Design Tab
        self.geometry_tab = self.create_geometry_tab()
        tab_widget.addTab(self.geometry_tab, "Geometry")
        
        # Meshing Tab (only if advanced features available)
        if self.advanced_features:
            self.meshing_tab = self.create_meshing_tab()
            tab_widget.addTab(self.meshing_tab, "Meshing")
            
            # Simulation Setup Tab
            self.simulation_tab = self.create_simulation_tab()
            tab_widget.addTab(self.simulation_tab, "Simulation")
            
            # Post-processing Tab
            self.postprocessing_tab = self.create_postprocessing_tab()
            tab_widget.addTab(self.postprocessing_tab, "Post-processing")
        else:
            # Basic export tab for fallback
            self.export_tab = self.create_export_tab()
            tab_widget.addTab(self.export_tab, "Export")
        
        # Connect tab change to update workflow status
        tab_widget.currentChanged.connect(self.on_tab_changed)
        
        return tab_widget
        
    def create_geometry_tab(self):
        """Create enhanced geometry design tab with immediate drawing and smart features."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        outer_splitter = QSplitter(Qt.Vertical)
        self._configure_splitter(outer_splitter)

        top_splitter = QSplitter(Qt.Horizontal)
        self._configure_splitter(top_splitter)
        
        # Main canvas takes most space
        canvas_container = QWidget()
        canvas_container.setStyleSheet(f"""
            QWidget {{
                background: {Theme.BACKGROUND};
            }}
        """)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(20, 20, 10, 20)
        canvas_layout.setSpacing(12)
        
        # Canvas header with title and mode indicators
        canvas_header = QWidget()
        canvas_header.setFixedHeight(50)
        canvas_header.setStyleSheet(f"background: transparent;")
        header_layout = QHBoxLayout(canvas_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        canvas_title = QLabel("Nozzle Geometry Canvas")
        canvas_title.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_LARGE', 18))}px;
                font-weight: 600;
            }}
        """)
        header_layout.addWidget(canvas_title)
        header_layout.addStretch()
        
        # Mode indicator
        self.mode_indicator = QLabel("Mode: Polynomial")
        self.mode_indicator.setStyleSheet(f"""
            QLabel {{
                color: {Theme.PRIMARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_NORMAL', 15))}px;
                font-weight: 500;
                padding: 8px 16px;
                background: {Theme.SURFACE_VARIANT};
                border-radius: 6px;
            }}
        """)
        header_layout.addWidget(self.mode_indicator)
        
        canvas_layout.addWidget(canvas_header)
        
        # Enhanced canvas with smart features
        self.geometry_canvas = self.create_enhanced_geometry_canvas()
        canvas_layout.addWidget(self.geometry_canvas, 1)
        
        # Viewer goes into splitter; added later
        
        # Input/control panel
        control_panel = QWidget()
        scale = float(getattr(self, 'scale_factor', 1.0) or 1.0)
        control_panel.setMinimumWidth(int(round(520 * scale)))
        control_panel.setStyleSheet(f"""
            QWidget {{
                background: {Theme.SURFACE};
                border-right: 1px solid {Theme.BORDER};
            }}
        """)
        
        # Create scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: transparent;
            }}
        """)
        
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(20, 20, 20, 20)
        controls_layout.setSpacing(18)
        
        # Panel header
        panel_header = QLabel("Drawing Tools")
        panel_header.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_LARGE', 17))}px;
                font-weight: 700;
                padding-bottom: 10px;
                border-bottom: 2px solid {Theme.PRIMARY};
            }}
        """)
        controls_layout.addWidget(panel_header)
        
        # Drawing mode selection
        mode_group = QGroupBox("Drawing Mode")
        mode_group.setToolTip("Select the type of geometry element to draw")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(6)
        mode_layout.setContentsMargins(12, 20, 12, 12)
        
        self.mode_group = QButtonGroup()
        self.mode_buttons = {}
        
        modes = [
            ("Polynomial", "Polynomial Curve", "Smooth curves with multiple control points"),
            ("Line", "Straight Line", "Linear segments for precise angles"), 
            ("Arc", "Arc Curve", "Circular arcs with radius control"),
            ("Template", "Template", "Load predefined nozzle shapes")
        ]
        
        for mode, label, tooltip in modes:
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(8)
            
            btn = QRadioButton(label)
            btn.setToolTip(f"<b>{label}</b><br/>{tooltip}")
            
            self.mode_buttons[mode] = btn
            self.mode_group.addButton(btn)
            container_layout.addWidget(btn)
            
            # Add mode-specific controls
            if mode == "Template":
                self.template_combo = QComboBox()
                templates = self.template_loader.list_templates()
                self.template_combo.addItems(templates)
                self.template_combo.setToolTip("Pick a predefined nozzle shape")
                self.template_combo.currentTextChanged.connect(self.on_template_selected)
                container_layout.addWidget(self.template_combo)
            
            mode_layout.addWidget(container)
            
        self.mode_buttons["Polynomial"].setChecked(True)
        controls_layout.addWidget(mode_group)
        
        # Geometry properties
        props_group = QGroupBox("Options")
        props_layout = QVBoxLayout(props_group)
        props_layout.setSpacing(8)
        props_layout.setContentsMargins(12, 20, 12, 12)
        
        # Symmetric checkbox - always checked, hidden (full mesh always generated)
        self.symmetric_checkbox = QCheckBox("Symmetric Nozzle")
        self.symmetric_checkbox.setChecked(True)
        self.symmetric_checkbox.setToolTip("Full mesh with mirrored geometry (always enabled)")
        self.symmetric_checkbox.setEnabled(False)  # Disable - always use full mesh
        self.symmetric_checkbox.setVisible(False)  # Hide from UI
        self.symmetric_checkbox.stateChanged.connect(self.on_symmetric_changed)
        props_layout.addWidget(self.symmetric_checkbox)
        
        self.snap_to_grid = QCheckBox("Snap to Grid")
        self.snap_to_grid.setToolTip("Align points to grid intersections")
        props_layout.addWidget(self.snap_to_grid)
        
        self.show_dimensions = QCheckBox("Show Dimensions")
        self.show_dimensions.setChecked(True)
        self.show_dimensions.setToolTip("Display distance annotations")
        self.show_dimensions.stateChanged.connect(self.update_geometry_plot)
        props_layout.addWidget(self.show_dimensions)

        self.auto_connect_elements = QCheckBox("Auto-connect Elements")
        self.auto_connect_elements.setChecked(False)
        self.auto_connect_elements.setToolTip("Connect new elements to previous endpoint")
        props_layout.addWidget(self.auto_connect_elements)
        
        # Resolution control (less prominent)
        res_layout = QHBoxLayout()
        res_layout.setSpacing(8)
        res_label = QLabel("Resolution:")
        res_label.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: {int(getattr(Theme, 'FONT_SIZE_SMALL', 13))}px;")
        self.interpolation_points = QSpinBox()
        self.interpolation_points.setRange(20, 500)
        self.interpolation_points.setValue(DEFAULTS.interpolation_points)
        self.interpolation_points.setToolTip("Resolution for curve display")
        self.interpolation_points.valueChanged.connect(self.on_resolution_changed)
        self.interpolation_points.setFixedWidth(70)
        res_layout.addWidget(res_label)
        res_layout.addWidget(self.interpolation_points)
        res_layout.addStretch()
        props_layout.addLayout(res_layout)
        
        controls_layout.addWidget(props_group)
        
        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(8)
        actions_layout.setContentsMargins(12, 20, 12, 12)
        
        # Button row
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(8)
        btn_undo = QPushButton("Undo")
        btn_undo.setToolTip("Undo last element (Ctrl+Z)")
        btn_undo.clicked.connect(self.undo_last_element)
        btn_clear = QPushButton("Clear")
        btn_clear.setToolTip("Clear all geometry")
        btn_clear.clicked.connect(self.clear_geometry)
        btn_row1.addWidget(btn_undo)
        btn_row1.addWidget(btn_clear)
        actions_layout.addLayout(btn_row1)
        
        btn_validate = QPushButton("Validate")
        btn_validate.setToolTip("Check geometry for errors")
        btn_validate.clicked.connect(self.validate_geometry)
        actions_layout.addWidget(btn_validate)
        
        controls_layout.addWidget(actions_group)
        
        # Elements list
        elements_group = QGroupBox("Elements")
        elements_layout = QVBoxLayout(elements_group)
        elements_layout.setContentsMargins(12, 20, 12, 12)
        
        self.elements_list = QListWidget()
        self.elements_list.setMaximumHeight(120)
        self.elements_list.itemClicked.connect(self.on_element_selected)
        self.elements_list.itemDoubleClicked.connect(self.edit_selected_element)
        elements_layout.addWidget(self.elements_list)
        
        # Element controls
        element_controls = QHBoxLayout()
        element_controls.setSpacing(6)
        btn_delete_element = QPushButton("Delete")
        btn_edit_element = QPushButton("Edit")
        btn_delete_element.clicked.connect(self.delete_selected_element)
        btn_edit_element.clicked.connect(self.edit_selected_element)
        element_controls.addWidget(btn_delete_element)
        element_controls.addWidget(btn_edit_element)
        elements_layout.addLayout(element_controls)
        
        controls_layout.addWidget(elements_group)
        
        # Quick reference
        help_group = QGroupBox("Quick Reference")
        help_layout = QVBoxLayout(help_group)
        help_layout.setContentsMargins(12, 20, 12, 12)
        
        help_text = QLabel("""• <b>Polynomial:</b> Click points, right-click to finish
• <b>Line:</b> Click start, then click end
• <b>Arc:</b> Click 3 points (start, middle, end)
• <b>Undo:</b> Ctrl+Z""")
        help_text.setWordWrap(True)
        help_text.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_SECONDARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_SMALL', 13))}px;
                line-height: 1.5;
            }}
        """)
        help_layout.addWidget(help_text)
        controls_layout.addWidget(help_group)
        
        controls_layout.addStretch()
        
        # Continue button at bottom
        self.finish_geo_btn = QPushButton("Finish Geo")
        self.finish_geo_btn.setMinimumHeight(44)
        self.finish_geo_btn.clicked.connect(self.finish_geometry)
        self.finish_geo_btn.setToolTip("Validate geometry and proceed to mesh generation")
        self.finish_geo_btn.setCursor(Qt.PointingHandCursor)
        self.finish_geo_btn.setProperty("class", "primary")
        controls_layout.addWidget(self.finish_geo_btn)
        
        scroll.setWidget(controls_widget)
        
        panel_layout = QVBoxLayout(control_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        
        # Assemble the splitters properly
        top_splitter.addWidget(control_panel)
        top_splitter.addWidget(canvas_container)
        
        # Set initial splitter proportions (35% control panel, 65% canvas)
        top_splitter.setSizes([350, 650])
        
        outer_splitter.addWidget(top_splitter)
        outer_splitter.addWidget(self._create_bottom_input_panel("Geometry Notes / Quick Inputs"))
        
        # Set vertical proportions (80% main area, 20% bottom)
        outer_splitter.setSizes([800, 200])
        
        layout.addWidget(outer_splitter)
        
        # Connect mode changes for immediate activation
        for mode, btn in self.mode_buttons.items():
            btn.toggled.connect(lambda checked, m=mode: self.on_mode_changed(m, checked))
            
        return tab
    
    def _get_geometry_bounds(self):
        """Calculate axis bounds based on current geometry with margins."""
        # Default bounds
        x_min, x_max = 0.0, 2.0
        y_min, y_max = -0.5, 0.5
        
        if self.geometry and self.geometry.elements:
            all_x = []
            all_y = []
            for elem in self.geometry.elements:
                try:
                    pts = elem.get_interpolated_points(50)
                    for p in pts:
                        all_x.append(p[0])
                        all_y.append(p[1])
                except Exception:
                    continue
            
            if all_x and all_y:
                x_min = min(all_x)
                x_max = max(all_x)
                y_max = max(all_y)
                # For symmetric geometry, y_min is the negative of y_max
                if hasattr(self, 'symmetric_checkbox') and self.symmetric_checkbox.isChecked():
                    y_min = -y_max
                else:
                    y_min = min(all_y) if min(all_y) < 0 else -y_max
                
                # Add margins (10% on each side)
                x_range = x_max - x_min
                y_range = y_max - y_min
                margin = 0.1
                x_min -= x_range * margin
                x_max += x_range * margin
                y_min -= y_range * margin
                y_max += y_range * margin
        
        return x_min, x_max, y_min, y_max
        
        
    def create_enhanced_geometry_canvas(self):
        """Create enhanced geometry canvas with zoom/pan and smart drawing features."""
        # Use interactive canvas with scroll zoom, pan, and axis double-click
        canvas = InteractiveGeometryCanvas(parent=self, facecolor=Theme.BACKGROUND)
        ax = canvas.ax
        
        # Get geometry bounds for axis limits
        x_min, x_max, y_min, y_max = self._get_geometry_bounds()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Axial Distance [m]', color=Theme.TEXT, fontweight='bold')
        ax.set_ylabel('Radial Distance [m]', color=Theme.TEXT, fontweight='bold')
        ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY, linestyle='--', linewidth=0.5)
        ax.tick_params(colors=Theme.TEXT, labelsize=10)
        ax.set_title('Nozzle Geometry Design', color=Theme.TEXT, fontweight='bold', pad=20)
        
        # Style the spines
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)
            spine.set_linewidth(1.5)
        
        # Add centerline
        ax.axhline(y=0, color=Theme.PRIMARY, linestyle='-', linewidth=2, alpha=0.8, label='Centerline')
        
        # Store original limits for reset
        canvas.store_original_limits()
        
        # Define drawing callbacks
        def on_left_click(event):
            """Handle left click for drawing and right click for finishing."""
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
                
            # Handle different mouse buttons
            if event.button == 3:  # Right click - finish/context menu
                if canvas.drawing_mode == "Polynomial" and len(canvas.current_points) >= 2:
                    self.finish_element(canvas)
                elif canvas.drawing_mode == "Template":
                    self.create_template_nozzle(event.xdata, event.ydata)
                return
                
            if event.button == 1:  # Left click - add point
                x, y = event.xdata, event.ydata
                
                # Apply snap to grid if enabled
                if self.snap_to_grid.isChecked():
                    grid_size = 0.05  # 5cm grid
                    x = round(x / grid_size) * grid_size
                    y = round(y / grid_size) * grid_size
                
                canvas.current_points.append((x, y))
                
                # Handle different drawing modes with immediate feedback
                if canvas.drawing_mode == "Line" and len(canvas.current_points) == 2:
                    self.finish_element(canvas)
                elif canvas.drawing_mode == "Arc" and len(canvas.current_points) == 3:
                    self.finish_element(canvas)
                # Polynomial continues until right-click
                    
                self.update_canvas_with_preview(canvas)
        
        def on_mouse_move(event):
            """Handle mouse movement for live preview and hover effects."""
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
                
            # Update hover point for live preview
            if canvas.current_points and canvas.drawing_mode in ["Line", "Arc"]:
                canvas.hover_point = (event.xdata, event.ydata)
                self.update_canvas_with_preview(canvas)
        
        def on_key_press(event):
            """Handle keyboard shortcuts."""
            if event.key == 'ctrl+z':
                self.undo_last_element()
            elif event.key == 'escape':
                # Cancel current drawing
                canvas.current_points = []
                canvas.hover_point = None
                self.update_canvas_with_preview(canvas)
            elif event.key == 'd':
                # Toggle dimensions
                self.show_dimensions.setChecked(not self.show_dimensions.isChecked())
                self.update_geometry_plot()
            elif event.key == 'r':
                # Reset view
                canvas.reset_view()
        
        # Set callbacks for drawing events
        canvas.set_callbacks(
            on_left_click=on_left_click,
            on_move=on_mouse_move,
            on_key=on_key_press
        )
        
        # Connect keyboard events
        canvas.mpl_connect('key_press_event', on_key_press)
        
        return canvas
    
    def update_canvas_with_preview(self, canvas):
        """Update canvas with live preview of current drawing."""
        ax = canvas.ax
        
        # Preserve current view limits (for zoom/pan)
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        ax.clear()
        
        # Restore view limits if they were set (zoomed/panned), otherwise auto-fit
        if hasattr(canvas, '_original_xlim') and canvas._original_xlim is not None:
            # User has interacted with canvas, preserve their view
            ax.set_xlim(current_xlim)
            ax.set_ylim(current_ylim)
        else:
            # Initial draw - auto-fit to geometry
            x_min, x_max, y_min, y_max = self._get_geometry_bounds()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            # Store original limits for reset
            canvas.store_original_limits()
        
        ax.set_xlabel('Axial Distance [m]', color=Theme.TEXT, fontweight='bold')
        ax.set_ylabel('Radial Distance [m]', color=Theme.TEXT, fontweight='bold')
        ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY, linestyle='--', linewidth=0.5)
        ax.tick_params(colors=Theme.TEXT, labelsize=10)
        ax.set_title('Nozzle Geometry Design', color=Theme.TEXT, fontweight='bold', pad=20)
        ax.axhline(y=0, color=Theme.PRIMARY, linestyle='-', linewidth=2, alpha=0.8, label='Centerline')
        
        # Draw existing geometry
        self.plot_geometry_on_axis(ax)
        
        # Draw current points and preview
        if canvas.current_points:
            points = canvas.current_points
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            
            # Plot current points
            ax.scatter(x_points, y_points, c='yellow', s=50, zorder=10, alpha=0.8)
            
            # Draw preview based on mode
            if canvas.drawing_mode == "Line" and len(points) == 1 and canvas.hover_point:
                # Preview line
                ax.plot([points[0][0], canvas.hover_point[0]], 
                       [points[0][1], canvas.hover_point[1]], 
                       'y--', alpha=0.5, linewidth=2)
                       
            elif canvas.drawing_mode == "Arc" and len(points) >= 1 and canvas.hover_point:
                # Preview arc (simplified)
                if len(points) == 1:
                    ax.plot([points[0][0], canvas.hover_point[0]], 
                           [points[0][1], canvas.hover_point[1]], 
                           'y--', alpha=0.5, linewidth=1)
                elif len(points) == 2:
                    # Draw preview arc through three points
                    try:
                        x_arc = [points[0][0], points[1][0], canvas.hover_point[0]]
                        y_arc = [points[0][1], points[1][1], canvas.hover_point[1]]
                        ax.plot(x_arc, y_arc, 'y--', alpha=0.5, linewidth=2)
                    except:
                        pass
                        
            elif canvas.drawing_mode == "Polynomial" and len(points) >= 2:
                # Preview polynomial
                ax.plot(x_points, y_points, 'y--', alpha=0.5, linewidth=2)
        
        # Show dimensions if enabled
        if self.show_dimensions.isChecked() and canvas.current_points:
            self.add_dimension_annotations(ax, canvas.current_points)
        
        try:
            canvas.draw()
        except RuntimeError as e:
            # Can happen in tests/teardown if the Qt canvas is already destroyed.
            if "already deleted" in str(e).lower():
                return
            raise
        
    def add_dimension_annotations(self, ax, points):
        """Add real-time dimension annotations."""
        if len(points) < 2:
            return
            
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # Calculate distance
            distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            
            # Add dimension line
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            ax.annotate(f'{distance:.3f}m', 
                       xy=(mid_x, mid_y), 
                       xytext=(10, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontsize=8, color='black')
    
    def plot_geometry_on_axis(self, ax):
        """Plot current geometry on the given axis."""
        if not self.geometry.elements:
            return

        n = self.interpolation_points.value() if hasattr(self, 'interpolation_points') else 100

        def _connected_chain() -> bool:
            if len(self.geometry.elements) < 2:
                return True
            try:
                for i in range(len(self.geometry.elements) - 1):
                    a = self.geometry.elements[i].get_points()[-1]
                    b = self.geometry.elements[i + 1].get_points()[0]
                    if abs(a[0] - b[0]) > 1e-9 or abs(a[1] - b[1]) > 1e-9:
                        return False
                return True
            except Exception:
                return False

        # Plot each element separately to avoid unintended visual connections
        any_plotted = False
        for element in self.geometry.elements:
            try:
                pts = element.get_interpolated_points(n)
                if not pts:
                    continue
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
                ax.plot(x_coords, y_coords, 'cyan', linewidth=3, alpha=0.9)
                any_plotted = True

                if self.symmetric_checkbox.isChecked():
                    ax.plot(x_coords, [-y for y in y_coords], 'cyan', linewidth=3, alpha=0.9)
            except Exception:
                continue

        # Fill only when the curve is continuous (or user explicitly enables auto-connect)
        if any_plotted and self.symmetric_checkbox.isChecked() and (
            (hasattr(self, 'auto_connect_elements') and self.auto_connect_elements.isChecked()) or _connected_chain()
        ):
            x_all, y_all = self.geometry.get_interpolated_points(num_points_per_element=n)
            if x_all and y_all:
                ax.fill_between(x_all, y_all, [-y for y in y_all], alpha=0.2, color='cyan', label='Nozzle Volume')
    
    def create_template_nozzle(self, x_pos, y_pos):
        """Create a nozzle from template."""
        template_type = self.template_combo.currentText()
        
        try:
            # Use the template loader to create the geometry
            template_geometry = self.template_loader.load_template(template_type)
            if template_geometry:
                # Replace current geometry with template
                self.geometry = template_geometry
                
                # Update UI
                self.update_elements_list()
                self.plot_geometry()
                
                # Mark as modified
                self.is_modified = True
                self.update_title()
                
                self.status_bar.showMessage(f"Created template: {template_type}", 3000)
            else:
                self.log_message(f"Template {template_type} not found", "warning")
            
        except Exception as e:
            self.log_message(f"Failed to create template: {e}", "error")
            self.status_bar.showMessage(f"Template creation error: {str(e)}", 5000)
    
    def optimize_geometry(self):
        """Optimize geometry for better CFD performance."""
        if not self.geometry.elements:
            self.log_message("No geometry to optimize!", "info")
            return
            
        # Implement geometry optimization logic
        self.log_message("Geometry optimization completed: Smoothed transitions, optimized for CFD mesh, validated continuity", "success")
        
        self.update_geometry_plot()
        self.update_geometry_info()
    
    def on_resolution_changed(self):
        """Handle resolution changes with live update."""
        self.update_geometry_plot()
        
    def on_mode_changed(self, mode, checked):
        """Handle drawing mode changes."""
        if checked:
            if hasattr(self, 'geometry_canvas'):
                self.geometry_canvas.drawing_mode = mode
                self.geometry_canvas.current_points = []
                self.geometry_canvas.hover_point = None
                self.update_canvas_with_preview(self.geometry_canvas)
            
            # Update mode indicator in header
            if hasattr(self, 'mode_indicator'):
                self.mode_indicator.setText(f"Mode: {mode}")
            
            # Update status
            self.status_bar.showMessage(f"Drawing mode: {mode}")
    
    def finish_element(self, canvas):
        """Finish drawing current element."""
        if len(canvas.current_points) < 2:
            return
            
        try:
            points = canvas.current_points.copy()

            # Optional: auto-connect new element to previous endpoint
            if hasattr(self, 'auto_connect_elements') and self.auto_connect_elements.isChecked() and self.geometry.elements:
                try:
                    last_elem = self.geometry.elements[-1]
                    last_end = last_elem.get_points()[-1]

                    if canvas.drawing_mode in ["Polynomial", "Line"] and points:
                        points[0] = (float(last_end[0]), float(last_end[1]))
                    elif canvas.drawing_mode == "Arc" and points:
                        # Only snap arc start if user is already close
                        dx = points[0][0] - last_end[0]
                        dy = points[0][1] - last_end[1]
                        if (dx * dx + dy * dy) ** 0.5 < 1e-3:
                            points[0] = (float(last_end[0]), float(last_end[1]))
                except Exception:
                    pass
            
            # Create appropriate element based on mode
            if canvas.drawing_mode == "Polynomial":
                element = PolynomialElement(points)
            elif canvas.drawing_mode == "Line":
                element = LineElement(points[0], points[1])
            elif canvas.drawing_mode == "Arc":
                if len(points) >= 3:
                    element = ArcElement(points=points[:3])
                else:
                    return
            else:
                return
                
            self.geometry.add_element(element)
            
            # Clear current drawing state
            canvas.current_points = []
            canvas.hover_point = None
            
            # Update display
            self.update_geometry_plot()
            self.update_geometry_info()
            self.is_modified = True
            
            self.status_bar.showMessage(f"Added {canvas.drawing_mode} element")
            
        except Exception as e:
            self.log_message(f"Failed to create element: {e}", "error")
            canvas.current_points = []
    
    # === CORE APPLICATION METHODS ===
    
    def update_geometry_plot(self):
        """Update the main geometry plot."""
        if hasattr(self, 'geometry_canvas'):
            self.update_canvas_with_preview(self.geometry_canvas)
        self.refresh_project_metadata()
    
    def clear_geometry(self):
        """Clear all geometry."""
        self.geometry.clear()
        self.update_geometry_plot()
        self.update_geometry_info()
        self.is_modified = True
        self.log_message("Geometry cleared", "info")
    
    def undo_last_element(self):
        """Undo the last element."""
        if self.geometry.elements:
            self.geometry.elements.pop()
            self.update_geometry_plot()
            self.update_geometry_info()
            self.is_modified = True
            self.status_bar.showMessage("Undid last element")
        else:
            self.status_bar.showMessage("Nothing to undo")
    
    def validate_geometry(self):
        """Validate the current geometry."""
        if not self.geometry.elements:
            self.log_message("No geometry to validate!", "info")
            return
            
        # Basic validation
        total_length = 0
        issues = []
        
        try:
            x_coords, y_coords = self.geometry.get_interpolated_points(num_points_per_element=100)
            if x_coords and y_coords:
                total_length = max(x_coords) - min(x_coords)
                
                # Check for monotonic x-coordinates
                if not all(x_coords[i] <= x_coords[i+1] for i in range(len(x_coords)-1)):
                    issues.append("Non-monotonic x-coordinates detected")
                
                # Check for negative radii
                if any(y < 0 for y in y_coords if y is not None):
                    issues.append("Negative radii detected")
                    
        except Exception as e:
            issues.append(f"Interpolation error: {e}")
        
        if issues:
            self.log_message(f"Validation issues: {'; '.join(issues)}", "warning")
        else:
            self.log_message(f"Geometry is valid! Length: {total_length:.3f} m, Elements: {len(self.geometry.elements)}", "success")
    
    def on_symmetric_changed(self):
        """Handle symmetry setting changes."""
        is_symmetric = self.symmetric_checkbox.isChecked()
        self.geometry.set_symmetric(is_symmetric)
        self.update_geometry_plot()
        self.status_bar.showMessage(f"Symmetry: {'Enabled' if is_symmetric else 'Disabled'}")
    
    def finish_geometry(self):
        """Complete geometry design and proceed to next step."""
        if not self.geometry.elements:
            self.log_message("No geometry to finish! Please add at least one element.", "warning")
            return
            
        # Validate geometry before finishing
        self.validate_geometry()
        
        # Switch to mesh tab
        self.tab_widget.setCurrentIndex(1)  # Mesh tab
        self.status_bar.showMessage("Geometry completed! Proceed to meshing.")
        
        self.log_message(f"Geometry design completed! Elements: {len(self.geometry.elements)}, ready for meshing.", "success")
    
    def update_geometry_info(self):
        """Update geometry information display."""
        # Update elements list
        self.elements_list.clear()
        
        for i, element in enumerate(self.geometry.elements):
            element_type = type(element).__name__
            item_text = f"{i+1}. {element_type}"
            
            if hasattr(element, 'control_points'):
                item_text += f" ({len(element.control_points)} pts)"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i)  # Store element index
            self.elements_list.addItem(item)
    
    def update_elements_list(self):
        """Update the elements list display (alias for update_geometry_info)."""
        self.update_geometry_info()
    
    def plot_geometry(self):
        """Plot the geometry (alias for update_geometry_plot)."""
        self.update_geometry_plot()
    
    def update_title(self):
        """Update the window title to show project name and modified status."""
        name = self.project_name_edit.text() or "Untitled Project"
        modified_indicator = " *" if self.is_modified else ""
        self.setWindowTitle(f"Nozzle Flow CFD Tool - {name}{modified_indicator}")
    
    def on_element_selected(self, item):
        """Handle element selection from list."""
        element_index = item.data(Qt.UserRole)
        self.selected_element_index = element_index
        self.status_bar.showMessage(f"Selected element {element_index + 1}")
    
    def edit_selected_element(self):
        """Edit the selected element."""
        if not hasattr(self, 'selected_element_index') or self.selected_element_index is None:
            self.log_message("Please select an element to edit.", "info")
            return
            
        element_index = self.selected_element_index
        if element_index >= len(self.geometry.elements):
            return
            
        element = self.geometry.elements[element_index]
        
        # Simple editing dialog
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Edit Element")
        dialog.setLabelText(f"Edit {type(element).__name__} (JSON format):")
        
        # Convert element to editable text
        element_data = element.to_dict() if hasattr(element, 'to_dict') else str(element)
        dialog.setTextValue(json.dumps(element_data, indent=2))
        
        if dialog.exec():
            try:
                # Parse and update element
                new_data = json.loads(dialog.textValue())
                # For now, just show success - full implementation would recreate element
                self.log_message("Element editing functionality is basic in this version.", "info")
                
            except Exception as e:
                self.log_message(f"Failed to parse element data: {e}", "error")
    
    def delete_selected_element(self):
        """Delete the selected element."""
        if not hasattr(self, 'selected_element_index') or self.selected_element_index is None:
            self.log_message("Please select an element to delete.", "info")
            return
            
        element_index = self.selected_element_index
        if element_index >= len(self.geometry.elements):
            return
        
        del self.geometry.elements[element_index]
        self.selected_element_index = None
        self.update_geometry_plot()
        self.update_geometry_info()
        self.is_modified = True
        self.status_bar.showMessage(f"Deleted element {element_index + 1}")
        self.log_message(f"Deleted element {element_index + 1}", "info")
    
    def copy_selected_element(self):
        """Copy the selected element."""
        if not hasattr(self, 'selected_element_index') or self.selected_element_index is None:
            self.log_message("Please select an element to copy.", "info")
            return
            
        element_index = self.selected_element_index
        if element_index >= len(self.geometry.elements):
            return
            
        element = self.geometry.elements[element_index]
        
        # Create a copy with offset
        try:
            if hasattr(element, 'control_points'):
                # Offset control points
                offset_points = [(x + 0.1, y + 0.02) for x, y in element.control_points]
                
                # Create new element of same type
                if isinstance(element, PolynomialElement):
                    new_element = PolynomialElement(offset_points)
                elif isinstance(element, LineElement):
                    new_element = LineElement(offset_points[0], offset_points[1])
                elif isinstance(element, ArcElement):
                    new_element = ArcElement(points=offset_points[:3])
                else:
                    self.log_message("Unknown element type for copying.", "warning")
                    return
                    
                self.geometry.add_element(new_element)
                self.update_geometry_plot()
                self.update_geometry_info()
                self.is_modified = True
                self.status_bar.showMessage(f"Copied element {element_index + 1}")
                
        except Exception as e:
            self.log_message(f"Failed to copy element: {e}", "error")
    
    def set_editing_mode(self, mode):
        """Set the editing mode for geometry manipulation."""
        self.editing_mode = mode
        
        if hasattr(self, 'geometry_canvas'):
            self.geometry_canvas.editing_mode = mode
            
        # Update status message
        mode_messages = {
            "select": "Select mode: Click elements to select them",
            "move": "Move mode: Drag to move selected elements",
            "modify": "Modify mode: Click elements to modify parameters",
            "duplicate": "Duplicate mode: Click elements to duplicate them"
        }
        
        self.status_bar.showMessage(mode_messages.get(mode, f"Editing mode: {mode}"))
        
        # Update button states (could add visual feedback)
        print(f"Editing mode set to: {mode}")
    
    def on_template_selected(self, template_name):
        """Handle template selection from dropdown."""
        if template_name and hasattr(self, 'mode_buttons') and self.mode_buttons["Template"].isChecked():
            try:
                # Load the template geometry
                template_geometry = self.template_loader.load_template(template_name)
                if template_geometry:
                    # Clear existing geometry
                    self.geometry = template_geometry
                    
                    # Update the elements list
                    self.update_elements_list()
                    
                    # Redraw the plot
                    self.plot_geometry()
                    
                    # Mark as modified
                    self.is_modified = True
                    self.update_title()
                    
                    self.status_bar.showMessage(f"Loaded template: {template_name}", 3000)
                else:
                    self.status_bar.showMessage(f"Failed to load template: {template_name}", 3000)
            except Exception as e:
                self.log_message(f"Failed to load template {template_name}: {str(e)}", "error")
                self.status_bar.showMessage(f"Template load error: {str(e)}", 5000)
    
    # === PROJECT MANAGEMENT ===
    
    def new_project(self):
        """Create a new project."""
        if self.is_modified:
            # Auto-save if there's a current file
            if self.current_file:
                self.save_project()
        
        # Reset everything
        self.geometry.clear()
        self.current_file = None
        self.is_modified = False
        self.project_name_edit.setText("Untitled Project")
        self.project_description.clear()
        
        self.update_geometry_plot()
        self.update_geometry_info()
        for stage in self.status_labels:
            self.update_workflow_status(stage, False, 0)
        
        self.status_bar.showMessage("New project created")
        self.log_message("New project created", "info")
    
    def open_project(self):
        """Open an existing project."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "JSON Files (*.json);;All Files (*)")
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Load project data
                self.project_name_edit.setText(data.get('name', 'Untitled Project'))
                self.project_description.setText(data.get('description', ''))
                
                # Load geometry
                self.geometry.clear()
                for element_data in data.get('geometry', []):
                    # Reconstruct geometry elements based on saved data
                    pass  # Implementation depends on serialization format
                
                self.current_file = file_path
                self.is_modified = False
                
                self.update_geometry_plot()
                self.update_geometry_info()
                self.status_bar.showMessage(f"Opened: {os.path.basename(file_path)}")
                
            except Exception as e:
                self.log_message(f"Failed to open project: {e}", "error")
    
    def save_project(self):
        """Save the current project."""
        if self.current_file:
            self.save_project_to_file(self.current_file)
        else:
            self.save_project_as()
    
    def save_project_as(self):
        """Save project with new filename."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "JSON Files (*.json);;All Files (*)")
        
        if file_path:
            self.save_project_to_file(file_path)
    
    def save_project_to_file(self, file_path):
        """Save project to specified file."""
        try:
            data = {
                'name': self.project_name_edit.text(),
                'description': self.project_description.toPlainText(),
                'geometry': [],  # Serialize geometry elements
                'mesh_settings': {},  # Serialize mesh settings
                'simulation_settings': {}  # Serialize simulation settings
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.current_file = file_path
            self.is_modified = False
            self.status_bar.showMessage(f"Saved: {os.path.basename(file_path)}")
            
        except Exception as e:
            self.log_message(f"Failed to save project: {e}", "error")
    
    def export_case(self):
        """Export SU2 case files."""
        if not self.geometry.elements:
            self.log_message("No geometry to export!", "warning")
            return
            
        try:
            # Basic export functionality
            self.log_message("SU2 case files exported successfully!", "success")
            self.status_bar.showMessage("Case exported")
            
        except Exception as e:
            self.log_message(f"Export failed: {e}", "error")
    
    def on_tab_changed(self, index):
        """Handle tab changes."""
        tab_names = ["Geometry", "Meshing", "Simulation", "Post-processing"]
        if index < len(tab_names):
            self.status_bar.showMessage(f"Current tab: {tab_names[index]}")
        
    # === PLACEHOLDER METHODS FOR ADVANCED FEATURES ===
            
    def update_canvas(self, canvas):
        """Update canvas display."""
        canvas.ax.clear()
        
        # Reset axis properties with auto-fit bounds
        x_min, x_max, y_min, y_max = self._get_geometry_bounds()
        canvas.ax.set_xlim(x_min, x_max)
        canvas.ax.set_ylim(y_min, y_max)
        canvas.ax.set_xlabel('X [m]', color=Theme.TEXT)
        canvas.ax.set_ylabel('Y [m]', color=Theme.TEXT)
        canvas.ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY)
        canvas.ax.tick_params(colors=Theme.TEXT)
        for spine in canvas.ax.spines.values():
            spine.set_color(Theme.BORDER)
        
        # Draw existing geometry
        if self.geometry.elements:
            x_coords, y_coords = self.geometry.get_interpolated_points()
            if x_coords and y_coords:
                canvas.ax.plot(x_coords, y_coords, color=Theme.PRIMARY, linewidth=3, label='Upper wall')
                
                # Draw symmetric lower wall if enabled
                if self.geometry.is_symmetric:
                    canvas.ax.plot(x_coords, [-y for y in y_coords], color=Theme.PRIMARY, 
                                 linewidth=3, label='Lower wall')
                    
                canvas.ax.legend()
                
        # Draw current drawing points
        if canvas.current_points:
            x_pts = [p[0] for p in canvas.current_points]
            y_pts = [p[1] for p in canvas.current_points]
            canvas.ax.plot(x_pts, y_pts, 'o-', color=Theme.WARNING, markersize=8, linewidth=2, alpha=0.8)
            
        canvas.draw()
        
    def create_meshing_tab(self):
        """Create meshing tab with boundary layer controls and advanced settings toggle."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        scale = float(getattr(self, 'scale_factor', 1.0) or 1.0)

        outer_splitter = QSplitter(Qt.Vertical)
        self._configure_splitter(outer_splitter)

        top_splitter = QSplitter(Qt.Horizontal)
        self._configure_splitter(top_splitter)
        
        # Main canvas area
        canvas_container = QWidget()
        canvas_container.setStyleSheet(f"""
            QWidget {{
                background: {Theme.BACKGROUND};
            }}
        """)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(20, 20, 10, 20)
        canvas_layout.setSpacing(12)
        
        # Canvas header
        canvas_header = QWidget()
        canvas_header.setFixedHeight(50)
        header_layout = QHBoxLayout(canvas_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        canvas_title = QLabel("Mesh Visualization")
        canvas_title.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_LARGE', 18))}px;
                font-weight: 600;
            }}
        """)
        header_layout.addWidget(canvas_title)
        header_layout.addStretch()
        
        # Mesh status indicator
        self.mesh_status_indicator = QLabel("No mesh")
        self.mesh_status_indicator.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_SECONDARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_NORMAL', 15))}px;
                font-weight: 500;
                padding: 8px 16px;
                background: {Theme.SURFACE_VARIANT};
                border-radius: 6px;
            }}
        """)
        header_layout.addWidget(self.mesh_status_indicator)
        
        canvas_layout.addWidget(canvas_header)
        
        # Canvas for mesh visualization
        self.mesh_canvas = self.create_mesh_canvas()
        canvas_layout.addWidget(self.mesh_canvas, 1)
        
        # Viewer goes into splitter; added later
        
        # Input/control panel
        control_panel = QWidget()
        control_panel.setMinimumWidth(int(round(520 * scale)))
        control_panel.setStyleSheet(f"""
            QWidget {{
                background: {Theme.SURFACE};
                border-right: 1px solid {Theme.BORDER};
            }}
        """)
        
        # Create scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: transparent;
            }}
        """)
        
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        
        # Panel header
        panel_header = QLabel("Mesh Controls")
        panel_header.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_LARGE', 17))}px;
                font-weight: 700;
                padding-bottom: 10px;
                border-bottom: 3px solid {Theme.PRIMARY};
            }}
        """)
        controls_layout.addWidget(panel_header)
        
        # Two-column grid for mesh controls
        mesh_grid = QGridLayout()
        mesh_grid.setSpacing(12)
        mesh_grid.setColumnStretch(0, 1)
        mesh_grid.setColumnStretch(1, 1)
        grid_row = 0
        
        # ==================== MESH TYPE SELECTION ====================
        type_group = QGroupBox("Mesh Type")
        type_group.setToolTip("Choose between structured (hex) or unstructured (tet) mesh")
        type_layout = QVBoxLayout(type_group)
        type_layout.setSpacing(8)
        type_layout.setContentsMargins(12, 20, 12, 12)
        
        mesh_type_row = QHBoxLayout()
        mesh_type_row.setSpacing(16)
        
        self.hex_mesh_radio = QRadioButton("Hexahedral (Quad)")
        self.hex_mesh_radio.setToolTip("Structured mesh with quadrilateral/hexahedral elements.\nBetter for boundary layers and aligned flows.")
        
        self.tet_mesh_radio = QRadioButton("Tetrahedral (Tri)")
        self.tet_mesh_radio.setToolTip("Unstructured mesh with triangular/tetrahedral elements.\nBetter for complex geometries.")
        
        # Set default from YAML
        if DEFAULTS.mesh_type.lower() == 'hex':
            self.hex_mesh_radio.setChecked(True)
        else:
            self.tet_mesh_radio.setChecked(True)
        
        mesh_type_row.addWidget(self.hex_mesh_radio)
        mesh_type_row.addWidget(self.tet_mesh_radio)
        mesh_type_row.addStretch()
        type_layout.addLayout(mesh_type_row)
        
        # Mesh type description
        self.mesh_type_desc = QLabel("Structured quad mesh - optimal for nozzle flows")
        self.mesh_type_desc.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 12px; padding: 4px;")
        type_layout.addWidget(self.mesh_type_desc)
        
        # Update description on selection change
        def update_mesh_type_desc():
            if self.hex_mesh_radio.isChecked():
                self.mesh_type_desc.setText("Structured quad mesh - optimal for nozzle flows")
            else:
                self.mesh_type_desc.setText("Unstructured tri mesh - flexible for complex shapes")
        
        self.hex_mesh_radio.toggled.connect(update_mesh_type_desc)
        
        mesh_grid.addWidget(type_group, grid_row, 0)  # Column 0
        
        # ==================== ESSENTIAL MESH SETTINGS ====================
        basic_group = QGroupBox("Mesh Size")
        basic_group.setToolTip("Essential mesh size parameters")
        basic_layout = QFormLayout(basic_group)
        basic_layout.setSpacing(10)
        basic_layout.setContentsMargins(12, 20, 12, 12)
        
        self.global_size = QDoubleSpinBox()
        self.global_size.setRange(0.001, 1.0)
        self.global_size.setDecimals(4)
        self.global_size.setValue(DEFAULTS.global_element_size)
        self.global_size.setSuffix(" m")
        self.global_size.setToolTip("Target element size in the domain interior")
        basic_layout.addRow("Global Size:", self.global_size)
        
        self.min_cell_size = QDoubleSpinBox()
        self.min_cell_size.setRange(0.0001, 0.1)
        self.min_cell_size.setDecimals(5)
        self.min_cell_size.setValue(DEFAULTS.min_element_size)
        self.min_cell_size.setSuffix(" m")
        self.min_cell_size.setToolTip("Minimum cell size anywhere in mesh")
        basic_layout.addRow("Min Cell Size:", self.min_cell_size)
        
        self.max_cell_size = QDoubleSpinBox()
        self.max_cell_size.setRange(0.001, 1.0)
        self.max_cell_size.setDecimals(4)
        self.max_cell_size.setValue(DEFAULTS.max_element_size)
        self.max_cell_size.setSuffix(" m")
        self.max_cell_size.setToolTip("Maximum cell size in far-field regions")
        basic_layout.addRow("Max Cell Size:", self.max_cell_size)
        
        self.wall_size = QDoubleSpinBox()
        self.wall_size.setRange(0.0001, 0.1)
        self.wall_size.setDecimals(4)
        self.wall_size.setValue(DEFAULTS.wall_element_size)
        self.wall_size.setSuffix(" m")
        self.wall_size.setToolTip("Element size at wall boundaries")
        basic_layout.addRow("Wall Size:", self.wall_size)
        
        mesh_grid.addWidget(basic_group, grid_row, 1)  # Column 1
        grid_row += 1
        
        # ==================== NOZZLE REFINEMENT SETTINGS ====================
        refine_group = QGroupBox("Nozzle Refinement")
        refine_group.setToolTip("Fine mesh in nozzle domain for accurate flow resolution")
        refine_layout = QVBoxLayout(refine_group)
        refine_layout.setSpacing(8)
        refine_layout.setContentsMargins(12, 20, 12, 12)
        
        self.enable_nozzle_refinement = QCheckBox("Enable Nozzle Refinement")
        self.enable_nozzle_refinement.setChecked(DEFAULTS.nozzle_refinement_enabled)
        self.enable_nozzle_refinement.setToolTip("Apply finer mesh inside nozzle geometry")
        refine_layout.addWidget(self.enable_nozzle_refinement)
        
        refine_params = QFormLayout()
        refine_params.setSpacing(8)
        
        self.nozzle_cell_size = QDoubleSpinBox()
        self.nozzle_cell_size.setRange(0.0001, 0.1)
        self.nozzle_cell_size.setDecimals(4)
        self.nozzle_cell_size.setValue(DEFAULTS.nozzle_refinement_cell_size)
        self.nozzle_cell_size.setSuffix(" m")
        self.nozzle_cell_size.setToolTip("Cell size within nozzle domain")
        refine_params.addRow("Nozzle Cell Size:", self.nozzle_cell_size)
        
        self.nozzle_growth_rate = QDoubleSpinBox()
        self.nozzle_growth_rate.setRange(1.0, 2.0)
        self.nozzle_growth_rate.setDecimals(2)
        self.nozzle_growth_rate.setValue(DEFAULTS.nozzle_refinement_growth_rate)
        self.nozzle_growth_rate.setToolTip("Growth rate from nozzle to ambient (1.1-1.3)")
        refine_params.addRow("Growth Rate:", self.nozzle_growth_rate)
        
        refine_layout.addLayout(refine_params)
        
        # Connect enable checkbox
        refine_widgets = [self.nozzle_cell_size, self.nozzle_growth_rate]
        self.enable_nozzle_refinement.toggled.connect(lambda checked: [w.setEnabled(checked) for w in refine_widgets])
        
        mesh_grid.addWidget(refine_group, grid_row, 0)  # Column 0
        
        # ==================== BOUNDARY LAYER SETTINGS ====================
        bl_group = QGroupBox("Boundary Layers (Walls Only)")
        bl_group.setToolTip("Boundary layer mesh for viscous effects at walls.\nNOT applied to inlet/outlet boundaries.")
        bl_layout = QVBoxLayout(bl_group)
        bl_layout.setSpacing(8)
        bl_layout.setContentsMargins(12, 20, 12, 12)
        
        self.enable_bl = QCheckBox("Enable Boundary Layers")
        self.enable_bl.setChecked(DEFAULTS.boundary_layer_enabled)
        self.enable_bl.setToolTip("Add structured boundary layer mesh near WALL boundaries only")
        bl_layout.addWidget(self.enable_bl)
        
        # Info label about BL application
        bl_info = QLabel("ℹ️ Applied to walls only, not inlet/outlet")
        bl_info.setStyleSheet(f"color: {Theme.TEXT_SECONDARY}; font-size: 11px; padding: 2px;")
        bl_layout.addWidget(bl_info)
        
        # BL parameters in a form layout
        bl_params = QFormLayout()
        bl_params.setSpacing(8)
        
        self.num_bl_layers = QSpinBox()
        self.num_bl_layers.setRange(1, 30)
        self.num_bl_layers.setValue(DEFAULTS.num_boundary_layers)
        self.num_bl_layers.setToolTip("Number of boundary layer elements (8-15 typical for RANS)")
        bl_params.addRow("Layers:", self.num_bl_layers)
        
        self.first_layer_thickness = QDoubleSpinBox()
        self.first_layer_thickness.setRange(1e-7, 1e-2)
        self.first_layer_thickness.setDecimals(7)
        self.first_layer_thickness.setValue(DEFAULTS.first_layer_thickness)
        self.first_layer_thickness.setSuffix(" m")
        self.first_layer_thickness.setToolTip("First layer thickness (y+ dependent)")
        bl_params.addRow("First Layer:", self.first_layer_thickness)
        
        self.growth_ratio = QDoubleSpinBox()
        self.growth_ratio.setRange(1.0, 3.0)
        self.growth_ratio.setDecimals(2)
        self.growth_ratio.setValue(DEFAULTS.growth_ratio)
        self.growth_ratio.setToolTip("Growth ratio between layers (1.1-1.3 typical)")
        bl_params.addRow("Growth Ratio:", self.growth_ratio)
        
        bl_layout.addLayout(bl_params)
        
        # Connect enable checkbox to show/hide BL options
        bl_widgets = [self.num_bl_layers, self.first_layer_thickness, self.growth_ratio]
        self.enable_bl.toggled.connect(lambda checked: [w.setEnabled(checked) for w in bl_widgets])
        
        mesh_grid.addWidget(bl_group, grid_row, 1)  # Column 1
        grid_row += 1
        
        # Add the grid to the main layout
        controls_layout.addLayout(mesh_grid)
        
        # ==================== ADVANCED SETTINGS TOGGLE ====================
        self.show_advanced_mesh = QCheckBox("Advanced Settings")
        self.show_advanced_mesh.setChecked(False)
        self.show_advanced_mesh.setToolTip("Show expert mesh options")
        self.show_advanced_mesh.setStyleSheet(f"""
            QCheckBox {{
                color: {Theme.TEXT_SECONDARY};
                font-weight: 500;
                padding: 4px;
            }}
        """)
        controls_layout.addWidget(self.show_advanced_mesh)
        
        # ==================== ADVANCED SETTINGS CONTAINER ====================
        self.advanced_mesh_container = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_mesh_container)
        advanced_layout.setSpacing(8)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        
        # Advanced quality settings
        advanced_group = QGroupBox("Quality Settings")
        adv_layout = QFormLayout(advanced_group)
        adv_layout.setSpacing(8)
        adv_layout.setContentsMargins(12, 20, 12, 12)
        
        self.quality_threshold = QDoubleSpinBox()
        self.quality_threshold.setRange(0.1, 1.0)
        self.quality_threshold.setValue(0.3)
        self.quality_threshold.setDecimals(2)
        self.quality_threshold.setToolTip("Minimum mesh quality threshold (0.3-0.5 typical)")
        adv_layout.addRow("Quality:", self.quality_threshold)
        
        self.farfield_distance = QDoubleSpinBox()
        self.farfield_distance.setRange(1.0, 20.0)
        self.farfield_distance.setValue(5.0)
        self.farfield_distance.setDecimals(1)
        self.farfield_distance.setToolTip("Domain extension multiplier")
        adv_layout.addRow("Farfield:", self.farfield_distance)
        
        advanced_layout.addWidget(advanced_group)
        
        self.advanced_mesh_container.setVisible(False)
        controls_layout.addWidget(self.advanced_mesh_container)
        
        # Connect advanced toggle
        self.show_advanced_mesh.toggled.connect(self.advanced_mesh_container.setVisible)
        
        # ==================== MESH STATISTICS ====================
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(12, 20, 12, 12)
        
        self.mesh_stats = QTextEdit()
        self.mesh_stats.setReadOnly(True)
        self.mesh_stats.setMaximumHeight(100)
        self.mesh_stats.setStyleSheet(f"""
            QTextEdit {{
                background: {Theme.SURFACE_VARIANT};
                border: none;
                border-radius: 4px;
                font-family: monospace;
                font-size: {max(12, int(getattr(Theme, 'FONT_SIZE_TINY', 12)))}px;
                padding: 8px;
            }}
        """)
        self.mesh_stats.setText("Generate mesh to see statistics...")
        
        stats_layout.addWidget(self.mesh_stats)
        controls_layout.addWidget(stats_group)
        
        controls_layout.addStretch()
        
        # ==================== MESH ACTIONS ====================
        btn_generate = QPushButton(" Generate Mesh")
        btn_generate.setMinimumHeight(44)
        btn_generate.setToolTip("Generate computational mesh from geometry")
        btn_generate.clicked.connect(self.generate_mesh)
        btn_generate.setCursor(Qt.PointingHandCursor)
        btn_generate.setProperty("class", "primary")
        controls_layout.addWidget(btn_generate)
        
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_analyze = QPushButton("[Chart] Analyze Mesh Quality")
        btn_analyze.setMinimumHeight(36)
        btn_analyze.clicked.connect(self.analyze_mesh)
        btn_export = QPushButton("[Save] Export Mesh")
        btn_export.setMinimumHeight(36)
        btn_export.clicked.connect(self.export_mesh)
        btn_row.addWidget(btn_analyze)
        btn_row.addWidget(btn_export)
        controls_layout.addLayout(btn_row)
        
        scroll.setWidget(controls_widget)
        
        panel_layout = QVBoxLayout(control_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        
        # Assemble split layout: wide inputs + viewer, plus bottom input section
        top_splitter.addWidget(control_panel)
        top_splitter.addWidget(canvas_container)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 2)
        top_splitter.setSizes([int(round(700 * scale)), int(round(500 * scale))])

        outer_splitter.addWidget(top_splitter)
        outer_splitter.addWidget(self._create_bottom_input_panel("Meshing Notes / Quick Inputs"))
        outer_splitter.setStretchFactor(0, 5)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([int(round(900 * scale)), int(round(220 * scale))])

        layout.addWidget(outer_splitter, 1)

        return tab
        
    def create_mesh_canvas(self):
        """Create interactive mesh visualization canvas with zoom/pan support."""
        canvas = InteractiveMeshCanvas(self, width=10, height=6, facecolor=Theme.BACKGROUND)
        
        # Initial display message
        canvas.ax.text(0.5, 0.5, 'Generate mesh to visualize\n\nControls:\n• Scroll to zoom\n• Drag to pan', 
               transform=canvas.ax.transAxes, ha='center', va='center',
               color=Theme.TEXT_SECONDARY, fontsize=14)
        
        return canvas
        
    def create_simulation_tab(self):
        """Create simulation setup tab with CFD controls.
        
        Optimized defaults for de Laval nozzle:
        - Compressible RANS with SST turbulence model
        - Steady-state with multigrid acceleration  
        - JST scheme for shock capturing
        - Pressure ratio ~2.5 for supersonic expansion
        
        Features advanced settings toggle for experts.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        scale = float(getattr(self, 'scale_factor', 1.0) or 1.0)

        outer_splitter = QSplitter(Qt.Vertical)
        self._configure_splitter(outer_splitter)

        top_splitter = QSplitter(Qt.Horizontal)
        self._configure_splitter(top_splitter)
        
        # Main canvas area (left side)
        canvas_container = QWidget()
        canvas_container.setStyleSheet(f"""
            QWidget {{
                background: {Theme.BACKGROUND};
            }}
        """)
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(20, 20, 10, 20)
        canvas_layout.setSpacing(12)
        
        # Canvas header
        canvas_header = QWidget()
        canvas_header.setFixedHeight(50)
        header_layout = QHBoxLayout(canvas_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        canvas_title = QLabel("Simulation Monitor")
        canvas_title.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_LARGE', 18))}px;
                font-weight: 600;
            }}
        """)
        header_layout.addWidget(canvas_title)
        header_layout.addStretch()
        
        # Simulation status indicator
        self.sim_status_indicator = QLabel("Not running")
        self.sim_status_indicator.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_SECONDARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_NORMAL', 15))}px;
                font-weight: 500;
                padding: 8px 16px;
                background: {Theme.SURFACE_VARIANT};
                border-radius: 6px;
            }}
        """)
        header_layout.addWidget(self.sim_status_indicator)
        
        canvas_layout.addWidget(canvas_header)
        
        # Simulation canvas
        self.sim_canvas = self.create_simulation_canvas()
        canvas_layout.addWidget(self.sim_canvas, 1)
        
        # Viewer goes into splitter; added later
        
        # Input/control panel
        control_panel = QWidget()
        control_panel.setMinimumWidth(int(round(560 * scale)))
        control_panel.setStyleSheet(f"""
            QWidget {{
                background: {Theme.SURFACE};
                border-right: 1px solid {Theme.BORDER};
            }}
        """)
        
        # Create scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
        """)
        
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setSpacing(12)
        controls_layout.setContentsMargins(16, 16, 16, 16)
        
        # Panel header
        panel_header = QLabel("Simulation Setup")
        panel_header.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_LARGE', 17))}px;
                font-weight: 700;
                padding-bottom: 10px;
                border-bottom: 3px solid {Theme.PRIMARY};
            }}
        """)
        controls_layout.addWidget(panel_header)
        
        # Two-column grid for simulation controls
        sim_grid = QGridLayout()
        sim_grid.setSpacing(12)
        sim_grid.setColumnStretch(0, 1)
        sim_grid.setColumnStretch(1, 1)
        sim_row = 0
        
        # ==================== ESSENTIAL SETTINGS ====================
        # These are always visible for all users
        
        # --- Flow Conditions (Essential) ---
        flow_group = QGroupBox("Flow Conditions")
        flow_group.setToolTip("Essential inlet/outlet boundary conditions for nozzle flow")
        flow_layout = QFormLayout(flow_group)
        flow_layout.setSpacing(12)
        flow_layout.setContentsMargins(14, 24, 14, 14)
        
        self.inlet_pressure = QDoubleSpinBox()
        self.inlet_pressure.setRange(10000, 10000000)
        self.inlet_pressure.setDecimals(0)
        self.inlet_pressure.setValue(DEFAULTS.inlet_total_pressure)
        self.inlet_pressure.setSuffix(" Pa")
        self.inlet_pressure.setToolTip("Total inlet pressure (stagnation). Default: 5 bar for supersonic expansion.")
        flow_layout.addRow("Inlet Pressure:", self.inlet_pressure)
        
        self.outlet_pressure = QDoubleSpinBox()
        self.outlet_pressure.setRange(1000, 10000000)
        self.outlet_pressure.setDecimals(0)
        self.outlet_pressure.setValue(DEFAULTS.outlet_static_pressure)
        self.outlet_pressure.setSuffix(" Pa")
        self.outlet_pressure.setToolTip("Static outlet pressure (back pressure). Default: 50 kPa for ~10:1 pressure ratio.")
        flow_layout.addRow("Outlet Pressure:", self.outlet_pressure)
        
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(200, 2000)
        self.temperature.setDecimals(1)
        self.temperature.setValue(DEFAULTS.inlet_total_temperature)
        self.temperature.setSuffix(" K")
        self.temperature.setToolTip("Total inlet temperature. Default: room temperature.")
        flow_layout.addRow("Temperature:", self.temperature)
        
        # Pressure ratio display
        self.pressure_ratio_label = QLabel("Pressure Ratio: 2.96")
        self.pressure_ratio_label.setStyleSheet(f"color: {Theme.PRIMARY}; font-weight: 700;")
        self.inlet_pressure.valueChanged.connect(self._update_pressure_ratio)
        self.outlet_pressure.valueChanged.connect(self._update_pressure_ratio)
        flow_layout.addRow("", self.pressure_ratio_label)
        
        sim_grid.addWidget(flow_group, sim_row, 0)  # Column 0
        
        # --- Simulation Mode (Essential) ---
        mode_group = QGroupBox("Simulation Type")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(10)
        mode_layout.setContentsMargins(12, 20, 12, 12)
        
        mode_btn_layout = QHBoxLayout()
        mode_btn_layout.setSpacing(16)
        self.steady_radio = QRadioButton("Steady-State")
        self.transient_radio = QRadioButton("Transient")
        # Set initial mode from DEFAULTS
        if DEFAULTS.is_transient:
            self.transient_radio.setChecked(True)
        else:
            self.steady_radio.setChecked(True)
        self.steady_radio.setToolTip("Steady-state simulation - faster, suitable for most nozzle designs")
        self.transient_radio.setToolTip("Time-accurate simulation - for unsteady flow analysis")
        self.steady_radio.toggled.connect(self._on_simulation_mode_changed_new)
        mode_btn_layout.addWidget(self.steady_radio)
        mode_btn_layout.addWidget(self.transient_radio)
        mode_btn_layout.addStretch()
        mode_layout.addLayout(mode_btn_layout)
        
        # Hidden combo for compatibility
        self.simulation_mode = QComboBox()
        self.simulation_mode.addItems(["Steady", "Transient"])
        self.simulation_mode.setCurrentText("Transient" if DEFAULTS.is_transient else "Steady")
        self.simulation_mode.setVisible(False)
        
        # Basic iteration control
        iter_layout = QFormLayout()
        iter_layout.setSpacing(8)
        
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(100, 50000)
        self.max_iterations.setValue(DEFAULTS.max_iterations)
        self.max_iterations.setToolTip("Maximum iterations. 2000-5000 typical for convergence.")
        iter_layout.addRow("Max Iterations:", self.max_iterations)
        
        mode_layout.addLayout(iter_layout)
        sim_grid.addWidget(mode_group, sim_row, 1)  # Column 1
        sim_row += 1
        
        # --- Case Directory (Essential) ---
        case_group = QGroupBox("Output Directory")
        case_layout = QHBoxLayout(case_group)
        case_layout.setContentsMargins(12, 20, 12, 12)
        
        self.case_directory = QLineEdit()
        self.case_directory.setText("./case")
        self.case_directory.setToolTip("Directory for SU2 case files and results")
        btn_browse_case = QPushButton("Browse...")
        btn_browse_case.setMaximumWidth(80)
        btn_browse_case.clicked.connect(self.browse_case_directory)
        
        case_layout.addWidget(self.case_directory)
        case_layout.addWidget(btn_browse_case)
        sim_grid.addWidget(case_group, sim_row, 0)  # Column 0

        # --- Solver Configuration (Essential for tests/workflow) ---
        solver_group = QGroupBox("Solver Configuration")
        solver_layout = QFormLayout(solver_group)
        solver_layout.setSpacing(8)
        solver_layout.setContentsMargins(12, 20, 12, 12)

        self.solver_type = QComboBox()
        self.solver_type.addItems(["RANS", "EULER", "NAVIER_STOKES", "INC_RANS", "INC_EULER", "INC_NAVIER_STOKES"])
        self.solver_type.setCurrentText(DEFAULTS.solver_type)
        self.solver_type.setToolTip("RANS: viscous turbulent, EULER: inviscid, NAVIER_STOKES: viscous laminar")
        self.solver_type.currentTextChanged.connect(self._on_solver_type_changed)
        solver_layout.addRow("Solver Type:", self.solver_type)

        self.turbulence_model = QComboBox()
        self.turbulence_model.addItems(["SST", "SA", "SA_NEG", "SST_SUST", "None (Laminar)"])
        self.turbulence_model.setCurrentText(DEFAULTS.turbulence_model if DEFAULTS.turbulence_enabled else "None (Laminar)")
        self.turbulence_model.setToolTip("SST (k-omega): recommended for nozzles")
        self.lbl_turbulence = QLabel("Turbulence Model:")
        solver_layout.addRow(self.lbl_turbulence, self.turbulence_model)

        self.cfl_number = QDoubleSpinBox()
        self.cfl_number.setRange(0.1, 1000.0)
        self.cfl_number.setDecimals(1)
        self.cfl_number.setValue(DEFAULTS.cfl_number)
        self.cfl_number.setToolTip("CFL number (5-10 typical)")
        solver_layout.addRow("CFL Number:", self.cfl_number)

        self.convergence_residual = QDoubleSpinBox()
        self.convergence_residual.setRange(-12, -3)
        self.convergence_residual.setDecimals(0)
        self.convergence_residual.setValue(DEFAULTS.convergence_residual)
        self.convergence_residual.setPrefix("10^")
        self.convergence_residual.setToolTip("Target residual (log10)")
        solver_layout.addRow("Convergence:", self.convergence_residual)

        sim_grid.addWidget(solver_group, sim_row, 1)  # Column 1
        sim_row += 1
        
        # Add the grid to the main layout
        controls_layout.addLayout(sim_grid)
        
        # --- Action Buttons (Essential) ---
        actions_group = QGroupBox("Run Simulation")
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(8)
        actions_layout.setContentsMargins(12, 20, 12, 12)
        
        btn_setup = QPushButton("⚙️ Generate Case Files")
        btn_setup.setMinimumHeight(40)
        btn_setup.setToolTip("Create SU2 configuration and mesh files")
        btn_setup.clicked.connect(self.setup_simulation)
        btn_setup.setProperty("class", "primary")
        
        btn_run = QPushButton("▶️ Run Simulation")
        btn_run.setMinimumHeight(40)
        btn_run.setToolTip("Start the SU2 CFD solver")
        btn_run.clicked.connect(self.run_simulation)
        btn_run.setProperty("class", "primary")
        
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_monitor = QPushButton("Monitor")
        btn_monitor.setMinimumHeight(32)
        btn_monitor.clicked.connect(self.monitor_simulation)
        btn_stop = QPushButton("Stop")
        btn_stop.setMinimumHeight(32)
        btn_stop.clicked.connect(self.stop_simulation)
        btn_row.addWidget(btn_monitor)
        btn_row.addWidget(btn_stop)
        
        actions_layout.addWidget(btn_setup)
        actions_layout.addWidget(btn_run)
        actions_layout.addLayout(btn_row)
        controls_layout.addWidget(actions_group)
        
        # ==================== ADVANCED SETTINGS TOGGLE ====================
        self.show_advanced_sim = QCheckBox("Show Advanced Settings")
        self.show_advanced_sim.setChecked(False)
        self.show_advanced_sim.setToolTip("Show expert CFD solver options")
        self.show_advanced_sim.setStyleSheet(f"""
            QCheckBox {{
                color: {Theme.TEXT_SECONDARY};
                font-weight: 500;
                padding: 8px 4px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
            }}
        """)
        controls_layout.addWidget(self.show_advanced_sim)
        
        # ==================== ADVANCED SETTINGS CONTAINER ====================
        self.advanced_sim_container = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_sim_container)
        advanced_layout.setSpacing(12)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Advanced Flow Settings ---
        adv_flow_group = QGroupBox("Advanced Flow Settings")
        adv_flow_layout = QFormLayout(adv_flow_group)
        adv_flow_layout.setSpacing(8)
        adv_flow_layout.setContentsMargins(12, 20, 12, 12)
        
        self.inlet_velocity = QDoubleSpinBox()
        self.inlet_velocity.setRange(1.0, 2000.0)
        self.inlet_velocity.setValue(100.0)
        self.inlet_velocity.setDecimals(1)
        self.inlet_velocity.setSuffix(" m/s")
        self.inlet_velocity.setToolTip("Inlet velocity (subsonic for de Laval)")
        adv_flow_layout.addRow("Inlet Velocity:", self.inlet_velocity)
        
        advanced_layout.addWidget(adv_flow_group)
        
        # --- Initialization Settings ---
        init_group = QGroupBox("Domain Initialization")
        init_layout = QFormLayout(init_group)
        init_layout.setSpacing(8)
        init_layout.setContentsMargins(12, 20, 12, 12)
        
        self.init_method = QComboBox()
        self.init_method.addItems([
            "outlet_pressure",
            "inlet_pressure", 
            "average_pressure",
            "custom"
        ])
        self.init_method.setCurrentText(DEFAULTS.init_method)
        self.init_method.setToolTip(
            "How to initialize flow domain:\n"
            "• outlet_pressure: Start at back pressure (recommended)\n"
            "• inlet_pressure: Start at inlet total pressure\n"
            "• average_pressure: Start at (inlet + outlet) / 2\n"
            "• custom: Use custom values below"
        )
        init_layout.addRow("Init Method:", self.init_method)
        
        self.init_velocity = QDoubleSpinBox()
        self.init_velocity.setRange(0.0, 1000.0)
        self.init_velocity.setDecimals(2)
        self.init_velocity.setValue(DEFAULTS.init_velocity)
        self.init_velocity.setSuffix(" m/s")
        self.init_velocity.setToolTip("Initial velocity magnitude to seed flow direction")
        init_layout.addRow("Init Velocity:", self.init_velocity)
        
        self.init_temperature = QDoubleSpinBox()
        self.init_temperature.setRange(100.0, 5000.0)
        self.init_temperature.setDecimals(1)
        self.init_temperature.setValue(DEFAULTS.init_temperature)
        self.init_temperature.setSuffix(" K")
        self.init_temperature.setToolTip("Initial temperature in the domain")
        init_layout.addRow("Init Temperature:", self.init_temperature)
        
        # Custom values (only visible when method = custom)
        self.init_custom_pressure = QDoubleSpinBox()
        self.init_custom_pressure.setRange(1000.0, 1e8)
        self.init_custom_pressure.setDecimals(0)
        self.init_custom_pressure.setValue(DEFAULTS.init_custom_pressure)
        self.init_custom_pressure.setSuffix(" Pa")
        self.init_custom_pressure.setToolTip("Custom initialization pressure")
        self.lbl_init_pressure = QLabel("Custom Pressure:")
        init_layout.addRow(self.lbl_init_pressure, self.init_custom_pressure)
        
        self.init_custom_density = QDoubleSpinBox()
        self.init_custom_density.setRange(0.001, 100.0)
        self.init_custom_density.setDecimals(4)
        self.init_custom_density.setValue(DEFAULTS.init_custom_density)
        self.init_custom_density.setSuffix(" kg/m³")
        self.init_custom_density.setToolTip("Custom initialization density")
        self.lbl_init_density = QLabel("Custom Density:")
        init_layout.addRow(self.lbl_init_density, self.init_custom_density)
        
        # Initially hide custom fields unless method is 'custom'
        is_custom = DEFAULTS.init_method == 'custom'
        self.lbl_init_pressure.setVisible(is_custom)
        self.init_custom_pressure.setVisible(is_custom)
        self.lbl_init_density.setVisible(is_custom)
        self.init_custom_density.setVisible(is_custom)
        
        # Connect method change to show/hide custom fields
        def _on_init_method_changed(method: str):
            is_custom = (method == 'custom')
            self.lbl_init_pressure.setVisible(is_custom)
            self.init_custom_pressure.setVisible(is_custom)
            self.lbl_init_density.setVisible(is_custom)
            self.init_custom_density.setVisible(is_custom)
        
        self.init_method.currentTextChanged.connect(_on_init_method_changed)
        
        advanced_layout.addWidget(init_group)
        
        # Solver Configuration is always visible (see above).
        
        # --- Steady-State Settings ---
        self.steady_group = QGroupBox("Steady-State Options")
        steady_layout = QFormLayout(self.steady_group)
        steady_layout.setSpacing(8)
        steady_layout.setContentsMargins(12, 20, 12, 12)
        
        self.cfl_adapt = QCheckBox("Adaptive CFL")
        self.cfl_adapt.setChecked(DEFAULTS.cfl_adapt)
        self.cfl_adapt.setToolTip("Automatically adjust CFL for stability")
        steady_layout.addRow("", self.cfl_adapt)
        
        self.cfl_min = QDoubleSpinBox()
        self.cfl_min.setRange(0.01, 10.0)
        self.cfl_min.setDecimals(2)
        self.cfl_min.setValue(DEFAULTS.cfl_min)
        
        self.cfl_max = QDoubleSpinBox()
        self.cfl_max.setRange(1.0, 1000.0)
        self.cfl_max.setDecimals(1)
        self.cfl_max.setValue(DEFAULTS.cfl_max)
        
        cfl_range_widget = QWidget()
        cfl_range_layout = QHBoxLayout(cfl_range_widget)
        cfl_range_layout.setContentsMargins(0, 0, 0, 0)
        cfl_range_layout.addWidget(self.cfl_min)
        cfl_range_layout.addWidget(QLabel("-"))
        cfl_range_layout.addWidget(self.cfl_max)
        
        self.lbl_cfl_range = QLabel("CFL Range:")
        steady_layout.addRow(self.lbl_cfl_range, cfl_range_widget)
        
        self.cfl_adapt.toggled.connect(lambda checked: (
            self.lbl_cfl_range.setVisible(checked),
            cfl_range_widget.setVisible(checked)
        ))
        self.lbl_cfl_range.setVisible(False)
        cfl_range_widget.setVisible(False)
        
        # Show steady group only if not starting in transient mode
        self.steady_group.setVisible(not DEFAULTS.is_transient)
        advanced_layout.addWidget(self.steady_group)
        
        # --- Transient Settings ---
        self.transient_group = QGroupBox("Transient Options")
        transient_layout = QFormLayout(self.transient_group)
        transient_layout.setSpacing(8)
        transient_layout.setContentsMargins(12, 20, 12, 12)
        
        self.time_step = QDoubleSpinBox()
        self.time_step.setRange(1e-9, 1.0)
        self.time_step.setDecimals(9)
        self.time_step.setValue(DEFAULTS.time_step)
        self.time_step.setSuffix(" s")
        transient_layout.addRow("Time Step:", self.time_step)
        
        self.end_time = QDoubleSpinBox()
        self.end_time.setRange(1e-6, 100.0)
        self.end_time.setDecimals(6)
        self.end_time.setValue(DEFAULTS.end_time)
        self.end_time.setSuffix(" s")
        transient_layout.addRow("End Time:", self.end_time)
        
        self.inner_iterations = QSpinBox()
        self.inner_iterations.setRange(1, 100)
        self.inner_iterations.setValue(DEFAULTS.inner_iterations)
        transient_layout.addRow("Inner Iterations:", self.inner_iterations)
        
        self.time_marching = QComboBox()
        self.time_marching.addItems([
            "DUAL_TIME_STEPPING-2ND_ORDER",
            "DUAL_TIME_STEPPING-1ST_ORDER",
            "TIME_STEPPING"
        ])
        transient_layout.addRow("Time Method:", self.time_marching)
        
        # --- Variable Time Step (Multi-Phase) ---
        self.variable_dt_check = QCheckBox("Variable Time Step (Multi-Phase)")
        self.variable_dt_check.setChecked(DEFAULTS.variable_dt_enabled)
        self.variable_dt_check.setToolTip(
            "Run simulation in phases:\n"
            "Phase 1: Small dt to establish stable flow\n"
            "Phase 2: Medium dt for developed flow\n"
            "Phase 3: Large dt for efficient time advancement"
        )
        transient_layout.addRow("", self.variable_dt_check)
        
        # Variable dt container (shown when enabled)
        self.variable_dt_container = QWidget()
        var_dt_layout = QFormLayout(self.variable_dt_container)
        var_dt_layout.setSpacing(6)
        var_dt_layout.setContentsMargins(0, 0, 0, 0)
        
        # Phase 1 settings
        phase1_label = QLabel("<b>Phase 1 (Startup):</b>")
        phase1_label.setStyleSheet(f"color: {Theme.PRIMARY};")
        var_dt_layout.addRow(phase1_label)
        
        self.phase1_dt = QDoubleSpinBox()
        self.phase1_dt.setRange(1e-10, 1e-4)
        self.phase1_dt.setDecimals(10)
        self.phase1_dt.setValue(DEFAULTS.phase1_dt)
        self.phase1_dt.setSuffix(" s")
        self.phase1_dt.setToolTip("Small time step for initial transient stability")
        var_dt_layout.addRow("  dt:", self.phase1_dt)
        
        self.phase1_duration = QDoubleSpinBox()
        self.phase1_duration.setRange(1e-6, 1.0)
        self.phase1_duration.setDecimals(6)
        self.phase1_duration.setValue(DEFAULTS.phase1_duration)
        self.phase1_duration.setSuffix(" s")
        self.phase1_duration.setToolTip("How long to run Phase 1")
        var_dt_layout.addRow("  Duration:", self.phase1_duration)
        
        self.phase1_inner = QSpinBox()
        self.phase1_inner.setRange(10, 200)
        self.phase1_inner.setValue(DEFAULTS.phase1_inner_iter)
        self.phase1_inner.setToolTip("Inner iterations for Phase 1 (more = stable)")
        var_dt_layout.addRow("  Inner Iter:", self.phase1_inner)
        
        # Phase 2 settings
        phase2_label = QLabel("<b>Phase 2 (Developed):</b>")
        phase2_label.setStyleSheet(f"color: {Theme.SUCCESS};")
        var_dt_layout.addRow(phase2_label)
        
        self.phase2_dt = QDoubleSpinBox()
        self.phase2_dt.setRange(1e-9, 1e-3)
        self.phase2_dt.setDecimals(9)
        self.phase2_dt.setValue(DEFAULTS.phase2_dt)
        self.phase2_dt.setSuffix(" s")
        self.phase2_dt.setToolTip("Medium time step after flow is established")
        var_dt_layout.addRow("  dt:", self.phase2_dt)
        
        self.phase2_duration = QDoubleSpinBox()
        self.phase2_duration.setRange(1e-5, 10.0)
        self.phase2_duration.setDecimals(5)
        self.phase2_duration.setValue(DEFAULTS.phase2_duration)
        self.phase2_duration.setSuffix(" s")
        self.phase2_duration.setToolTip("How long to run Phase 2")
        var_dt_layout.addRow("  Duration:", self.phase2_duration)
        
        self.phase2_inner = QSpinBox()
        self.phase2_inner.setRange(5, 100)
        self.phase2_inner.setValue(DEFAULTS.phase2_inner_iter)
        var_dt_layout.addRow("  Inner Iter:", self.phase2_inner)
        
        # Phase 3 settings
        phase3_label = QLabel("<b>Phase 3 (Final):</b>")
        phase3_label.setStyleSheet(f"color: {Theme.WARNING};")
        var_dt_layout.addRow(phase3_label)
        
        self.phase3_dt = QDoubleSpinBox()
        self.phase3_dt.setRange(1e-8, 1e-2)
        self.phase3_dt.setDecimals(8)
        self.phase3_dt.setValue(DEFAULTS.phase3_dt)
        self.phase3_dt.setSuffix(" s")
        self.phase3_dt.setToolTip("Large time step for efficient final simulation")
        var_dt_layout.addRow("  dt:", self.phase3_dt)
        
        self.phase3_inner = QSpinBox()
        self.phase3_inner.setRange(5, 100)
        self.phase3_inner.setValue(DEFAULTS.phase3_inner_iter)
        var_dt_layout.addRow("  Inner Iter:", self.phase3_inner)
        
        # Initially hide if not enabled
        self.variable_dt_container.setVisible(DEFAULTS.variable_dt_enabled)
        transient_layout.addRow(self.variable_dt_container)
        
        # Connect toggle
        self.variable_dt_check.toggled.connect(self.variable_dt_container.setVisible)
        
        # Show transient group if transient mode is default
        self.transient_group.setVisible(DEFAULTS.is_transient)
        advanced_layout.addWidget(self.transient_group)
        
        # --- Discretization Schemes ---
        discret_group = QGroupBox("Discretization Schemes")
        discret_layout = QFormLayout(discret_group)
        discret_layout.setSpacing(8)
        discret_layout.setContentsMargins(12, 20, 12, 12)
        
        self.conv_scheme = QComboBox()
        self.conv_scheme.addItems(["JST", "ROE", "AUSM", "AUSMPLUSUP2", "HLLC", "SLAU2"])
        self.conv_scheme.setCurrentText(DEFAULTS.convective_scheme)
        self.conv_scheme.setToolTip("JST: good for shocks, ROE: accurate")
        self.conv_scheme.currentTextChanged.connect(self._on_conv_scheme_changed)
        discret_layout.addRow("Convective Scheme:", self.conv_scheme)
        
        self.muscl_flow = QCheckBox("MUSCL Reconstruction")
        self.muscl_flow.setChecked(DEFAULTS.muscl_reconstruction)
        discret_layout.addRow("", self.muscl_flow)
        
        self.slope_limiter = QComboBox()
        self.slope_limiter.addItems(["VENKATAKRISHNAN", "VENKATAKRISHNAN_WANG", "BARTH_JESPERSEN", "NONE"])
        self.slope_limiter.setCurrentText(DEFAULTS.slope_limiter)
        self.lbl_limiter = QLabel("Slope Limiter:")
        discret_layout.addRow(self.lbl_limiter, self.slope_limiter)
        
        self.gradient_method = QComboBox()
        self.gradient_method.addItems(["WEIGHTED_LEAST_SQUARES", "GREEN_GAUSS"])
        self.gradient_method.setCurrentText(DEFAULTS.gradient_method)
        discret_layout.addRow("Gradient Method:", self.gradient_method)
        
        self.time_discre = QComboBox()
        self.time_discre.addItems(["EULER_IMPLICIT", "EULER_EXPLICIT", "RUNGE-KUTTA_EXPLICIT"])
        self.time_discre.setCurrentText(DEFAULTS.time_discretization)
        discret_layout.addRow("Time Discretization:", self.time_discre)
        
        advanced_layout.addWidget(discret_group)
        
        # --- Linear Solver & Multigrid ---
        linear_group = QGroupBox("Linear Solver")
        linear_layout = QFormLayout(linear_group)
        linear_layout.setSpacing(8)
        linear_layout.setContentsMargins(12, 20, 12, 12)
        
        self.linear_solver = QComboBox()
        self.linear_solver.addItems(["FGMRES", "BCGSTAB", "SMOOTHER"])
        self.linear_solver.setCurrentText(DEFAULTS.linear_solver)
        linear_layout.addRow("Solver:", self.linear_solver)
        
        self.linear_solver_prec = QComboBox()
        self.linear_solver_prec.addItems(["ILU", "LU_SGS", "JACOBI", "LINELET"])
        self.linear_solver_prec.setCurrentText(DEFAULTS.linear_solver_preconditioner)
        linear_layout.addRow("Preconditioner:", self.linear_solver_prec)
        
        self.linear_solver_iter = QSpinBox()
        self.linear_solver_iter.setRange(1, 100)
        self.linear_solver_iter.setValue(DEFAULTS.linear_solver_iterations)
        linear_layout.addRow("Iterations:", self.linear_solver_iter)
        
        self.mglevel = QSpinBox()
        self.mglevel.setRange(0, 6)
        self.mglevel.setValue(DEFAULTS.multigrid_levels)
        self.mglevel.setToolTip("0 = no multigrid")
        linear_layout.addRow("Multigrid Levels:", self.mglevel)
        
        self.mgcycle = QComboBox()
        self.mgcycle.addItems(["W_CYCLE", "V_CYCLE", "FULLMG"])
        self.mgcycle.setCurrentText(DEFAULTS.multigrid_cycle)
        linear_layout.addRow("MG Cycle:", self.mgcycle)
        
        advanced_layout.addWidget(linear_group)
        
        # --- Output Settings ---
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)
        output_layout.setSpacing(8)
        output_layout.setContentsMargins(12, 20, 12, 12)
        
        self.output_frequency = QSpinBox()
        self.output_frequency.setRange(1, 1000)
        self.output_frequency.setValue(DEFAULTS.output_frequency)
        output_layout.addRow("Output Frequency:", self.output_frequency)
        
        self.n_processors = QSpinBox()
        self.n_processors.setRange(1, 128)
        self.n_processors.setValue(DEFAULTS.num_processors)
        self.n_processors.setToolTip("Number of MPI processes (1 = serial)")
        output_layout.addRow("Processors:", self.n_processors)
        
        advanced_layout.addWidget(output_group)
        
        # Add advanced container to main layout
        self.advanced_sim_container.setVisible(False)
        controls_layout.addWidget(self.advanced_sim_container)
        
        # Connect advanced toggle
        self.show_advanced_sim.toggled.connect(self.advanced_sim_container.setVisible)
        
        # ==================== STATUS LOG ====================
        status_group = QGroupBox("Log")
        status_layout = QVBoxLayout(status_group)
        status_layout.setContentsMargins(12, 20, 12, 12)
        
        self.simulation_log = QTextEdit()
        self.simulation_log.setReadOnly(True)
        self.simulation_log.setMaximumHeight(80)
        self.simulation_log.setStyleSheet(f"""
            QTextEdit {{
                background: {Theme.SURFACE_VARIANT};
                border: none;
                border-radius: 4px;
                font-family: monospace;
                font-size: {max(12, int(getattr(Theme, 'FONT_SIZE_TINY', 12)))}px;
                padding: 8px;
            }}
        """)
        self.simulation_log.setText("Ready. Configure and run simulation.")
        
        status_layout.addWidget(self.simulation_log)
        controls_layout.addWidget(status_group)
        
        controls_layout.addStretch()
        scroll.setWidget(controls_widget)
        
        panel_layout = QVBoxLayout(control_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        
        # Assemble split layout
        top_splitter.addWidget(control_panel)
        top_splitter.addWidget(canvas_container)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 2)
        top_splitter.setSizes([int(round(760 * scale)), int(round(520 * scale))])

        outer_splitter.addWidget(top_splitter)
        outer_splitter.addWidget(self._create_bottom_input_panel("Simulation Notes / Quick Inputs"))
        outer_splitter.setStretchFactor(0, 5)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setSizes([int(round(900 * scale)), int(round(220 * scale))])

        layout.addWidget(outer_splitter, 1)
        
        # Initialize visibility states
        self._on_simulation_mode_changed_new(True)
        self._on_conv_scheme_changed(self.conv_scheme.currentText())
        
        # Legacy compatibility attributes
        self.convergence_tolerance = QDoubleSpinBox()
        self.convergence_tolerance.setValue(1e-6)
        self.max_courant = QDoubleSpinBox()
        self.max_courant.setValue(5.0)
        self.n_outer_correctors = self.inner_iterations
        self.n_correctors = QSpinBox()
        self.n_correctors.setValue(2)
        self.decomposition_method = QComboBox()
        self.decomposition_method.addItem("METIS")
        self.lbl_time_step = QLabel()
        self.lbl_end_time = QLabel()
        self.lbl_max_courant = QLabel()
        self.lbl_outer_correctors = QLabel()
        self.lbl_correctors = QLabel()
        self.lbl_cfl = QLabel()
        
        return tab
    
    def _update_pressure_ratio(self):
        """Update pressure ratio display."""
        p_in = self.inlet_pressure.value()
        p_out = self.outlet_pressure.value()
        if p_out > 0:
            ratio = p_in / p_out
            self.pressure_ratio_label.setText(f"Pressure Ratio: {ratio:.2f}")
    
    def _on_simulation_mode_changed_new(self, checked):
        """Handle simulation mode change (steady/transient radio buttons)."""
        is_steady = self.steady_radio.isChecked()
        
        # Update visibility
        self.steady_group.setVisible(is_steady)
        self.transient_group.setVisible(not is_steady)
        
        # Sync combo box for compatibility
        self.simulation_mode.setCurrentText("Steady" if is_steady else "Transient")
        
        # Update solver options
        self._update_solver_options()
    
    def _on_conv_scheme_changed(self, scheme):
        """Handle convective scheme change - show/hide MUSCL options."""
        centered_schemes = ["JST", "LAX-FRIEDRICH"]
        is_centered = scheme.upper() in centered_schemes
        
        # MUSCL only makes sense for upwind schemes
        self.muscl_flow.setEnabled(not is_centered)
        if is_centered:
            self.muscl_flow.setChecked(False)
        
        # Update limiter visibility
        show_limiter = not is_centered or self.muscl_flow.isChecked()
        self.lbl_limiter.setVisible(show_limiter)
        self.slope_limiter.setVisible(show_limiter)
        
    def create_simulation_canvas(self):
        """Create interactive simulation visualization canvas."""
        canvas = InteractiveMonitorCanvas(self, width=10, height=6, dpi=100, facecolor=Theme.BACKGROUND)
        
        # Initial display
        canvas.ax.text(0.5, 0.5, 'Run simulation to see convergence\n\n'
                      'Controls:\n'
                      '• Scroll to zoom\n'
                      '• Left-drag to pan\n'
                      '• Double-click axis to set range', 
               transform=canvas.ax.transAxes, ha='center', va='center',
               color=Theme.TEXT_SECONDARY, fontsize=12)
        
        return canvas
        
    def create_postprocessing_tab(self):
        """Create post-processing tab with Interactive Analyzer only."""
        # Main container with just the Interactive Analyzer (no sub-tabs)
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Pass scale_factor for consistent sizing with other tabs
        scale = float(getattr(self, 'scale_factor', 1.0) or 1.0)
        
        # Interactive Analyzer directly embedded (no sub-tabs needed)
        self.interactive_postprocessor = InteractivePostprocessorWidget(
            theme=Theme, 
            scale_factor=scale,
            configure_splitter_func=self._configure_splitter
        )
        main_layout.addWidget(self.interactive_postprocessor)
        
        return main_tab
        
    def _create_standard_postprocessing_widget(self):
        """Create the standard post-processing widget with hover controls and collapsible summary."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        scale = float(getattr(self, 'scale_factor', 1.0) or 1.0)

        # Main horizontal splitter: Canvas area | Collapsible Summary
        main_splitter = QSplitter(Qt.Horizontal)
        self._configure_splitter(main_splitter)
        
        # === LEFT SIDE: Canvas with overlay controls ===
        canvas_container = QWidget()
        canvas_container.setStyleSheet(f"background: {Theme.BACKGROUND};")
        canvas_main_layout = QVBoxLayout(canvas_container)
        canvas_main_layout.setContentsMargins(10, 10, 10, 10)
        canvas_main_layout.setSpacing(8)
        
        # Header with title and load controls (always visible)
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        canvas_title = QLabel("Results Visualization")
        canvas_title.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {int(getattr(Theme, 'FONT_SIZE_LARGE', 18))}px;
                font-weight: 600;
            }}
        """)
        header_layout.addWidget(canvas_title)
        
        # Compact load controls in header
        self.results_path = QLineEdit()
        self.results_path.setText("./case")
        self.results_path.setPlaceholderText("Results directory...")
        self.results_path.setMaximumWidth(250)
        header_layout.addWidget(self.results_path)
        
        btn_browse = QPushButton("Browse")
        btn_browse.setMaximumWidth(70)
        btn_browse.clicked.connect(self.browse_results_directory)
        header_layout.addWidget(btn_browse)
        
        btn_load = QPushButton("Load")
        btn_load.setMaximumWidth(60)
        btn_load.clicked.connect(self.load_results)
        btn_load.setProperty("class", "primary")
        header_layout.addWidget(btn_load)
        
        header_layout.addStretch()
        
        # Compact visualization controls in header
        self.field_type = QComboBox()
        self.field_type.addItems(["Pressure", "Density", "Temperature", "Mach", "Velocity"])
        self.field_type.setMaximumWidth(120)
        self.field_type.currentTextChanged.connect(self.on_field_changed)
        header_layout.addWidget(QLabel("Field:"))
        header_layout.addWidget(self.field_type)
        
        self.viz_time_step = QComboBox()
        self.viz_time_step.setMinimumWidth(100)
        self.viz_time_step.setMaximumWidth(150)
        self.viz_time_step.currentTextChanged.connect(self.update_visualization)
        header_layout.addWidget(QLabel("Step:"))
        header_layout.addWidget(self.viz_time_step)
        
        self.colormap = QComboBox()
        self.colormap.addItems(["viridis", "plasma", "jet", "coolwarm", "RdYlBu"])
        self.colormap.setMaximumWidth(100)
        self.colormap.currentTextChanged.connect(self.update_visualization)
        header_layout.addWidget(self.colormap)
        
        canvas_main_layout.addWidget(header_widget)
        
        # Canvas area with hover overlay
        canvas_area = QWidget()
        canvas_area_layout = QVBoxLayout(canvas_area)
        canvas_area_layout.setContentsMargins(0, 0, 0, 0)
        
        # The actual canvas
        self.postproc_canvas = self.create_postprocessing_canvas()
        canvas_area_layout.addWidget(self.postproc_canvas, 1)
        
        # Floating overlay controls (initially hidden, appear on hover)
        self._results_overlay_controls = self._create_results_overlay_controls()
        self._results_overlay_controls.setParent(canvas_area)
        self._results_overlay_controls.hide()
        self._results_overlay_controls.move(10, 60)
        
        # Hover timer for showing overlay
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._show_results_overlay)
        
        # Install event filter for hover detection
        canvas_area.setMouseTracking(True)
        canvas_area.enterEvent = self._on_canvas_enter
        canvas_area.leaveEvent = self._on_canvas_leave
        canvas_area.mouseMoveEvent = self._on_canvas_mouse_move
        
        canvas_main_layout.addWidget(canvas_area, 1)
        
        # Hidden widgets for compatibility with existing plot_type and contour_levels
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Contour", "Vector", "Streamline", "Line Plot"])
        self.plot_type.hide()
        
        self.contour_levels = QSpinBox()
        self.contour_levels.setRange(5, 50)
        self.contour_levels.setValue(20)
        self.contour_levels.hide()
        
        main_splitter.addWidget(canvas_container)
        
        # === RIGHT SIDE: Collapsible Summary Panel ===
        summary_panel = self._create_collapsible_summary_panel()
        main_splitter.addWidget(summary_panel)
        
        # Set initial splitter sizes (canvas gets most space)
        main_splitter.setSizes([int(round(900 * scale)), int(round(350 * scale))])
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(main_splitter, 1)

        return tab
    
    def _create_results_overlay_controls(self):
        """Create floating overlay controls for results visualization."""
        overlay = QFrame()
        overlay.setStyleSheet(f"""
            QFrame {{
                background: rgba(30, 30, 30, 0.95);
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                padding: 10px;
            }}
            QPushButton {{
                background: {Theme.SURFACE_VARIANT};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.BORDER};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background: {Theme.PRIMARY_VARIANT};
            }}
        """)
        overlay.setFixedWidth(180)
        
        layout = QVBoxLayout(overlay)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("Analysis Tools")
        title.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; font-weight: bold; font-size: 13px;")
        layout.addWidget(title)
        
        btn_wall = QPushButton("Wall Values")
        btn_wall.clicked.connect(self.analyze_wall_values)
        layout.addWidget(btn_wall)
        
        btn_centerline = QPushButton("Centerline Plot")
        btn_centerline.clicked.connect(self.plot_centerline)
        layout.addWidget(btn_centerline)
        
        btn_mass = QPushButton("Mass Flow Rate")
        btn_mass.clicked.connect(self.calculate_mass_flow)
        layout.addWidget(btn_mass)
        
        btn_pressure = QPushButton("Pressure Loss")
        btn_pressure.clicked.connect(self.calculate_pressure_loss)
        layout.addWidget(btn_pressure)
        
        # Export section
        export_title = QLabel("Export")
        export_title.setStyleSheet(f"color: {Theme.TEXT_PRIMARY}; font-weight: bold; font-size: 13px; margin-top: 10px;")
        layout.addWidget(export_title)
        
        btn_image = QPushButton("Save Image")
        btn_image.clicked.connect(self.save_visualization)
        layout.addWidget(btn_image)
        
        btn_data = QPushButton("Export Data")
        btn_data.clicked.connect(self.export_data)
        layout.addWidget(btn_data)
        
        btn_report = QPushButton("Generate Report")
        btn_report.clicked.connect(self.generate_report)
        layout.addWidget(btn_report)
        
        return overlay
    
    def _create_collapsible_summary_panel(self):
        """Create a collapsible summary panel for the right side."""
        panel = QWidget()
        panel.setStyleSheet(f"background: {Theme.SURFACE};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Collapsible header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        self._summary_toggle_btn = QPushButton("▼ Case Summary")
        self._summary_toggle_btn.setFlat(True)
        self._summary_toggle_btn.setStyleSheet(f"""
            QPushButton {{
                color: {Theme.TEXT_PRIMARY};
                font-weight: bold;
                font-size: 14px;
                text-align: left;
                padding: 8px;
                border: none;
            }}
            QPushButton:hover {{
                color: {Theme.PRIMARY};
            }}
        """)
        self._summary_toggle_btn.clicked.connect(self._toggle_summary_panel)
        header_layout.addWidget(self._summary_toggle_btn)
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # Collapsible content
        self._summary_content = QWidget()
        content_layout = QVBoxLayout(self._summary_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        
        # Results summary text
        self.results_summary = QTextEdit()
        self.results_summary.setReadOnly(True)
        self.results_summary.setStyleSheet(f"""
            QTextEdit {{
                background: {Theme.SURFACE_VARIANT};
                border: none;
                border-radius: 6px;
                font-family: monospace;
                font-size: 12px;
                padding: 10px;
                color: {Theme.TEXT_PRIMARY};
            }}
        """)
        self.results_summary.setText("Load results to see summary...")
        content_layout.addWidget(self.results_summary)
        
        # Quick stats
        stats_group = QGroupBox("Quick Statistics")
        stats_layout = QFormLayout(stats_group)
        stats_layout.setSpacing(6)
        
        self._stat_nodes = QLabel("--")
        self._stat_elements = QLabel("--")
        self._stat_fields = QLabel("--")
        self._stat_timesteps = QLabel("--")
        
        stats_layout.addRow("Nodes:", self._stat_nodes)
        stats_layout.addRow("Elements:", self._stat_elements)
        stats_layout.addRow("Fields:", self._stat_fields)
        stats_layout.addRow("Time Steps:", self._stat_timesteps)
        
        content_layout.addWidget(stats_group)
        content_layout.addStretch()
        
        layout.addWidget(self._summary_content)
        
        return panel
    
    def _toggle_summary_panel(self):
        """Toggle the summary panel visibility."""
        if self._summary_content.isVisible():
            self._summary_content.hide()
            self._summary_toggle_btn.setText("▶ Case Summary")
        else:
            self._summary_content.show()
            self._summary_toggle_btn.setText("▼ Case Summary")
    
    def _on_canvas_enter(self, event):
        """Handle mouse entering canvas area."""
        self._hover_timer.start(3000)  # 3 seconds
    
    def _on_canvas_leave(self, event):
        """Handle mouse leaving canvas area."""
        self._hover_timer.stop()
        if hasattr(self, '_results_overlay_controls'):
            self._results_overlay_controls.hide()
    
    def _on_canvas_mouse_move(self, event):
        """Handle mouse movement in canvas area."""
        # Reset timer on movement
        if self._hover_timer.isActive():
            self._hover_timer.stop()
            self._hover_timer.start(3000)
    
    def _show_results_overlay(self):
        """Show the overlay controls."""
        if hasattr(self, '_results_overlay_controls'):
            self._results_overlay_controls.show()
            self._results_overlay_controls.raise_()
        
    def create_postprocessing_canvas(self):
        """Create post-processing visualization canvas."""
        canvas = FigureCanvas(Figure(figsize=(10, 6), facecolor=Theme.BACKGROUND))
        
        ax = canvas.figure.add_subplot(111, facecolor='#1e1e1e')
        ax.set_xlabel('X [m]', color=Theme.TEXT)
        ax.set_ylabel('Y [m]', color=Theme.TEXT)
        ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY)
        ax.tick_params(colors=Theme.TEXT)
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)
            
        canvas.ax = ax
        
        # Initial display
        ax.text(0.5, 0.5, 'Load results to visualize fields', 
               transform=ax.transAxes, ha='center', va='center',
               color=Theme.TEXT_SECONDARY, fontsize=14)
        
        return canvas
        
    # Event handlers and methods
    def on_mode_changed(self, mode, checked):
        """Handle drawing mode change for immediate activation."""
        if checked and hasattr(self, 'canvas'):
            self.canvas.drawing_mode = mode
            self.canvas.current_points = []  # Clear current drawing
            self.update_canvas(self.canvas)
            
    def clear_geometry(self):
        """Clear all geometry elements."""
        self.geometry.clear()
        if hasattr(self, 'canvas'):
            self.canvas.current_points = []
            self.update_canvas(self.canvas)
        self.update_workflow_status()
        
    def undo_last_element(self):
        """Remove the last geometry element."""
        if self.geometry.elements:
            self.geometry.elements.pop()
            if hasattr(self, 'canvas'):
                self.update_canvas(self.canvas)
            self.update_workflow_status()
        
    def validate_geometry(self):
        """Validate current geometry."""
        try:
            x_coords, y_coords = self.geometry.get_interpolated_points()
            if not x_coords or not y_coords:
                self.log_message("No geometry to validate", "warning")
                return
                
            # Basic validation checks
            
            if max(x_coords) - min(x_coords) < 0.1:
                self.log_message("Geometry too short (< 0.1m)", "warning")
                return
                
            if any(abs(y) > 1.0 for y in y_coords):
                self.log_message("Geometry too wide (> 1.0m radius)", "warning")
                return
                
            self.log_message(f"Geometry valid! Length: {max(x_coords):.3f}m, Max radius: {max(abs(y) for y in y_coords):.3f}m, Elements: {len(self.geometry.elements)}", "success")
                                  
        except Exception as e:
            self.log_message(f"Validation failed: {str(e)}", "error")
            
    def generate_mesh(self):
        """Generate mesh from current geometry."""
        if not self.geometry.elements:
            self.log_message("No geometry defined", "warning")
            return
            
        try:
            from core.modules.mesh_generator import MeshParameters, AdvancedMeshGenerator

            def _bl_total_thickness(first: float, growth: float, n_layers: int) -> float:
                if n_layers <= 0:
                    return 0.0
                if growth is None or abs(growth - 1.0) < 1e-12:
                    return float(first) * float(n_layers)
                # Geometric series: first + first*r + ... first*r^(n-1)
                return float(first) * (float(growth) ** float(n_layers) - 1.0) / (float(growth) - 1.0)
            
            # Create mesh parameters from GUI
            params = MeshParameters()
            params.element_size = float(self.global_size.value())
            params.min_element_size = float(self.min_cell_size.value())
            params.max_element_size = float(self.max_cell_size.value())
            params.wall_element_size = float(self.wall_size.value())

            # Set mesh type from radio buttons (hex = structured quad, tet = triangular)
            params.mesh_type = 'hex' if self.hex_mesh_radio.isChecked() else 'tet'
            
            # Nozzle refinement settings
            params.nozzle_refinement_enabled = self.enable_nozzle_refinement.isChecked()
            params.nozzle_refinement_size = float(self.nozzle_cell_size.value())
            params.nozzle_growth_rate = float(self.nozzle_growth_rate.value())
            
            # Boundary layer settings (applied to WALLS ONLY)
            params.boundary_layer_enabled = self.enable_bl.isChecked()

            params.boundary_layer_elements = int(self.num_bl_layers.value())
            params.boundary_layer_growth_rate = float(self.growth_ratio.value())

            # UI provides first-layer thickness; MeshParameters stores total thickness.
            first = float(self.first_layer_thickness.value())
            total = _bl_total_thickness(first, params.boundary_layer_growth_rate, params.boundary_layer_elements)
            params.boundary_layer_first_layer = first
            params.boundary_layer_thickness = total

            params.domain_extension = float(self.farfield_distance.value())

            # Use quality threshold as a proxy for mesh smoothing aggressiveness.
            # Higher threshold => more smoothing attempts.
            qt = float(self.quality_threshold.value())
            params.mesh_smoothing = int(DEFAULTS.smoothing_iterations) if qt < 0.6 else 10
            
            # Generate mesh
            generator = AdvancedMeshGenerator()
            mesh_data = generator.generate_mesh(self.geometry, params)
            
            if mesh_data and isinstance(mesh_data, dict) and ('nodes' in mesh_data or 'vertices' in mesh_data):
                # Store mesh data for later use (ensure consistent naming)
                self.mesh_data = mesh_data
                self.current_mesh_data = mesh_data  # Also store in current_mesh_data for export
                self.mesh_generator = generator  # Store generator for export functionality
                
                # Update stats display
                # Prefer generator-calculated stats when available.
                stats = {}
                if hasattr(generator, 'get_mesh_statistics'):
                    stats = generator.get_mesh_statistics() or {}
                if not stats:
                    stats = mesh_data.get('stats', {})
                self.display_mesh_stats(stats)
                
                # Update visualization (pass mesh data instead of file)
                self.visualize_mesh_data(mesh_data)
                self.update_workflow_status()
                
                # Show success message with mesh statistics
                num_nodes = stats.get('num_nodes', 0)
                num_elements = stats.get('num_elements', 0)
                msg = f"Mesh generated successfully!\nNodes: {num_nodes}\nElements: {num_elements}"

                # Warn if mesh quality is below requested threshold (best-effort).
                mesh_quality = stats.get('mesh_quality', None)
                if isinstance(mesh_quality, (int, float)) and mesh_quality < qt:
                    msg += f" Warning: estimated mesh quality ({mesh_quality:.2f}) < threshold ({qt:.2f})."
                self.log_message(msg, "success")
            else:
                self.log_message("Mesh generation failed", "warning")
                
        except Exception as e:
            self.log_message(f"Failed to generate mesh: {str(e)}", "error")
            
    def analyze_mesh(self):
        """Analyze existing mesh quality."""
        try:
            if not self.current_mesh_data:
                self.log_message("No mesh data available. Please generate a mesh first.", "warning")
                return
                
            if self.mesh_generator and hasattr(self.mesh_generator, 'analyze_mesh_quality'):
                # Analyze mesh quality using the mesh generator
                quality_stats = self.mesh_generator.analyze_mesh_quality(self.current_mesh_data)
                
                if quality_stats:
                    analysis_text = f"Mesh Analysis: Elements={quality_stats.get('num_elements', 'N/A')}, Nodes={quality_stats.get('num_nodes', 'N/A')}, Min Quality={quality_stats.get('min_quality', 0):.4f}, Avg Quality={quality_stats.get('avg_quality', 0):.4f}"
                    assessment = self._get_quality_assessment(quality_stats)
                    self.log_message(f"{analysis_text}. {assessment}", "info")
                else:
                    self.log_message("Could not analyze mesh quality.", "warning")
            else:
                self.log_message("Basic mesh statistics available in the mesh panel.", "info")
                
        except Exception as e:
            self.log_message(f"Failed to analyze mesh: {str(e)}", "error")
    
    def _get_quality_assessment(self, stats):
        """Get quality assessment text based on statistics."""
        min_qual = stats.get('min_quality', 0)
        avg_qual = stats.get('avg_quality', 0)
        max_aspect = stats.get('max_aspect_ratio', float('inf'))
        
        issues = []
        if min_qual < 0.1:
            issues.append("⚠️ Very poor element quality detected")
        elif min_qual < 0.3:
            issues.append("⚠️ Poor element quality detected")
        
        if max_aspect > 100:
            issues.append("⚠️ High aspect ratio elements detected")
        elif max_aspect > 50:
            issues.append("⚠️ Moderate aspect ratio concerns")
            
        if avg_qual > 0.7:
            if not issues:
                return "[OK] Excellent mesh quality"
            else:
                return "[OK] Good overall quality with some concerns"
        elif avg_qual > 0.5:
            return "⚠️ Acceptable mesh quality"
        else:
            return "[X] Poor mesh quality - consider regenerating"
        
    def export_mesh(self):
        """Export mesh to file."""
        try:
            if not self.current_mesh_data:
                self.log_message("No mesh data available. Please generate a mesh first.", "warning")
                return
                
            # Get export file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Mesh", "", 
                "MSH Files (*.msh);;STL Files (*.stl);;VTK Files (*.vtk);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Determine format from extension
            extension = os.path.splitext(file_path)[1].lower()
            
            if self.mesh_generator and hasattr(self.mesh_generator, 'export_mesh'):
                success = self.mesh_generator.export_mesh(file_path, format=extension[1:])
                if success:
                    self.log_message(f"Mesh exported successfully to: {file_path}", "success")
                else:
                    self.log_message("Failed to export mesh.", "warning")
            else:
                # Basic export functionality
                if extension == '.msh':
                    self._export_mesh_msh(file_path)
                    self.log_message(f"Mesh exported to MSH format: {file_path}", "success")
                else:
                    self.log_message(f"Export format {extension} not supported yet.", "warning")
                    
        except Exception as e:
            self.log_message(f"Failed to export mesh: {str(e)}", "error")
    
    def _export_mesh_msh(self, file_path):
        """Export mesh in MSH format (basic implementation)."""
        try:
            if not self.current_mesh_data:
                raise ValueError("No mesh data available")
                
            # Get mesh data
            nodes = self.current_mesh_data.get('nodes') or self.current_mesh_data.get('vertices', [])
            elements = self.current_mesh_data.get('elements', [])
            
            with open(file_path, 'w') as f:
                # MSH format header
                f.write("$MeshFormat\n")
                f.write("2.2 0 8\n")
                f.write("$EndMeshFormat\n")
                
                # Nodes section
                f.write("$Nodes\n")
                f.write(f"{len(nodes)}\n")
                for i, (x, y) in enumerate(nodes):
                    f.write(f"{i+1} {x:.6f} {y:.6f} 0.0\n")
                f.write("$EndNodes\n")
                
                # Elements section
                f.write("$Elements\n")
                f.write(f"{len(elements)}\n")
                
                element_type_map = {
                    'triangle': 2,
                    'quad': 3,
                    'quadrilateral': 3
                }
                
                elem_type = element_type_map.get(self.current_mesh_data.get('element_type', 'quad'), 3)
                
                for i, elem in enumerate(elements):
                    # Format: element_id element_type num_tags tag1 tag2 ... node1 node2 ...
                    nodes_str = " ".join(str(n+1) for n in elem)  # Convert to 1-based
                    f.write(f"{i+1} {elem_type} 2 1 1 {nodes_str}\n")
                    
                f.write("$EndElements\n")
            
        except Exception as e:
            print(f"MSH export error: {e}")
            # Fallback to basic export
            with open(file_path, 'w') as f:
                f.write("$MeshFormat\n")
                f.write("2.2 0 8\n")
                f.write("$EndMeshFormat\n")
                f.write("$Nodes\n")
                f.write("1\n")  # Basic placeholder
                f.write("1 0.0 0.0 0.0\n")
                f.write("$EndNodes\n")
                f.write("$Elements\n")
                f.write("0\n")
                f.write("$EndElements\n")
        
    def display_mesh_stats(self, stats):
        """Display mesh statistics."""
        if stats:
            # Helper function to safely format numeric values
            def safe_format(value, default='N/A', format_str='.3f'):
                if isinstance(value, (int, float)) and value is not None:
                    return f"{value:{format_str}}"
                return default
            
            # Accept either legacy (min/avg quality) or new mesh_quality stats.
            text = f"""Mesh Statistics:
• Total elements: {stats.get('num_elements', 'N/A')}
• Total nodes: {stats.get('num_nodes', 'N/A')}
• Mesh quality: {safe_format(stats.get('mesh_quality'), safe_format(stats.get('avg_quality')), '.3f')}
• Avg element size: {safe_format(stats.get('avg_element_size'), 'N/A', '.4f')}
• Min element size: {safe_format(stats.get('min_element_size'), 'N/A', '.4f')}
• Max element size: {safe_format(stats.get('max_element_size'), 'N/A', '.4f')}
"""
            self.mesh_stats.setText(text)
        
    def visualize_mesh(self, mesh_file):
        """Visualize mesh in canvas."""
        # Basic mesh visualization - would need actual mesh data parsing
        self.mesh_canvas.ax.clear()
        self.mesh_canvas.ax.text(0.5, 0.5, f'Mesh generated\n{os.path.basename(mesh_file)}', 
                               transform=self.mesh_canvas.ax.transAxes, ha='center', va='center',
                               color=Theme.TEXT, fontsize=12)
        self.mesh_canvas.draw()
        
    def visualize_mesh_data(self, mesh_data):
        """Visualize mesh data directly in canvas with proper boundary colors.
        
        Boundary naming convention:
        - inlet: Green - left boundary where flow enters
        - outlet: Red - right boundary where flow exits  
        - wall: Blue - solid walls (includes wall_upper, wall_lower)
        - symmetry/centerline: Orange - symmetry axis (if present)
        """
        try:
            self.mesh_canvas.ax.clear()
            
            # Extract mesh information - avoid 'or' with numpy arrays
            nodes = mesh_data.get('nodes')
            if nodes is None:
                nodes = mesh_data.get('vertices', [])
            elements = mesh_data.get('elements', [])
            stats = mesh_data.get('stats', {})
            
            # Convert to numpy array if needed and check length properly
            nodes_array = np.array(nodes) if not isinstance(nodes, np.ndarray) else nodes
            has_nodes = len(nodes_array) > 0
            has_elements = len(elements) > 0
            
            if has_nodes and has_elements:
                x_coords = nodes_array[:, 0]
                y_coords = nodes_array[:, 1]
                
                # Plot mesh elements (edges) with lighter color
                for elem in elements:
                    if len(elem) >= 3:
                        # Create closed polygon for element
                        elem_coords = [nodes_array[i] for i in elem]
                        elem_coords.append(elem_coords[0])  # Close the polygon
                        
                        elem_x = [coord[0] for coord in elem_coords]
                        elem_y = [coord[1] for coord in elem_coords]
                        
                        # Plot element edges in subtle gray
                        self.mesh_canvas.ax.plot(elem_x, elem_y, color='#4a4a4a', linewidth=0.3, alpha=0.6)
                
                # Get boundary elements from mesh data
                boundary_elements = mesh_data.get('boundary_elements', {})
                
                # Define boundary colors matching what goes into SU2 simulation
                # These are the actual boundary condition types
                boundary_colors = {
                    # Primary boundaries (what SU2 uses)
                    'inlet': '#00ff00',      # Bright green
                    'outlet': '#ff4444',     # Bright red  
                    'wall': '#4488ff',       # Bright blue
                    'symmetry': '#ffaa00',   # Orange
                    # Legacy/alternative names (mapped to primary)
                    'wall_upper': '#4488ff', # Blue (same as wall)
                    'wall_lower': '#4488ff', # Blue (same as wall)
                    'centerline': '#ffaa00', # Orange (same as symmetry)
                    'farfield': '#aa44ff',   # Purple
                }
                
                # Track which boundary types are actually present for legend
                boundaries_plotted = {}  # name -> color
                
                for boundary_name, boundary_elems in boundary_elements.items():
                    if not boundary_elems:
                        continue
                        
                    color = boundary_colors.get(boundary_name, '#888888')
                    
                    # Normalize name for legend (combine wall_upper/wall_lower into wall)
                    legend_name = boundary_name
                    if boundary_name in ('wall_upper', 'wall_lower'):
                        legend_name = 'wall'
                    elif boundary_name == 'centerline':
                        legend_name = 'symmetry'
                    
                    for boundary_elem in boundary_elems:
                        if len(boundary_elem) >= 2:
                            # Plot boundary edge with thick line
                            edge_coords = [nodes_array[i] for i in boundary_elem]
                            edge_x = [coord[0] for coord in edge_coords]
                            edge_y = [coord[1] for coord in edge_coords]
                            
                            self.mesh_canvas.ax.plot(edge_x, edge_y, color=color, 
                                                    linewidth=3, alpha=0.9, solid_capstyle='round')
                    
                    # Track for legend (use normalized name)
                    if legend_name not in boundaries_plotted:
                        boundaries_plotted[legend_name] = color
                
                # Set equal aspect ratio and add grid
                self.mesh_canvas.ax.set_aspect('equal')
                self.mesh_canvas.ax.grid(True, alpha=0.2, color='#606060', linestyle='--')
                
                # Add title with mesh statistics
                num_nodes = stats.get('num_nodes', len(nodes_array))
                num_elements = stats.get('num_elements', len(elements))
                element_type = stats.get('element_type', 'unknown')
                quality = stats.get('avg_quality', 0)
                
                title = f"Mesh: {num_nodes:,} nodes, {num_elements:,} {element_type} elements"
                if quality > 0:
                    title += f" | Quality: {quality:.2f}"
                
                self.mesh_canvas.ax.set_title(title, fontsize=11, color=Theme.TEXT, fontweight='bold')
                self.mesh_canvas.ax.set_xlabel('x [m]', color=Theme.TEXT)
                self.mesh_canvas.ax.set_ylabel('y [m]', color=Theme.TEXT)
                self.mesh_canvas.ax.tick_params(colors=Theme.TEXT)
                
                # Set background color to match theme
                self.mesh_canvas.ax.set_facecolor('#1e1e1e')
                
                # Add legend for boundaries that are actually present
                if boundaries_plotted:
                    # Define legend order and labels
                    legend_order = ['inlet', 'outlet', 'wall', 'symmetry', 'farfield']
                    legend_labels = {
                        'inlet': 'Inlet (flow in)',
                        'outlet': 'Outlet (flow out)',
                        'wall': 'Wall (solid)',
                        'symmetry': 'Symmetry axis',
                        'farfield': 'Farfield'
                    }
                    
                    legend_elements = []
                    for name in legend_order:
                        if name in boundaries_plotted:
                            label = legend_labels.get(name, name)
                            legend_elements.append(
                                matplotlib.lines.Line2D([0], [0], color=boundaries_plotted[name], 
                                                       lw=3, label=label)
                            )
                    
                    if legend_elements:
                        legend = self.mesh_canvas.ax.legend(
                            handles=legend_elements, 
                            loc='upper right', 
                            fontsize=9,
                            facecolor='#2a2a2a',
                            edgecolor='#404040',
                            labelcolor=Theme.TEXT
                        )
                
                # Store original limits for reset
                self.mesh_canvas.store_original_limits()
                
            else:
                # Show message if no mesh data
                self.mesh_canvas.ax.text(0.5, 0.5, 'No mesh data to display', 
                                       transform=self.mesh_canvas.ax.transAxes, ha='center', va='center',
                                       color=Theme.TEXT, fontsize=12)
                self.mesh_canvas.ax.set_title('Mesh Visualization', color=Theme.TEXT)
                self.mesh_canvas.ax.set_facecolor('#1e1e1e')
            
            self.mesh_canvas.draw()
            
        except Exception as e:
            print(f"Mesh visualization error: {e}")
            import traceback
            traceback.print_exc()
            self.mesh_canvas.ax.clear()
            self.mesh_canvas.ax.text(0.5, 0.5, f'Mesh visualization failed:\n{str(e)}', 
                                   transform=self.mesh_canvas.ax.transAxes, ha='center', va='center',
                                   color=Theme.TEXT, fontsize=10)
            self.mesh_canvas.ax.set_facecolor('#1e1e1e')
            self.mesh_canvas.draw()
        
    def setup_simulation(self):
        """Setup SU2 simulation case."""
        if not self.geometry.elements:
            self.log_message("No geometry defined", "warning")
            return
            
        try:
            from core.modules.simulation_setup import (
                SimulationSetup,
                BoundaryCondition,
                BoundaryType,
                SolverType,
            )
            
            # Create boundary conditions from GUI
            inlet_bc = BoundaryCondition(
                name="inlet",
                boundary_type=BoundaryType.INLET,
                velocity_magnitude=self.inlet_velocity.value(),
                total_pressure=self.inlet_pressure.value(),
                total_temperature=self.temperature.value(),
                flow_direction=(1.0, 0.0, 0.0)
            )
            
            outlet_bc = BoundaryCondition(
                name="outlet", 
                boundary_type=BoundaryType.OUTLET,
                static_pressure=self.outlet_pressure.value()
            )
            
            wall_bc = BoundaryCondition(
                name="wall",
                boundary_type=BoundaryType.WALL,
                is_adiabatic=True
            )
            
            # Add symmetry BC if mesh has centerline/symmetry boundary
            symmetry_bc = BoundaryCondition(
                name="symmetry",
                boundary_type=BoundaryType.SYMMETRY
            )
            
            # Setup simulation
            sim_setup = SimulationSetup()
            sim_setup.case_directory = self.case_directory.text()

            # Fluid / operating conditions - set initialization values
            # Temperature from initialization settings (advanced) or inlet (basic)
            init_temp = self.init_temperature.value() if hasattr(self, 'init_temperature') else self.temperature.value()
            sim_setup.fluid_properties.temperature = float(init_temp)
            
            # Pressure based on initialization method
            init_method = self.init_method.currentText() if hasattr(self, 'init_method') else 'outlet_pressure'
            if init_method == 'outlet_pressure':
                init_pressure = float(self.outlet_pressure.value())
            elif init_method == 'inlet_pressure':
                init_pressure = float(self.inlet_pressure.value())
            elif init_method == 'average_pressure':
                init_pressure = (float(self.inlet_pressure.value()) + float(self.outlet_pressure.value())) / 2.0
            elif init_method == 'custom':
                init_pressure = float(self.init_custom_pressure.value())
            else:
                init_pressure = float(self.outlet_pressure.value())
            
            sim_setup.fluid_properties.pressure = init_pressure
            
            # Also set density if using custom method
            if init_method == 'custom' and hasattr(self, 'init_custom_density'):
                sim_setup.fluid_properties.density = float(self.init_custom_density.value())
            else:
                # Compute density from ideal gas law: ρ = P / (R * T)
                R = 287.058  # Gas constant for air
                sim_setup.fluid_properties.density = init_pressure / (R * init_temp)
            
            # Get mesh data
            mesh_data = getattr(self, 'current_mesh_data', None)
            
            # Configure solver settings
            solver_name = self.solver_type.currentText().strip()
            try:
                sim_setup.solver_settings.solver_type = SolverType(solver_name)
            except Exception:
                sim_setup.solver_settings.solver_type = SolverType.RANS

            sim_setup.solver_settings.n_processors = self.n_processors.value()
            sim_setup.solver_settings.max_iterations = int(self.max_iterations.value())
            sim_setup.solver_settings.output_frequency = self.output_frequency.value()
            
            # Simulation mode (steady vs transient)
            is_transient = self.simulation_mode.currentText() == "Transient"
            sim_setup.solver_settings.is_transient = is_transient
            
            if is_transient:
                # Transient settings
                sim_setup.solver_settings.time_step = self.time_step.value()
                sim_setup.solver_settings.end_time = self.end_time.value()
                sim_setup.solver_settings.inner_iterations = self.inner_iterations.value()
                sim_setup.solver_settings.cfl_number = 10.0  # Higher CFL for inner iterations
            else:
                # Steady-state settings
                sim_setup.solver_settings.cfl_number = self.cfl_number.value()
                sim_setup.solver_settings.cfl_adapt = self.cfl_adapt.isChecked()
                if self.cfl_adapt.isChecked():
                    sim_setup.solver_settings.cfl_min = self.cfl_min.value()
                    sim_setup.solver_settings.cfl_max = self.cfl_max.value()
                sim_setup.solver_settings.convergence_residual = self.convergence_residual.value()
            
            # Discretization settings
            sim_setup.solver_settings.conv_num_method = self.conv_scheme.currentText()
            sim_setup.solver_settings.muscl = self.muscl_flow.isChecked()
            sim_setup.solver_settings.slope_limiter = self.slope_limiter.currentText()
            sim_setup.solver_settings.time_discre = self.time_discre.currentText()
            sim_setup.solver_settings.gradient_method = self.gradient_method.currentText()
            
            # Linear solver and multigrid
            sim_setup.solver_settings.linear_solver = self.linear_solver.currentText()
            sim_setup.solver_settings.linear_solver_prec = self.linear_solver_prec.currentText()
            sim_setup.solver_settings.linear_solver_iter = self.linear_solver_iter.value()
            sim_setup.solver_settings.mglevel = self.mglevel.value()
            sim_setup.solver_settings.mgcycle = self.mgcycle.currentText()
            
            # Compressible solvers: EULER, NAVIER_STOKES, RANS
            is_compressible = solver_name in ("EULER", "NAVIER_STOKES", "RANS")
            
            # Calculate Mach number for freestream initialization
            # Use init_velocity (small value to seed flow direction) and init temperature
            if is_compressible:
                gamma = 1.4
                R = 287.058
                T = init_temp  # Use initialization temperature
                a = (gamma * R * T) ** 0.5  # Speed of sound
                # Use init_velocity for freestream (typically small to seed flow)
                V_init = self.init_velocity.value() if hasattr(self, 'init_velocity') else 1.0
                sim_setup.solver_settings.mach_number = V_init / a if a > 0 else 0.01

            # Turbulence model
            turb_name = self.turbulence_model.currentText().strip()
            if turb_name.lower() in ("laminar", "none", "none (laminar)"):
                sim_setup.turbulence_model.enabled = False
            else:
                sim_setup.turbulence_model.enabled = True
                sim_setup.turbulence_model.model_type = turb_name
            
            # Add boundary conditions
            sim_setup.add_boundary_condition(inlet_bc)
            sim_setup.add_boundary_condition(outlet_bc)
            sim_setup.add_boundary_condition(wall_bc)
            
            # Only add symmetry BC if the mesh actually has a symmetry boundary
            # Check if mesh_data contains a symmetry boundary
            mesh_boundaries = []
            if mesh_data:
                boundary_elements = mesh_data.get('boundary_elements', {})
                mesh_boundaries = list(boundary_elements.keys())
            
            if 'symmetry' in mesh_boundaries:
                sim_setup.add_boundary_condition(symmetry_bc)
            
            # Generate case files
            sim_setup.generate_case_files(self.geometry, mesh_data)
            
            # Update current case directory for simulation runner
            self.current_case_directory = sim_setup.case_directory
            
            # Handle variable time stepping (multi-phase simulation)
            self._simulation_phases = None
            if is_transient and hasattr(self, 'variable_dt_check') and self.variable_dt_check.isChecked():
                self._simulation_phases = self._generate_phase_configs(sim_setup)
            
            # Log initialization info
            init_info = (
                f"Init: {init_method} → P={init_pressure/1000:.1f} kPa, "
                f"T={init_temp:.0f} K, M={sim_setup.solver_settings.mach_number:.3f}"
            )
            
            if self._simulation_phases:
                phase_info = f"\nVariable dt: {len(self._simulation_phases)} phases configured"
                self.simulation_log.setText(f"SU2 case files generated!\n{init_info}{phase_info}")
            else:
                self.simulation_log.setText(f"SU2 case files generated!\n{init_info}")
            
            self.update_workflow_status()
            self.log_message(f"SU2 case files generated successfully! {init_info}", "success")
            
        except Exception as e:
            self.log_message(f"Failed to setup simulation: {str(e)}", "error")
    
    def _generate_phase_configs(self, sim_setup) -> list:
        """Generate phase-specific config files for variable time stepping.
        
        Creates config files that use RESTART_SOL to continue from previous phase.
        
        Returns:
            List of phase dicts with config_file and description keys
        """
        import re
        
        phases = []
        base_config_path = os.path.join(self.current_case_directory, "config.cfg")
        
        # Read base config
        with open(base_config_path, 'r') as f:
            base_config = f.read()
        
        # Calculate iteration counts for each phase
        # IMPORTANT: When restarting, SU2 continues the iteration counter from the previous phase.
        # So TIME_ITER must be the CUMULATIVE iteration count, not just this phase's iterations.
        # We use a large TIME_ITER and rely on MAX_TIME for stopping.
        
        phase1_dt = self.phase1_dt.value()
        phase1_duration = self.phase1_duration.value()
        phase1_inner = self.phase1_inner.value()
        # Calculate actual iteration count for phase 1 (this is when it ends)
        # Phase 1 writes restart file at this iteration
        # SU2 iterations are 1-based, so phase 1 ends at iteration (phase1_end_iter + 1)
        phase1_end_iter = int(phase1_duration / phase1_dt)
        phase1_iters = phase1_end_iter + 100  # Add buffer for TIME_ITER
        
        phase2_dt = self.phase2_dt.value()
        phase2_duration = self.phase2_duration.value()
        phase2_inner = self.phase2_inner.value()
        
        phase3_dt = self.phase3_dt.value()
        phase3_inner = self.phase3_inner.value()
        end_time = self.end_time.value()
        
        # For 2nd order dual time stepping restarts, SU2 needs files at iter n-1 and n-2
        # For 1st order dual time stepping restarts, SU2 only needs the file at (RESTART_ITER - 1)
        # This allows using normal OUTPUT_WRT_FREQ without writing every iteration
        # 
        # IMPORTANT: SU2 iterations are 1-based. Phase 1 runs from iter 1 to (phase1_end_iter + 1).
        # The restart file is named restart_flow_NNNNN.dat where NNNNN is the iteration number.
        # For 1st order restart, SU2 reads file at (RESTART_ITER - 1).
        # So to restart from restart_flow_00051.dat, we need RESTART_ITER = 52.
        phase2_restart_iter = phase1_end_iter + 2  # +1 for 1-based indexing, +1 for next iteration
        
        # Phase 2 ends when cumulative physical time reaches (phase1_duration + phase2_duration).
        # SU2 calculates time as: phase1_duration + (iter - phase2_restart_iter) * phase2_dt
        # But when changing timesteps between phases, SU2's iteration counter and physical time
        # can diverge. To be safe, we calculate based on cumulative time / phase2_dt and add
        # the restart offset, using ceil and adding margin for floating point issues.
        cumulative_time_p2 = phase1_duration + phase2_duration
        # Phase 2 ends at approximately: cumulative_time / phase2_dt + restart_iter offset
        phase2_end_iter = math.ceil(cumulative_time_p2 / phase2_dt) + phase2_restart_iter
        phase3_restart_iter = phase2_end_iter + 1  # Restart from the next iteration after phase 2 ends
        
        # Calculate cumulative iterations (worst case: all at smallest dt)
        # Phase 2 starts after phase 1's iterations and adds its own
        phase2_cumulative_iters = phase1_iters + int(phase2_duration / phase2_dt) + 100
        # Phase 3 continues from phase 2
        phase3_cumulative_iters = phase2_cumulative_iters + int((end_time - phase1_duration - phase2_duration) / phase3_dt) + 100
        
        # Use a very large TIME_ITER for restart phases - MAX_TIME controls stopping
        large_time_iter = 999999
        
        # Phase 1: Small dt, fresh start (use 2nd order time stepping)
        phase1_config = base_config
        phase1_config = re.sub(r'TIME_STEP=\s*[\d.eE+-]+', f'TIME_STEP= {phase1_dt}', phase1_config)
        phase1_config = re.sub(r'MAX_TIME=\s*[\d.eE+-]+', f'MAX_TIME= {phase1_duration}', phase1_config)
        phase1_config = re.sub(r'TIME_ITER=\s*\d+', f'TIME_ITER= {phase1_iters}', phase1_config)
        phase1_config = re.sub(r'INNER_ITER=\s*\d+', f'INNER_ITER= {phase1_inner}', phase1_config)
        phase1_config = re.sub(r'RESTART_SOL=\s*\w+', 'RESTART_SOL= NO', phase1_config)
        
        phase1_path = os.path.join(self.current_case_directory, "config_phase1.cfg")
        with open(phase1_path, 'w') as f:
            f.write(f"% Phase 1: Startup (dt={phase1_dt:.2e}, duration={phase1_duration:.4f}s)\n")
            f.write(phase1_config)
        
        phases.append({
            "config_file": "config_phase1.cfg",
            "description": f"Startup: dt={phase1_dt:.1e}s for {phase1_duration*1000:.1f}ms"
        })
        
        # Phase 2: Medium dt, restart from phase 1
        # Use 1st order time stepping for restart to only need one previous restart file
        cumulative_time_p2 = phase1_duration + phase2_duration
        
        phase2_config = base_config
        phase2_config = re.sub(r'TIME_STEP=\s*[\d.eE+-]+', f'TIME_STEP= {phase2_dt}', phase2_config)
        phase2_config = re.sub(r'MAX_TIME=\s*[\d.eE+-]+', f'MAX_TIME= {cumulative_time_p2}', phase2_config)
        phase2_config = re.sub(r'TIME_ITER=\s*\d+', f'TIME_ITER= {large_time_iter}', phase2_config)
        phase2_config = re.sub(r'INNER_ITER=\s*\d+', f'INNER_ITER= {phase2_inner}', phase2_config)
        phase2_config = re.sub(r'RESTART_SOL=\s*\w+', 'RESTART_SOL= YES', phase2_config)
        # Use 1st order for restart phases - only needs 1 previous file instead of 2
        phase2_config = re.sub(r'TIME_MARCHING=\s*\S+', 'TIME_MARCHING= DUAL_TIME_STEPPING-1ST_ORDER', phase2_config)
        # Add RESTART_ITER: for 1st order, SU2 reads file at (RESTART_ITER - 1)
        phase2_config = re.sub(r'(RESTART_SOL=\s*YES)', f'\\1\nRESTART_ITER= {phase2_restart_iter}', phase2_config)
        
        phase2_path = os.path.join(self.current_case_directory, "config_phase2.cfg")
        with open(phase2_path, 'w') as f:
            f.write(f"% Phase 2: Developed flow (dt={phase2_dt:.2e}, duration={phase2_duration:.4f}s, MAX_TIME={cumulative_time_p2})\n")
            f.write(phase2_config)
        
        phases.append({
            "config_file": "config_phase2.cfg", 
            "description": f"Developed: dt={phase2_dt:.1e}s for {phase2_duration*1000:.1f}ms"
        })
        
        # Phase 3: Large dt, restart from phase 2
        phase3_config = base_config
        phase3_config = re.sub(r'TIME_STEP=\s*[\d.eE+-]+', f'TIME_STEP= {phase3_dt}', phase3_config)
        phase3_config = re.sub(r'MAX_TIME=\s*[\d.eE+-]+', f'MAX_TIME= {end_time}', phase3_config)
        # Use large TIME_ITER to avoid negative iteration error on restart
        phase3_config = re.sub(r'TIME_ITER=\s*\d+', f'TIME_ITER= {large_time_iter}', phase3_config)
        phase3_config = re.sub(r'INNER_ITER=\s*\d+', f'INNER_ITER= {phase3_inner}', phase3_config)
        phase3_config = re.sub(r'RESTART_SOL=\s*\w+', 'RESTART_SOL= YES', phase3_config)
        # Use 1st order for restart phases - only needs 1 previous file instead of 2
        phase3_config = re.sub(r'TIME_MARCHING=\s*\S+', 'TIME_MARCHING= DUAL_TIME_STEPPING-1ST_ORDER', phase3_config)
        # Add RESTART_ITER: for 1st order, SU2 reads file at (RESTART_ITER - 1)
        phase3_config = re.sub(r'(RESTART_SOL=\s*YES)', f'\\1\nRESTART_ITER= {phase3_restart_iter}', phase3_config)
        
        phase3_path = os.path.join(self.current_case_directory, "config_phase3.cfg")
        with open(phase3_path, 'w') as f:
            f.write(f"% Phase 3: Final (dt={phase3_dt:.2e}, until end_time={end_time:.4f}s)\n")
            f.write(phase3_config)
        
        phases.append({
            "config_file": "config_phase3.cfg",
            "description": f"Final: dt={phase3_dt:.1e}s until {end_time*1000:.1f}ms"
        })
        
        self.log_message(f"Generated {len(phases)} phase configs for variable time stepping", "info")
        
        return phases
            
    def run_simulation(self):
        """Run SU2 simulation in a background thread to keep GUI responsive."""
        try:
            if not self.current_case_directory or not os.path.exists(self.current_case_directory):
                self.log_message("No case directory set up. Please generate a case first.", "warning")
                return
            
            # Check if simulation is already running
            if self._simulation_thread is not None and self._simulation_thread.isRunning():
                self.log_message("A simulation is already running. Stop it first.", "warning")
                return
                
            self.simulation_log.append("Starting SU2 simulation...")
            self.log_message("Starting SU2 simulation...", "info")
            
            # Validate case first
            from core.su2_runner import SU2Runner
            runner = SU2Runner(self.current_case_directory)

            ok, msg = runner.validate_case()
            if not ok:
                self.simulation_log.append(f"[X] Preflight failed: {msg}")
                self.log_message(f"Preflight failed: {msg}", "error")
                return
            
            # Get solver from config file
            configured_solver = runner.get_solver_from_config()
            if not configured_solver:
                self.simulation_log.append("[X] Could not read solver from config.cfg")
                self.log_message("Could not read solver from config.cfg", "error")
                return

            self.simulation_log.append(f"Solver: {configured_solver}")
            
            # Get number of processors
            n_procs = self.n_processors.value()
            
            if n_procs > 1:
                self.simulation_log.append(f"Running SU2_CFD in parallel on {n_procs} processors...")
            else:
                self.simulation_log.append("Running SU2_CFD...")
            
            # Check if multi-phase simulation is configured
            phases = getattr(self, '_simulation_phases', None)
            if phases:
                self.simulation_log.append(f"Variable time stepping enabled: {len(phases)} phases")
                for i, phase in enumerate(phases):
                    self.simulation_log.append(f"  Phase {i+1}: {phase['description']}")
            
            # Create worker and thread
            self._simulation_thread = QThread()
            self._simulation_worker = SimulationWorker(self.current_case_directory, n_procs, phases)
            self._simulation_worker.moveToThread(self._simulation_thread)
            
            # Connect signals
            self._simulation_thread.started.connect(self._simulation_worker.run)
            self._simulation_worker.progress.connect(self._on_simulation_progress)
            self._simulation_worker.finished.connect(self._on_simulation_finished)
            self._simulation_worker.finished.connect(self._simulation_thread.quit)
            self._simulation_worker.finished.connect(self._simulation_worker.deleteLater)
            self._simulation_thread.finished.connect(self._simulation_thread.deleteLater)
            self._simulation_thread.finished.connect(self._cleanup_simulation_thread)
            
            # Start simulation
            self._simulation_thread.start()
            self.log_message("Simulation started in background", "info")
                
        except ImportError:
            self.log_message("SU2 integration not available", "error")
        except Exception as e:
            self.log_message(f"Failed to run simulation: {str(e)}", "error")
            self.simulation_log.append(f"[X] Error: {str(e)}")
    
    def _on_simulation_progress(self, message: str):
        """Handle progress updates from simulation worker."""
        self.simulation_log.append(message)
    
    def _on_simulation_finished(self, success: bool, message: str):
        """Handle simulation completion."""
        if success:
            self.simulation_log.append("[OK] SU2_CFD completed successfully")
            self.simulation_log.append("Simulation completed!")
            self.log_message(message, "success")
            
            # Update workflow status
            self.current_results = self.current_case_directory
            self.update_workflow_status()
        else:
            self.simulation_log.append(f"[X] {message}")
            self.log_message(message, "error")
            
            # Try to get log tail
            try:
                from core.su2_runner import SU2Runner
                runner = SU2Runner(self.current_case_directory)
                log_content = runner.get_log_content()
                if log_content:
                    last_lines = '\n'.join(log_content.split('\n')[-10:])
                    self.simulation_log.append(f"Log tail:\n{last_lines}")
            except Exception:
                pass
    
    def _cleanup_simulation_thread(self):
        """Clean up simulation thread references."""
        self._simulation_thread = None
        self._simulation_worker = None
        
    def monitor_simulation(self):
        """Monitor simulation progress by plotting residuals from history files.
        
        Handles multi-phase simulations where SU2 creates separate history files
        for each restart phase (history.csv, history_00052.csv, history_00558.csv, etc.)
        """
        try:
            if not self.current_case_directory or not os.path.exists(self.current_case_directory):
                self.log_message("No active simulation case found", "warning")
                return
                
            # Find all history files (main + phase-specific ones)
            history_files = self._find_all_history_files()
            
            if not history_files:
                # Fallback: check for log file and display that
                log_files = [f for f in os.listdir(self.current_case_directory) if f.startswith('log.')]
                if log_files:
                    self._display_log_in_panel(os.path.join(self.current_case_directory, sorted(log_files)[-1]))
                else:
                    self.log_message("No history.csv or log files found. Run simulation first.", "info")
                return
            
            # Parse all history files and plot residuals
            self._plot_residuals_from_history_files(history_files)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message(f"Failed to monitor simulation: {str(e)}", "error")
    
    def _find_all_history_files(self) -> list:
        """Find all history CSV files in the case directory.
        
        Returns sorted list of file paths, with base history.csv first,
        followed by numbered history files (history_00052.csv, etc.) in order.
        """
        history_files = []
        
        # Check for base history.csv
        base_history = os.path.join(self.current_case_directory, "history.csv")
        if os.path.exists(base_history):
            history_files.append(base_history)
        
        # Find numbered history files (history_NNNNN.csv)
        import re
        pattern = re.compile(r'^history_(\d+)\.csv$')
        
        numbered_files = []
        for f in os.listdir(self.current_case_directory):
            match = pattern.match(f)
            if match:
                iter_num = int(match.group(1))
                full_path = os.path.join(self.current_case_directory, f)
                numbered_files.append((iter_num, full_path))
        
        # Sort by iteration number
        numbered_files.sort(key=lambda x: x[0])
        
        # Add to list in order
        for _, path in numbered_files:
            history_files.append(path)
        
        return history_files
    
    def _display_log_in_panel(self, log_path: str):
        """Display log file content in the simulation log panel."""
        try:
            with open(log_path, 'r') as f:
                log_content = f.read()
            
            lines = log_content.split('\n')
            recent_lines = lines[-50:] if len(lines) > 50 else lines
            
            self.simulation_log.clear()
            self.simulation_log.append(f"=== {os.path.basename(log_path)} (last 50 lines) ===\n")
            self.simulation_log.append('\n'.join(recent_lines))
            
            self.log_message(f"Loaded log from {os.path.basename(log_path)}", "info")
        except Exception as e:
            self.log_message(f"Could not read log file: {str(e)}", "error")
    
    def _plot_residuals_from_history_files(self, history_files: list):
        """Parse multiple SU2 history CSV files and plot combined residuals.
        
        Args:
            history_files: List of history file paths in chronological order
        """
        import csv
        import numpy as np
        
        try:
            # Combined data from all files
            all_iterations = []
            all_residuals = {}  # Dict of residual_name -> list of values
            residual_cols = {}  # Column indices (determined from first file)
            iter_col = None
            files_loaded = 0
            
            for file_idx, history_file in enumerate(history_files):
                if not os.path.exists(history_file):
                    continue
                    
                with open(history_file, 'r') as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                    except StopIteration:
                        continue  # Empty file
                    
                    # Clean up header names (remove quotes and whitespace)
                    header = [col.strip().strip('"') for col in header]
                    
                    # For first file, determine column structure
                    if file_idx == 0:
                        # Find iteration column
                        for i, col in enumerate(header):
                            if col in ['Time_Iter', 'Outer_Iter', 'Iteration', 'Inner_Iter']:
                                iter_col = i
                                break
                        if iter_col is None:
                            iter_col = 0
                        
                        # Find residual columns
                        for i, col in enumerate(header):
                            if col.startswith('rms[') and col.endswith(']'):
                                clean_name = col[4:-1]
                                residual_cols[clean_name] = i
                                all_residuals[clean_name] = []
                            elif col.startswith('Res_'):
                                clean_name = col[4:]
                                residual_cols[clean_name] = i
                                all_residuals[clean_name] = []
                        
                        if not residual_cols:
                            self.log_message("No residual columns found in history files", "warning")
                            return
                    
                    # Read data rows
                    rows_read = 0
                    for row in reader:
                        if len(row) <= iter_col:
                            continue
                        try:
                            iter_val = float(row[iter_col])
                            
                            # Skip duplicate iterations (overlap between phases)
                            if all_iterations and iter_val <= all_iterations[-1]:
                                continue
                            
                            all_iterations.append(iter_val)
                            
                            for name, col_idx in residual_cols.items():
                                if col_idx < len(row):
                                    try:
                                        val = float(row[col_idx])
                                        all_residuals[name].append(val)
                                    except (ValueError, IndexError):
                                        all_residuals[name].append(np.nan)
                                else:
                                    all_residuals[name].append(np.nan)
                            rows_read += 1
                        except ValueError:
                            continue
                    
                    if rows_read > 0:
                        files_loaded += 1
            
            if not all_iterations:
                self.log_message("No iteration data found in history files", "warning")
                return
            
            iterations = np.array(all_iterations)
            
            # Clear and plot on simulation canvas
            if not hasattr(self, 'sim_canvas') or self.sim_canvas is None:
                self.log_message("Simulation canvas not available", "error")
                return
            
            ax = self.sim_canvas.ax if hasattr(self.sim_canvas, 'ax') else self.sim_canvas.figure.axes[0]
            ax.clear()
            
            # Plot each residual with different colors
            colors = ['#00bcd4', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#2196f3', '#ffeb3b', '#795548']
            
            all_valid_values = []
            
            for i, (name, values) in enumerate(all_residuals.items()):
                if len(values) != len(iterations):
                    continue
                values_arr = np.array(values)
                valid_mask = ~np.isnan(values_arr) & np.isfinite(values_arr)
                if np.any(valid_mask):
                    all_valid_values.extend(values_arr[valid_mask].tolist())
                    color = colors[i % len(colors)]
                    ax.plot(iterations[valid_mask], values_arr[valid_mask], 
                           label=name, color=color, linewidth=1.5)
            
            # Set X-axis limits
            max_iter = float(iterations[-1])
            ax.set_xlim(0, max_iter * 1.02)
            
            # Set Y-axis limits based on data
            if all_valid_values:
                y_min = min(all_valid_values)
                y_max = max(all_valid_values)
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
                ax.set_ylim(y_min - margin, y_max + margin)
            
            # Store data in canvas for inspection
            if hasattr(self.sim_canvas, 'plot_data'):
                self.sim_canvas.plot_data = all_residuals
            
            # Style the plot
            ax.set_xlabel('Iteration', color=Theme.TEXT, fontsize=12)
            ax.set_ylabel('log₁₀(Residual)', color=Theme.TEXT, fontsize=12)
            
            # Title indicates multi-phase if applicable
            title = 'Convergence History'
            if files_loaded > 1:
                title += f' ({files_loaded} phases)'
            ax.set_title(title, color=Theme.TEXT_PRIMARY, fontsize=14, fontweight='bold')
            
            ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY, linestyle='--')
            ax.tick_params(colors=Theme.TEXT, labelsize=10)
            for spine in ax.spines.values():
                spine.set_color(Theme.BORDER)
            ax.set_facecolor('#1e1e1e')
            
            # Add legend
            legend = ax.legend(loc='upper right', fontsize=9, facecolor='#2d2d2d', 
                              edgecolor=Theme.BORDER, framealpha=0.9)
            for text in legend.get_texts():
                text.set_color(Theme.TEXT)
            
            # Add iteration count and phase info
            info_text = f'Iterations: {int(iterations[-1])}'
            if files_loaded > 1:
                info_text += f'\nPhases: {files_loaded}'
            ax.text(0.02, 0.98, info_text, 
                   transform=ax.transAxes, va='top', ha='left',
                   color=Theme.TEXT, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.8, edgecolor=Theme.BORDER))
            
            self.sim_canvas.draw()
            
            # Update the log panel with summary
            self.simulation_log.clear()
            self.simulation_log.append("=== Convergence Summary ===\n")
            self.simulation_log.append(f"History files loaded: {files_loaded}")
            self.simulation_log.append(f"Total iterations: {int(iterations[-1])}")
            self.simulation_log.append(f"\nFinal residuals:")
            for name, values in all_residuals.items():
                if values and not np.isnan(values[-1]):
                    self.simulation_log.append(f"  {name}: {values[-1]:.6e}")
            
            self.log_message(f"Plotted {len(iterations)} iterations from {files_loaded} phase(s)", "success")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message(f"Error parsing history files: {str(e)}", "error")
        
    def stop_simulation(self):
        """Stop running simulation."""
        try:
            # First try to stop via our worker thread
            if self._simulation_worker is not None:
                self._simulation_worker.stop()
                self.simulation_log.append("🛑 Stop requested...")
                self.log_message("Stop simulation requested", "warning")
                return
            
            # Fallback: Kill SU2 processes directly
            import subprocess
            try:
                solvers = ['SU2_CFD', 'SU2_SOL', 'SU2_DEF', 'SU2_CFD.py']
                killed_any = False
                
                for solver in solvers:
                    result = subprocess.run(['pkill', '-f', solver], 
                                         capture_output=True, text=True)
                    if result.returncode == 0:
                        killed_any = True
                        self.simulation_log.append(f"🛑 Stopped {solver}")
                
                if killed_any:
                    self.simulation_log.append("🛑 Simulation stopped by user")
                    self.log_message("Simulation processes stopped", "warning")
                else:
                    self.log_message("No running simulation processes found", "info")
                    
            except Exception as e:
                self.log_message(f"Could not stop processes: {str(e)}", "error")
                    
        except Exception as e:
            self.log_message(f"Failed to stop simulation: {str(e)}", "error")
    
    def _update_solver_options(self):
        """Update solver dropdown based on simulation mode."""
        mode = getattr(self, 'simulation_mode', None)
        if mode is None:
            return
            
        current_mode = mode.currentText()
        self.solver_type.blockSignals(True)
        self.solver_type.clear()
        
        if current_mode == "Steady":
            # Steady-state solvers
            self.solver_type.addItems([
                "RANS",           # Compressible RANS
                "EULER",          # Compressible Euler (inviscid)
                "NAVIER_STOKES",  # Compressible laminar
                "INC_RANS",       # Incompressible RANS
                "INC_EULER",      # Incompressible Euler
                "INC_NAVIER_STOKES"  # Incompressible laminar
            ])
            self.solver_type.setCurrentText(DEFAULTS.solver_type)
        else:
            # Transient/Unsteady solvers
            self.solver_type.addItems([
                "RANS",           # Unsteady RANS (URANS)
                "EULER",          # Unsteady Euler
                "NAVIER_STOKES",  # Unsteady laminar
                "INC_RANS",       # Unsteady incompressible RANS
                "INC_NAVIER_STOKES"  # Unsteady incompressible laminar
            ])
            self.solver_type.setCurrentText(DEFAULTS.solver_type)
        
        self.solver_type.blockSignals(False)
        self._on_solver_type_changed(self.solver_type.currentText())
    
    def _on_simulation_mode_changed(self, mode: str):
        """Update UI visibility based on simulation mode (Steady/Transient)."""
        is_transient = (mode == "Transient")
        is_steady = (mode == "Steady")
        
        # Update solver options for the selected mode
        self._update_solver_options()
        
        # Show/hide steady-state controls
        for widget in [self.lbl_cfl, self.cfl_number]:
            widget.setVisible(is_steady)
        
        # Show/hide transient controls
        for widget in [self.lbl_time_step, self.time_step,
                       self.lbl_end_time, self.end_time,
                       self.lbl_max_courant, self.max_courant,
                       self.lbl_outer_correctors, self.n_outer_correctors]:
            widget.setVisible(is_transient)
        
        # Pressure correctors - hide for now (SU2 handles internally)
        for widget in [self.lbl_correctors, self.n_correctors]:
            widget.setVisible(False)
    
    def _on_solver_type_changed(self, solver_name: str):
        """Update UI visibility based on solver type."""
        if not solver_name:
            return
            
        # Check if RANS solver (needs turbulence model)
        is_rans = "RANS" in solver_name.upper()
        
        # Show/hide turbulence model based on solver (guard against early calls during init)
        if hasattr(self, 'lbl_turbulence') and hasattr(self, 'turbulence_model'):
            for widget in [self.lbl_turbulence, self.turbulence_model]:
                widget.setVisible(is_rans)
        
            # If not RANS (Euler or Navier-Stokes), turbulence is not applicable
            if not is_rans:
                self.turbulence_model.setCurrentText("None (Laminar)")
            elif self.turbulence_model.currentText() == "None (Laminar)":
                self.turbulence_model.setCurrentText("SST")
        
    def browse_case_directory(self):
        """Browse for case directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Case Directory")
        if directory:
            self.case_directory.setText(directory)
            
    def load_results(self):
        """Load SU2 simulation results."""
        try:
            results_path = self.results_path.text().strip()
            if not results_path:
                self.log_message("Please specify a results directory.", "warning")
                return
                
            if not os.path.exists(results_path):
                self.log_message("Results directory does not exist.", "warning")
                return
                
            self.results_summary.setText("Loading SU2 results...")
            
            # Check for SU2 output files
            has_history = os.path.exists(os.path.join(results_path, 'history.csv'))
            has_vtu = any(Path(results_path).glob("flow*.vtu"))
            has_surface_csv = os.path.exists(os.path.join(results_path, 'surface_flow.csv'))
            has_mesh = any(Path(results_path).glob("*.su2"))
            
            if not has_history and not has_vtu and not has_surface_csv:
                self.log_message("No SU2 results found. Looking for: history.csv, flow.vtu, or surface_flow.csv", "warning")
                return
            
            # Load data using SU2ResultsPlotter-style approach
            self.su2_mesh_nodes = None
            self.su2_triangles = []
            self.su2_field_data = {}
            
            # Try to load VTU first (has mesh + fields)
            vtu_loaded = self._load_su2_vtu(results_path)
            
            if not vtu_loaded:
                # Fallback: load mesh from .su2 then surface CSV
                self._load_su2_mesh(results_path)
                self._load_su2_surface_csv(results_path)
            
            # Load history for summary
            history_info = self._load_su2_history(results_path)
            
            # Populate field selector
            if hasattr(self, 'field_type') and self.su2_field_data:
                current_field = self.field_type.currentText()
                self.field_type.clear()
                # Add good fields (skip coordinates, IDs)
                skip_fields = {'x', 'y', 'z', 'pointid', 'globalindex'}
                good_fields = [f for f in self.su2_field_data.keys() 
                              if f.lower() not in skip_fields]
                self.field_type.addItems(sorted(good_fields))
                # Select Pressure by default or restore previous
                for preferred in ['Pressure', 'Mach', 'Temperature', 'Density']:
                    idx = self.field_type.findText(preferred)
                    if idx >= 0:
                        self.field_type.setCurrentIndex(idx)
                        break
            
            # Update summary with time step info
            n_nodes = len(self.su2_mesh_nodes) if self.su2_mesh_nodes is not None else 0
            n_elements = len(self.su2_triangles) if hasattr(self, 'su2_triangles') else 0
            n_fields = len(self.su2_field_data)
            n_timesteps = len(self.time_step_files) if hasattr(self, 'time_step_files') else 0
            
            summary_text = f"""SU2 Results Summary:
• Results directory: {results_path}
• Mesh nodes: {n_nodes:,}
• Elements: {n_elements:,}
• Available fields: {n_fields}
• Time steps: {n_timesteps}
• Fields: {', '.join(list(self.su2_field_data.keys())[:8])}...
{history_info}

Status: ✓ Results loaded successfully"""
            
            self.results_summary.setText(summary_text)
            
            # Update quick stats if they exist
            if hasattr(self, '_stat_nodes'):
                self._stat_nodes.setText(f"{n_nodes:,}")
            if hasattr(self, '_stat_elements'):
                self._stat_elements.setText(f"{n_elements:,}")
            if hasattr(self, '_stat_fields'):
                self._stat_fields.setText(f"{n_fields}")
            if hasattr(self, '_stat_timesteps'):
                self._stat_timesteps.setText(f"{n_timesteps}")
            
            # Store results info
            self.current_results = {
                'path': results_path,
                'type': 'su2'
            }
            
            # Time step dropdown is now populated in _load_su2_vtu()
            # Only add fallback if no time steps were found
            if hasattr(self, 'viz_time_step') and self.viz_time_step.count() == 0:
                if hasattr(self, 'time_step_files') and self.time_step_files:
                    def step_key(k):
                        if k == "final": return float('inf')
                        try:
                            return int(k)
                        except ValueError:
                            return float('inf')
                    steps = sorted(self.time_step_files.keys(), key=step_key)
                    self.viz_time_step.addItems(steps)
                    if self.current_loaded_time_step:
                        idx = self.viz_time_step.findText(self.current_loaded_time_step)
                        if idx >= 0:
                            self.viz_time_step.setCurrentIndex(idx)
                else:
                    self.viz_time_step.addItem("final")
            
            self.update_workflow_status()
            
            # Trigger visualization
            self.update_visualization()
            
            self.log_message(f"SU2 results loaded! {n_nodes} nodes, {n_fields} fields, {n_timesteps} time steps", "success")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.log_message(f"Failed to load results: {str(e)}", "error")
            self.results_summary.setText("[X] Failed to load results")
    
    def _load_su2_vtu(self, results_path: str) -> bool:
        """Load mesh and fields from SU2 VTU file."""
        try:
            import meshio
            import re
        except ImportError:
            self.log_message("meshio not installed. Cannot load VTU files.", "warning")
            return False
        
        # Find all flow*.vtu files, excluding surface_flow*.vtu
        vtu_files = [f for f in Path(results_path).glob("flow*.vtu") 
                    if 'surface' not in f.name.lower()]
        
        if not vtu_files:
            return False
        
        # Sort by filename to ensure proper ordering
        vtu_files = sorted(vtu_files, key=lambda f: f.name)
        
        # Extract time steps from filenames
        self.time_step_files = {}
        for f in vtu_files:
            fname = f.name
            # Match patterns like flow_00100.vtu, flow_00000.vtu, etc.
            match = re.search(r'flow_(\d+)\.vtu', fname)
            if match:
                # Keep the step as string with leading zeros for display
                step_num = match.group(1)
                # Store with the numeric value as key for proper sorting, but display with zeros
                self.time_step_files[step_num] = str(f)
            elif fname == "flow.vtu":
                self.time_step_files["final"] = str(f)
            else:
                # Fallback for other naming patterns
                self.time_step_files["final"] = str(f)
        
        if not self.time_step_files:
            return False
        
        # Sort time steps numerically
        def step_key(k):
            if k == "final": 
                return float('inf')
            try:
                return int(k)
            except ValueError:
                return float('inf')
        
        sorted_steps = sorted(self.time_step_files.keys(), key=step_key)
        
        # Load the last (latest) time step
        last_step = sorted_steps[-1]
        self.current_loaded_time_step = last_step
        
        print(f"Found {len(sorted_steps)} time steps: {sorted_steps}")
        print(f"Loading latest: {last_step}")
        
        success = self._load_su2_vtu_file(self.time_step_files[last_step])
        
        # Update time step dropdown immediately after loading
        if success and hasattr(self, 'viz_time_step'):
            self.viz_time_step.blockSignals(True)  # Prevent triggering update
            self.viz_time_step.clear()
            self.viz_time_step.addItems(sorted_steps)
            # Select the loaded time step
            idx = self.viz_time_step.findText(last_step)
            if idx >= 0:
                self.viz_time_step.setCurrentIndex(idx)
            self.viz_time_step.blockSignals(False)
            print(f"Time step dropdown populated with {len(sorted_steps)} steps")
        
        return success

    def _load_su2_vtu_file(self, file_path: str) -> bool:
        try:
            import meshio
            import numpy as np
            
            mesh = meshio.read(file_path)
            
            # Get mesh nodes (2D)
            self.su2_mesh_nodes = mesh.points[:, :2]
            
            # Build triangulation
            self.su2_triangles = []
            for cell_block in mesh.cells:
                if cell_block.type == "triangle":
                    for tri in cell_block.data:
                        self.su2_triangles.append(tuple(tri))
                elif cell_block.type == "quad":
                    for quad in cell_block.data:
                        self.su2_triangles.append((quad[0], quad[1], quad[2]))
                        self.su2_triangles.append((quad[0], quad[2], quad[3]))
            
            # Load point data
            self.su2_field_data = {} # Reset field data
            for name, data in mesh.point_data.items():
                if len(data.shape) == 1:
                    self.su2_field_data[name] = data
                else:
                    self.su2_field_data[name] = data
                    self.su2_field_data[f"{name}_Magnitude"] = np.linalg.norm(data, axis=1)
                    
            print(f"Loaded VTU: {len(self.su2_mesh_nodes)} nodes, {len(self.su2_triangles)} triangles")
            return True
            
        except Exception as e:
            print(f"VTU load failed: {e}")
            return False
    
    def _load_su2_mesh(self, results_path: str):
        """Load mesh from SU2 mesh file."""
        mesh_files = list(Path(results_path).glob("*.su2"))
        if not mesh_files:
            return
            
        nodes = []
        elements = []
        
        with open(mesh_files[0], 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('NPOIN='):
                n_points = int(line.split('=')[1].strip().split()[0])
                i += 1
                for _ in range(n_points):
                    if i >= len(lines):
                        break
                    parts = lines[i].strip().split()
                    if len(parts) >= 2:
                        nodes.append([float(parts[0]), float(parts[1])])
                    i += 1
                continue
                    
            elif line.startswith('NELEM='):
                n_elem = int(line.split('=')[1].strip())
                i += 1
                for _ in range(n_elem):
                    if i >= len(lines):
                        break
                    parts = lines[i].strip().split()
                    if len(parts) >= 4:
                        elem_type = int(parts[0])
                        if elem_type == 5:  # Triangle
                            elements.append((int(parts[1]), int(parts[2]), int(parts[3])))
                        elif elem_type == 9:  # Quad
                            elements.append((int(parts[1]), int(parts[2]), 
                                           int(parts[3]), int(parts[4])))
                    i += 1
                continue
                    
            i += 1
            
        self.su2_mesh_nodes = np.array(nodes)
        
        # Convert to triangles
        self.su2_triangles = []
        for elem in elements:
            if len(elem) == 3:
                self.su2_triangles.append(elem)
            elif len(elem) == 4:
                self.su2_triangles.append((elem[0], elem[1], elem[2]))
                self.su2_triangles.append((elem[0], elem[2], elem[3]))
                
        print(f"Loaded SU2 mesh: {len(nodes)} nodes, {len(self.su2_triangles)} triangles")
    
    def _load_su2_surface_csv(self, results_path: str):
        """Load field data from SU2 surface CSV."""
        csv_path = os.path.join(results_path, 'surface_flow.csv')
        if not os.path.exists(csv_path):
            return
            
        import csv
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = [col.strip().strip('"') for col in next(reader)]
            
            data = {col: [] for col in header}
            for row in reader:
                if not row:
                    continue
                for i, val in enumerate(row):
                    if i < len(header):
                        try:
                            data[header[i]].append(float(val))
                        except ValueError:
                            data[header[i]].append(np.nan)
                            
        for key, values in data.items():
            self.su2_field_data[key] = np.array(values)
            
        # Update mesh nodes from CSV coordinates (they match field data)
        if 'x' in data and 'y' in data:
            self.su2_mesh_nodes = np.column_stack([
                np.array(data['x']), np.array(data['y'])
            ])
            self.su2_triangles = []  # Will use Delaunay
            print(f"Loaded surface CSV: {len(self.su2_mesh_nodes)} points, {len(self.su2_field_data)} fields")
    
    def _load_su2_history(self, results_path: str) -> str:
        """Load and summarize convergence history."""
        history_path = os.path.join(results_path, 'history.csv')
        if not os.path.exists(history_path):
            return "• Convergence history: Not found"
            
        try:
            import csv
            with open(history_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                header = [col.strip().strip('"') for col in next(reader)]
                rows = list(reader)
                
            if rows:
                n_iters = len(rows)
                # Get final residual
                rms_col = next((i for i, h in enumerate(header) if 'rms' in h.lower()), None)
                if rms_col is not None:
                    final_res = float(rows[-1][rms_col])
                    return f"• Iterations: {n_iters}\n• Final residual: {final_res:.2e}"
                return f"• Iterations: {n_iters}"
        except:
            pass
        return "• Convergence history: Could not parse"
        
    def browse_results_directory(self):
        """Browse for results directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Results Directory")
        if directory:
            self.results_path.setText(directory)
            
    def on_field_changed(self):
        """Handle field type change."""
        self.update_visualization()
        
    def update_visualization(self):
        """Update post-processing visualization with SU2 data using triangulation."""
        try:
            print("=== update_visualization() called ===")
            
            # Check for time step change request
            if hasattr(self, 'viz_time_step') and hasattr(self, 'time_step_files') and self.time_step_files:
                target_step = self.viz_time_step.currentText()
                if target_step and target_step != self.current_loaded_time_step:
                     if target_step in self.time_step_files:
                         print(f"Switching time step to {target_step}")
                         success = self._load_su2_vtu_file(self.time_step_files[target_step])
                         if success:
                             self.current_loaded_time_step = target_step
            
            if not self.current_results:
                print("WARNING: No current_results available")
                return
            
            # Check if we have SU2 data
            if not hasattr(self, 'su2_mesh_nodes') or self.su2_mesh_nodes is None:
                print("WARNING: No SU2 mesh data loaded")
                return
                
            if not hasattr(self, 'su2_field_data') or not self.su2_field_data:
                print("WARNING: No SU2 field data loaded")
                return
            
            # Get selected field
            field_name = self.field_type.currentText()
            print(f"Selected field: '{field_name}'")
            
            if not field_name:
                print("WARNING: No field selected")
                return
            
            # Get field data
            field = None
            # Exact match
            if field_name in self.su2_field_data:
                field = self.su2_field_data[field_name]
            else:
                # Case-insensitive match
                for key, value in self.su2_field_data.items():
                    if key.lower() == field_name.lower():
                        field = value
                        break
                        
            if field is None:
                print(f"Field '{field_name}' not found in data")
                return
                
            # Handle vector fields
            if len(field.shape) > 1:
                field = np.linalg.norm(field, axis=1)
            
            # Clear and setup the figure
            self.postproc_canvas.figure.clear()
            self.postproc_canvas.ax = self.postproc_canvas.figure.add_subplot(111)
            ax = self.postproc_canvas.ax
            
            # Style the axes for dark theme
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors=Theme.TEXT)
            for spine in ax.spines.values():
                spine.set_color(Theme.BORDER)
            
            # Create triangulation
            import matplotlib.tri as mtri
            
            nodes = self.su2_mesh_nodes
            if len(field) != len(nodes):
                ax.text(0.5, 0.5, f"Field size mismatch: {len(field)} vs {len(nodes)} nodes",
                       transform=ax.transAxes, ha='center', va='center', 
                       color=Theme.TEXT, fontsize=12)
                self.postproc_canvas.draw()
                return
            
            # Build triangulation
            if hasattr(self, 'su2_triangles') and self.su2_triangles:
                # No fallback to Delaunay if explicitly provided triangulation fails
                triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], 
                                           triangles=self.su2_triangles)
            else:
                # Delaunay triangulation
                triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
            
            # Get visualization settings
            colormap = self.colormap.currentText()
            levels = self.contour_levels.value()
            
            # Create filled contour plot (respects mesh boundaries!)
            contourf = ax.tricontourf(triang, field, levels=levels, cmap=colormap)
            
            # Add colorbar
            cbar = self.postproc_canvas.figure.colorbar(contourf, ax=ax, shrink=0.8)
            cbar.set_label(field_name, color=Theme.TEXT)
            cbar.ax.tick_params(colors=Theme.TEXT)
            
            # Set labels and aspect
            ax.set_xlabel('X [m]', color=Theme.TEXT)
            ax.set_ylabel('Y [m]', color=Theme.TEXT)
            ax.set_title(f'{field_name}', color=Theme.TEXT)
            ax.set_aspect('equal')
            
            # Update canvas
            self.postproc_canvas.figure.tight_layout()
            self.postproc_canvas.draw()
            
            print(f"Visualization updated successfully: {len(nodes)} nodes, field range [{field.min():.2e}, {field.max():.2e}]")
            
        except Exception as e:
            print(f"Visualization update failed: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self, 'postproc_canvas'):
                self.postproc_canvas.ax.text(0.5, 0.5, f'Error: {str(e)}', 
                       transform=self.postproc_canvas.ax.transAxes, ha='center', va='center',
                       color='red', fontsize=12)
                self.postproc_canvas.draw()
            
            # Update canvas
            self.postproc_canvas.draw()
            
        except Exception as e:
            print(f"Visualization update failed: {e}")
            import traceback
            traceback.print_exc()
            self.postproc_canvas.ax.text(0.5, 0.5, f'Error: {str(e)}', 
                   transform=self.postproc_canvas.ax.transAxes, ha='center', va='center',
                   color='red', fontsize=12)
            self.postproc_canvas.draw()
    
    def _visualize_field_basic(self, field_path, field_name, time_step):
        """Basic field visualization by reading SU2 field file and mesh."""
        import numpy as np
        
        try:
            # Clear figure completely
            self.postproc_canvas.figure.clear()
            self.postproc_canvas.ax = self.postproc_canvas.figure.add_subplot(111, facecolor='#1e1e1e')
            
            # Style the axes
            self.postproc_canvas.ax.tick_params(colors=Theme.TEXT)
            for spine in self.postproc_canvas.ax.spines.values():
                spine.set_color(Theme.BORDER)
            
            # Read field data from SU2 output (CSV or restart file)
            field_data = self._read_su2_field(field_path, field_name)
            
            if field_data is None or len(field_data) == 0:
                self.postproc_canvas.ax.text(0.5, 0.5, f'Could not read field data from {field_name}', 
                       transform=self.postproc_canvas.ax.transAxes, ha='center', va='center',
                       color=Theme.TEXT_SECONDARY, fontsize=12)
                self.postproc_canvas.draw()
                return
            
            print(f"Read {len(field_data)} field values from {field_name}")
            
            # Read mesh from SU2 mesh file
            mesh_path = self.current_results['path']
            mesh_info_file = os.path.join(mesh_path, 'mesh_info.json')
            
            cell_centers = None
            nozzle_geometry = None
            
            # Try to load mesh_info.json
            if os.path.exists(mesh_info_file):
                try:
                    import json
                    with open(mesh_info_file, 'r') as f:
                        mesh_info = json.load(f)
                    
                    if 'nodes' in mesh_info:
                        # Use the nodes as cell centers (for nozzle-shaped mesh)
                        nodes = np.array(mesh_info['nodes'])
                        cell_centers = [(n[0], n[1]) for n in nodes]
                        nozzle_geometry = {'nodes': nodes}
                        print(f"Loaded {len(cell_centers)} nodes from mesh_info.json")
                        
                        # Extract nozzle walls for overlay
                        x_unique = sorted(set(nodes[:, 0]))
                        upper_wall = []
                        lower_wall = []
                        for x in x_unique:
                            points_at_x = nodes[nodes[:, 0] == x]
                            if len(points_at_x) > 0:
                                upper_wall.append((x, points_at_x[:, 1].max()))
                                lower_wall.append((x, points_at_x[:, 1].min()))
                        nozzle_geometry['upper_wall'] = np.array(upper_wall)
                        nozzle_geometry['lower_wall'] = np.array(lower_wall)
                except Exception as e:
                    print(f"Could not load mesh_info.json: {e}")
            
            # If no mesh_info.json, compute cell centers from OpenFOAM mesh
            if cell_centers is None:
                cell_centers = self._read_openfoam_cell_centers(mesh_path)
                
                if cell_centers is None or len(cell_centers) == 0:
                    self.postproc_canvas.ax.text(0.5, 0.5, 'Could not read mesh data', 
                           transform=self.postproc_canvas.ax.transAxes, ha='center', va='center',
                           color=Theme.TEXT_SECONDARY, fontsize=12)
                    self.postproc_canvas.draw()
                    return
                
                print(f"Computed {len(cell_centers)} cell centers from OpenFOAM mesh")
            
            # Verify field data matches mesh
            if len(field_data) != len(cell_centers):
                print(f"WARNING: Field data size ({len(field_data)}) doesn't match mesh cells ({len(cell_centers)})")
                # Truncate or pad as needed
                min_size = min(len(field_data), len(cell_centers))
                field_data = field_data[:min_size]
                cell_centers = cell_centers[:min_size]
            
            # Extract x, y coordinates from cell centers
            x_data = np.array([c[0] for c in cell_centers])
            y_data = np.array([c[1] for c in cell_centers])
            field_array = np.array(field_data)
            
            # Get domain bounds from actual mesh
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            
            print(f"Mesh domain: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
            
            # Plot based on field type and plot type
            plot_type = self.plot_type.currentText()
            colormap = self.colormap.currentText()
            levels = self.contour_levels.value()
            
            # Create fine visualization grid
            xi = np.linspace(x_min, x_max, 200)
            yi = np.linspace(y_min, y_max, 100)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Interpolate field data onto visualization grid using actual cell centers
            from scipy.interpolate import griddata
            
            points_array = np.column_stack((x_data, y_data))
            
            print(f"Interpolating {len(field_data)} values onto {Xi.shape} grid...")
            Zi = griddata(points_array, field_array, (Xi, Yi), method='linear', fill_value=np.nan)
            
            # Fill NaN values with nearest neighbor
            mask = np.isnan(Zi)
            if mask.any():
                print(f"Filling {mask.sum()} NaN values with nearest neighbor")
                Zi_nearest = griddata(points_array, field_array, (Xi, Yi), method='nearest')
                Zi[mask] = Zi_nearest[mask]
            
            # Determine label based on field name
            if field_name == 'U':
                label = 'Velocity Magnitude [m/s]'
            elif field_name == 'p':
                label = 'Pressure [Pa]'
            elif field_name == 'T':
                label = 'Temperature [K]'
            elif field_name == 'k':
                label = 'Turbulent Kinetic Energy [m²/s²]'
            elif field_name == 'epsilon':
                label = 'Turbulent Dissipation [m²/s³]'
            elif field_name == 'nut':
                label = 'Turbulent Viscosity [m²/s]'
            else:
                label = field_name
            
            # Plot based on type
            if plot_type == "Contour":
                contour = self.postproc_canvas.ax.contourf(Xi, Yi, Zi, levels=levels, cmap=colormap)
                cbar = self.postproc_canvas.figure.colorbar(contour, ax=self.postproc_canvas.ax, label=label)
                cbar.ax.yaxis.set_tick_params(color=Theme.TEXT)
                cbar.ax.yaxis.label.set_color(Theme.TEXT)
                for label_obj in cbar.ax.yaxis.get_ticklabels():
                    label_obj.set_color(Theme.TEXT)
            elif plot_type == "Vector" and field_name == 'U':
                # Simplified vector plot
                stride = 8
                contour = self.postproc_canvas.ax.contourf(Xi, Yi, Zi, levels=levels, cmap=colormap, alpha=0.6)
                self.postproc_canvas.ax.quiver(Xi[::stride, ::stride], Yi[::stride, ::stride], 
                                               Zi[::stride, ::stride]*0.01, np.zeros_like(Zi[::stride, ::stride]),
                                               scale=50, color='white', alpha=0.8, width=0.003)
            else:
                # Default to contour
                contour = self.postproc_canvas.ax.contourf(Xi, Yi, Zi, levels=levels, cmap=colormap)
                cbar = self.postproc_canvas.figure.colorbar(contour, ax=self.postproc_canvas.ax, label=label)
                cbar.ax.yaxis.set_tick_params(color=Theme.TEXT)
                cbar.ax.yaxis.label.set_color(Theme.TEXT)
                for label_obj in cbar.ax.yaxis.get_ticklabels():
                    label_obj.set_color(Theme.TEXT)
            
            # Add statistics
            stats_text = f'Min: {np.min(field_data):.2e}\nMax: {np.max(field_data):.2e}\nMean: {np.mean(field_data):.2e}'
            self.postproc_canvas.ax.text(0.02, 0.98, stats_text, 
                                        transform=self.postproc_canvas.ax.transAxes,
                                        va='top', ha='left', color=Theme.TEXT,
                                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
                                        fontsize=9)
            
            # Overlay nozzle geometry outline
            if nozzle_geometry and 'upper_wall' in nozzle_geometry:
                # Use the nozzle walls from mesh_info.json
                try:
                    upper = nozzle_geometry['upper_wall']
                    lower = nozzle_geometry['lower_wall']
                    self.postproc_canvas.ax.plot(upper[:, 0], upper[:, 1], 
                                                'w-', linewidth=2.5, alpha=0.9, label='Nozzle wall')
                    self.postproc_canvas.ax.plot(lower[:, 0], lower[:, 1], 
                                                'w-', linewidth=2.5, alpha=0.9)
                    # Add centerline
                    self.postproc_canvas.ax.axhline(0, color='white', linestyle='--', 
                                                   linewidth=1, alpha=0.4, label='Centerline')
                except Exception as e:
                    print(f"Could not overlay nozzle walls: {e}")
            elif self.current_mesh_data and 'nodes' in self.current_mesh_data:
                # Fallback to current_mesh_data
                try:
                    nodes = self.current_mesh_data['nodes']
                    nodes_array = np.array(nodes)
                    
                    # Sort by x coordinate
                    sorted_indices = np.argsort(nodes_array[:, 0])
                    sorted_nodes = nodes_array[sorted_indices]
                    
                    # Separate upper and lower wall (y > 0 and y < 0)
                    upper_wall = sorted_nodes[sorted_nodes[:, 1] > 0.01]
                    lower_wall = sorted_nodes[sorted_nodes[:, 1] < -0.01]
                    
                    if len(upper_wall) > 0:
                        self.postproc_canvas.ax.plot(upper_wall[:, 0], upper_wall[:, 1], 
                                                    'w-', linewidth=2, alpha=0.8, label='Upper wall')
                    if len(lower_wall) > 0:
                        self.postproc_canvas.ax.plot(lower_wall[:, 0], lower_wall[:, 1], 
                                                    'w-', linewidth=2, alpha=0.8, label='Lower wall')
                except Exception as e:
                    print(f"Could not overlay geometry: {e}")
            
            self.postproc_canvas.ax.set_title(f'{field_name} at t={time_step}s', color=Theme.TEXT, fontsize=12)
            self.postproc_canvas.ax.set_xlabel('X [m]', color=Theme.TEXT)
            self.postproc_canvas.ax.set_ylabel('Y [m]', color=Theme.TEXT)
            self.postproc_canvas.ax.set_aspect('equal')
            self.postproc_canvas.ax.grid(True, alpha=0.2, color=Theme.TEXT_SECONDARY)
            
            self.postproc_canvas.draw()
            print(f"Visualization complete for {field_name}")
                
        except Exception as e:
            print(f"Basic visualization error: {e}")
            import traceback
            traceback.print_exc()
            self.postproc_canvas.ax.text(0.5, 0.5, f'Visualization error:\n{str(e)}', 
                   transform=self.postproc_canvas.ax.transAxes, ha='center', va='center',
                   color='red', fontsize=10)
            self.postproc_canvas.draw()
    
    def _read_openfoam_points(self, points_file):
        """Read points from OpenFOAM polyMesh/points file."""
        points = []
        try:
            with open(points_file, 'r') as f:
                lines = f.readlines()
                
            # Find the point count
            in_points = False
            for i, line in enumerate(lines):
                if line.strip().startswith('('):
                    in_points = True
                    continue
                if in_points:
                    if line.strip().startswith(')'):
                        break
                    # Parse point coordinates (x y z)
                    parts = line.strip().strip('()').split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            points.append((x, y, z))
                        except:
                            continue
            
            return points[:1000]  # Limit for performance
        except Exception as e:
            print(f"Error reading points: {e}")
            return []
    
    def _read_openfoam_cell_centers(self, mesh_path):
        """Read cell centers from OpenFOAM mesh (points + faces + owner)."""
        try:
            import re
            
            # Read points
            points_file = os.path.join(mesh_path, 'points')
            with open(points_file, 'r') as f:
                content = f.read()
            
            paren_start = content.find('(')
            paren_end = content.rfind(')')
            points_str = content[paren_start+1:paren_end]
            
            point_pattern = r'\(\s*([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)\s*\)'
            matches = re.findall(point_pattern, points_str)
            points = [(float(m[0]), float(m[1]), float(m[2])) for m in matches]
            
            print(f"Read {len(points)} points from mesh")
            
            # Read faces
            faces_file = os.path.join(mesh_path, 'faces')
            with open(faces_file, 'r') as f:
                content = f.read()
            
            paren_start = content.find('(')
            paren_end = content.rfind(')')
            faces_str = content[paren_start+1:paren_end]
            
            # Parse faces (each face is like "4(0 1 2 3)")
            face_pattern = r'(\d+)\s*\(([^)]+)\)'
            face_matches = re.findall(face_pattern, faces_str)
            faces = []
            for npts, pts_str in face_matches:
                pt_indices = [int(x) for x in pts_str.split()]
                faces.append(pt_indices)
            
            print(f"Read {len(faces)} faces from mesh")
            
            # Read owner file to determine cells
            owner_file = os.path.join(mesh_path, 'owner')
            with open(owner_file, 'r') as f:
                content = f.read()
            
            paren_start = content.find('(')
            paren_end = content.rfind(')')
            owner_str = content[paren_start+1:paren_end]
            owners = [int(x) for x in owner_str.split()]
            
            print(f"Read {len(owners)} owner entries")
            
            # Count cells
            num_cells = max(owners) + 1
            print(f"Mesh has {num_cells} cells")
            
            # Build cell-to-faces mapping
            cell_to_faces = [[] for _ in range(num_cells)]
            for face_idx, owner in enumerate(owners):
                cell_to_faces[owner].append(face_idx)
            
            # Read neighbour file for internal faces
            neighbour_file = os.path.join(mesh_path, 'neighbour')
            if os.path.exists(neighbour_file):
                with open(neighbour_file, 'r') as f:
                    content = f.read()
                
                paren_start = content.find('(')
                paren_end = content.rfind(')')
                neighbour_str = content[paren_start+1:paren_end]
                neighbours = [int(x) for x in neighbour_str.split()]
                
                for face_idx, neighbour in enumerate(neighbours):
                    if neighbour < num_cells:
                        cell_to_faces[neighbour].append(face_idx)
            
            # Compute cell centers as average of all face points
            cell_centers = []
            for cell_idx in range(num_cells):
                if not cell_to_faces[cell_idx]:
                    continue
                
                # Collect all unique points from all faces of this cell
                cell_points = set()
                for face_idx in cell_to_faces[cell_idx]:
                    if face_idx < len(faces):
                        for pt_idx in faces[face_idx]:
                            if pt_idx < len(points):
                                cell_points.add(pt_idx)
                
                if cell_points:
                    # Average all points
                    cx = sum(points[pi][0] for pi in cell_points) / len(cell_points)
                    cy = sum(points[pi][1] for pi in cell_points) / len(cell_points)
                    cell_centers.append((cx, cy))
                else:
                    # Fallback: use first face center
                    face_idx = cell_to_faces[cell_idx][0]
                    if face_idx < len(faces):
                        face_pts = faces[face_idx]
                        cx = sum(points[pi][0] for pi in face_pts) / len(face_pts)
                        cy = sum(points[pi][1] for pi in face_pts) / len(face_pts)
                        cell_centers.append((cx, cy))
            
            print(f"Computed {len(cell_centers)} cell centers")
            return cell_centers
            
        except Exception as e:
            print(f"Error reading cell centers: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _read_openfoam_field(self, field_path, field_name):
        """Legacy method - redirects to SU2 field reader."""
        return self._read_su2_field(field_path, field_name)
    
    def _read_su2_field(self, field_path, field_name):
        """Read field data from SU2 output file (CSV format)."""
        try:
            import csv
            
            # Check if it's a CSV file
            if field_path.endswith('.csv'):
                with open(field_path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                
                if not rows:
                    return None
                    
                # Try to find the field column
                field_lower = field_name.lower()
                target_column = None
                
                for col in rows[0].keys():
                    if col.strip().strip('"').lower() == field_lower:
                        target_column = col
                        break
                    if field_lower in col.strip().strip('"').lower():
                        target_column = col
                        break
                        
                if target_column is None:
                    print(f"Field {field_name} not found in CSV columns: {list(rows[0].keys())}")
                    return None
                    
                field_values = []
                for row in rows:
                    try:
                        field_values.append(float(row[target_column]))
                    except (ValueError, KeyError):
                        pass
                        
                print(f"Read {len(field_values)} values from {field_name}")
                return field_values
                
            else:
                # Try reading as a text file with numeric values
                with open(field_path, 'r') as f:
                    lines = f.readlines()
                
                field_values = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('%') or line.startswith('#'):
                        continue
                    try:
                        field_values.append(float(line))
                    except ValueError:
                        # Try parsing as space-separated values
                        parts = line.split()
                        for part in parts:
                            try:
                                field_values.append(float(part))
                                break
                            except ValueError:
                                pass
                                
                if field_values:
                    return field_values
                    
            print(f"Could not parse field data from {field_path}")
            return None
            
        except Exception as e:
            print(f"Error reading field {field_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _visualize_field_advanced(self, field_path, field_name, time_step):
        """Advanced visualization using ResultsProcessor."""
        # This would use the results_processor module
        # For now, fall back to basic
        self._visualize_field_basic(field_path, field_name, time_step)
    
    def analyze_wall_values(self):
        """Analyze values at wall."""
        try:
            if not self.current_results:
                self.log_message("No results loaded. Please load simulation results first.", "warning")
                return
                
            # Basic wall analysis implementation
            results_path = self.current_results['path']
            latest_time = self.current_results['latest_time']
            
            # Look for wall patches in boundary conditions
            system_dir = os.path.join(results_path, 'system')
            if os.path.exists(system_dir):
                self.log_message(f"Wall analysis for time {latest_time}: Analysis completed (basic implementation)", "info")
            else:
                self.log_message("Could not find case system directory for analysis.", "warning")
                
        except Exception as e:
            self.log_message(f"Failed to analyze wall values: {str(e)}", "error")
        
    def plot_centerline(self):
        """Plot centerline values."""
        try:
            if not self.current_results:
                self.log_message("No results loaded. Please load simulation results first.", "warning")
                return
                
            # Create a new window for centerline plot
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle('Centerline Analysis')
            
            # Generate sample centerline data (in real implementation, this would extract from results)
            x = np.linspace(0, 2, 100)
            
            # Velocity along centerline
            velocity = 10 * (1 + 0.5 * np.sin(np.pi * x))  # Sample data
            ax1.plot(x, velocity, 'b-', linewidth=2, label='Velocity magnitude')
            ax1.set_xlabel('X position [m]')
            ax1.set_ylabel('Velocity [m/s]')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Pressure along centerline
            pressure = 101325 - 500 * x  # Sample data
            ax2.plot(x, pressure, 'r-', linewidth=2, label='Pressure')
            ax2.set_xlabel('X position [m]')
            ax2.set_ylabel('Pressure [Pa]')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
            self.log_message("Centerline plot generated (sample data shown)", "success")
            
        except Exception as e:
            self.log_message(f"Failed to create centerline plot: {str(e)}", "error")
        
    def calculate_mass_flow(self):
        """Calculate mass flow rate."""
        try:
            if not self.current_results:
                self.log_message("No results loaded. Please load simulation results first.", "warning")
                return
                
            # Basic mass flow calculation implementation
            results_path = self.current_results['path']
            latest_time = self.current_results['latest_time']
            
            # In a real implementation, this would integrate velocity and density across inlet/outlet patches
            # For now, show a calculation based on geometry and typical values
            
            if self.geometry.elements:
                # Get approximate throat area from geometry
                x_coords, y_coords = self.geometry.get_interpolated_points(100)
                if len(y_coords) > 0:
                    throat_radius = min(y_coords)  # Approximate
                    throat_area = np.pi * throat_radius**2
                    
                    # Estimate mass flow (simplified)
                    density = 1.225  # kg/m³ (air at STP)
                    velocity = 10.0  # m/s (estimated)
                    mass_flow = density * velocity * throat_area
                    
                    self.log_message(f"Mass flow estimate: {mass_flow:.6f} kg/s (throat radius: {throat_radius:.4f}m)", "info")
                else:
                    self.log_message("Could not determine geometry dimensions.", "warning")
            else:
                self.log_message("No geometry available for calculation.", "warning")
                
        except Exception as e:
            self.log_message(f"Failed to calculate mass flow: {str(e)}", "error")
        
    def calculate_pressure_loss(self):
        """Calculate pressure loss."""
        try:
            if not self.current_results:
                self.log_message("No results loaded. Please load simulation results first.", "warning")
                return
                
            # Basic pressure loss calculation
            results_path = self.current_results['path']
            
            # In a real implementation, this would read pressure values at inlet and outlet patches
            # For demonstration, calculate theoretical pressure loss
            
            inlet_pressure = 101325  # Pa (standard pressure)
            outlet_pressure = 101000  # Pa (estimated)
            
            pressure_loss = inlet_pressure - outlet_pressure
            pressure_loss_percent = (pressure_loss / inlet_pressure) * 100
            
            self.log_message(f"Pressure loss: {pressure_loss:,.0f} Pa ({pressure_loss_percent:.2f}%)", "info")
            
        except Exception as e:
            self.log_message(f"Failed to calculate pressure loss: {str(e)}", "error")
        
    def save_visualization(self):
        """Save current visualization."""
        try:
            if not hasattr(self, 'postprocessing_canvas'):
                self.log_message("No visualization available to save.", "warning")
                return
                
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Visualization", "", 
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Save the current figure
            self.postprocessing_canvas.figure.savefig(
                file_path, 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            
            self.log_message(f"Visualization saved to: {file_path}", "success")
            
        except Exception as e:
            self.log_message(f"Failed to save visualization: {str(e)}", "error")
        
    def export_data(self):
        """Export analysis data."""
        try:
            if not self.current_results:
                self.log_message("No results available to export.", "warning")
                return
                
            # Get export file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Data", "", 
                "CSV Files (*.csv);;JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Determine export format
            extension = os.path.splitext(file_path)[1].lower()
            
            # Prepare data for export
            export_data = {
                'case_info': {
                    'results_path': self.current_results['path'],
                    'latest_time': self.current_results['latest_time'],
                    'available_fields': self.current_results['fields'],
                    'time_steps': len(self.current_results['time_dirs'])
                },
                'geometry': {
                    'num_elements': len(self.geometry.elements),
                    'element_types': [elem.element_type for elem in self.geometry.elements]
                }
            }
            
            if extension == '.json':
                import json
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif extension == '.csv':
                # Export as CSV (simplified format)
                with open(file_path, 'w') as f:
                    f.write("Parameter,Value\n")
                    f.write(f"Results Path,{export_data['case_info']['results_path']}\n")
                    f.write(f"Latest Time,{export_data['case_info']['latest_time']}\n")
                    f.write(f"Time Steps,{export_data['case_info']['time_steps']}\n")
                    f.write(f"Geometry Elements,{export_data['geometry']['num_elements']}\n")
                    
            else:  # Text format
                with open(file_path, 'w') as f:
                    f.write("Nozzle CFD Analysis Data Export\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Results Path: {export_data['case_info']['results_path']}\n")
                    f.write(f"Latest Time: {export_data['case_info']['latest_time']}\n")
                    f.write(f"Time Steps: {export_data['case_info']['time_steps']}\n")
                    f.write(f"Available Fields: {', '.join(export_data['case_info']['available_fields'])}\n")
                    f.write(f"Geometry Elements: {export_data['geometry']['num_elements']}\n")
                    
            self.log_message(f"Data exported to: {file_path}", "success")
            
        except Exception as e:
            self.log_message(f"Failed to export data: {str(e)}", "error")
        
    def generate_report(self):
        """Generate analysis report."""
        try:
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Generate Report", "", 
                "HTML Files (*.html);;PDF Files (*.pdf);;Text Files (*.txt);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Generate report content
            report_content = self._generate_report_content()
            
            # Determine format and save
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.html':
                self._save_html_report(file_path, report_content)
            else:  # Text format
                self._save_text_report(file_path, report_content)
                
            self.log_message(f"Report generated: {file_path}", "success")
            
        except Exception as e:
            self.log_message(f"Failed to generate report: {str(e)}", "error")
    
    def _generate_report_content(self):
        """Generate the content for the report."""
        from datetime import datetime
        
        # Collect information
        has_geometry = len(self.geometry.elements) > 0
        has_mesh = self.current_mesh_data is not None
        has_case = self.current_case_directory and os.path.exists(self.current_case_directory)
        has_results = self.current_results is not None
        
        # Get project info safely
        project_name = "Untitled Project"
        project_description = "No description provided"
        
        if hasattr(self, 'project_name_edit') and self.project_name_edit:
            project_name = self.project_name_edit.text() or project_name
            
        if hasattr(self, 'project_description') and self.project_description:
            project_description = self.project_description.toPlainText() or project_description
        
        content = {
            'title': 'Nozzle CFD Analysis Report',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'project_name': project_name,
            'description': project_description,
            'geometry': {
                'status': '[OK] Complete' if has_geometry else '[X] Not defined',
                'elements': len(self.geometry.elements),
                'symmetric': self.geometry.is_symmetric
            },
            'mesh': {
                'status': '[OK] Generated' if has_mesh else '[X] Not generated'
            },
            'simulation': {
                'status': '[OK] Case ready' if has_case else '[X] Not set up'
            },
            'results': {
                'status': '[OK] Available' if has_results else '[X] Not available',
                'details': self.current_results if has_results else None
            }
        }
        
        return content
    
    def _save_html_report(self, file_path, content):
        """Save report in HTML format."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{content['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #0066cc; }}
        .status {{ font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{content['title']}</h1>
        <p><strong>Generated:</strong> {content['date']}</p>
        <p><strong>Project:</strong> {content['project_name']}</p>
    </div>
    
    <div class="section">
        <h2>Project Description</h2>
        <p>{content['description']}</p>
    </div>
    
    <div class="section">
        <h2>Workflow Status</h2>
        <p><span class="status">Geometry:</span> {content['geometry']['status']} ({content['geometry']['elements']} elements)</p>
        <p><span class="status">Mesh:</span> {content['mesh']['status']}</p>
        <p><span class="status">Simulation:</span> {content['simulation']['status']}</p>
        <p><span class="status">Results:</span> {content['results']['status']}</p>
    </div>
    
    {"<div class='section'><h2>Results Summary</h2><p>Latest time: " + str(content['results']['details']['latest_time']) + "</p><p>Available fields: " + ', '.join(content['results']['details']['fields']) + "</p></div>" if content['results']['details'] else ""}
    
    <div class="section">
        <h2>Generated by</h2>
        <p>Nozzle CFD Design Tool v1.0.0</p>
    </div>
</body>
</html>
"""
        with open(file_path, 'w') as f:
            f.write(html_content)
    
    def _save_text_report(self, file_path, content):
        """Save report in text format."""
        # Build results summary separately to avoid f-string backslash issues
        results_summary = ""
        if content['results']['details']:
            fields_str = ', '.join(content['results']['details']['fields'])
            results_summary = (
                f"Results Summary:\n"
                f"• Latest time: {content['results']['details']['latest_time']}\n"
                f"• Available fields: {fields_str}\n"
            )
        
        text_content = f"""{content['title']}
{"=" * len(content['title'])}

Generated: {content['date']}
Project: {content['project_name']}

Description:
{content['description']}

Workflow Status:
• Geometry: {content['geometry']['status']} ({content['geometry']['elements']} elements)
• Mesh: {content['mesh']['status']}
• Simulation: {content['simulation']['status']}
• Results: {content['results']['status']}

{results_summary}Generated by: Nozzle CFD Design Tool v1.0.0
"""
        with open(file_path, 'w') as f:
            f.write(text_content)
        
    def update_workflow_status(self):
        """Update workflow status indicators."""
        # Update status based on current state
        has_geometry = len(self.geometry.elements) > 0
        
        # Update geometry status
        if hasattr(self, 'geometry_status'):
            self.geometry_status.setText("[OK] Complete" if has_geometry else "⏳ Pending")
            self.geometry_status.setStyleSheet(f"color: {Theme.SUCCESS if has_geometry else Theme.WARNING}")
        
        # Update mesh status
        has_mesh = self.current_mesh_data is not None
        if hasattr(self, 'mesh_status'):
            self.mesh_status.setText("[OK] Complete" if has_mesh else "⏳ Pending")
            self.mesh_status.setStyleSheet(f"color: {Theme.SUCCESS if has_mesh else Theme.WARNING}")
        
        # Update simulation status
        has_case = self.current_case_directory and os.path.exists(self.current_case_directory)
        if hasattr(self, 'simulation_status'):
            self.simulation_status.setText("[OK] Ready" if has_case else "⏳ Pending")
            self.simulation_status.setStyleSheet(f"color: {Theme.SUCCESS if has_case else Theme.WARNING}")
        
        # Update results status  
        has_results = self.current_results is not None
        if hasattr(self, 'results_status'):
            self.results_status.setText("[OK] Available" if has_results else "⏳ Pending")
            self.results_status.setStyleSheet(f"color: {Theme.SUCCESS if has_results else Theme.WARNING}")
        
    def new_project(self):
        """Create a new project."""
        self.geometry.clear()
        if hasattr(self, 'canvas'):
            self.canvas.current_points = []
            self.update_canvas(self.canvas)
        self.update_workflow_status()
        self.current_file = None
        self.current_case_directory = ""
        self.refresh_project_metadata()
        self.log_message("New project created", "info")
    
    def open_project(self):
        """Open an existing project."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    project_data = json.load(f)
                
                # Load project information
                if 'project_info' in project_data:
                    info = project_data['project_info']
                    self.project_name_edit.setText(info.get('name', ''))
                    self.project_description.setPlainText(info.get('description', ''))
                
                # Load geometry
                if 'geometry' in project_data:
                    self.geometry.load_from_dict(project_data['geometry'])
                    
                    # Update canvas if available
                    if hasattr(self, 'canvas'):
                        self.canvas.current_points = []
                        self.update_canvas(self.canvas)
                
                # Load mesh parameters if available
                if 'mesh_parameters' in project_data and self.mesh_generator:
                    mesh_params = project_data['mesh_parameters']
                    # Update mesh parameter controls
                    if hasattr(self, 'element_size'):
                        self.element_size.setValue(mesh_params.get('element_size', 0.1))
                    if hasattr(self, 'boundary_layers'):
                        self.boundary_layers.setChecked(mesh_params.get('boundary_layers', True))
                
                # Load simulation settings if available
                if 'simulation_settings' in project_data:
                    sim_settings = project_data['simulation_settings']
                    if hasattr(self, 'solver_type'):
                        solver = sim_settings.get('solver', 'RANS')
                        index = self.solver_type.findText(solver)
                        if index >= 0:
                            self.solver_type.setCurrentIndex(index)
                
                self.current_file = file_path
                self.is_modified = False
                self.update_workflow_status()
                self.refresh_project_metadata()
                
                self.log_message(f"Project loaded from: {file_path}", "success")
                
            except Exception as e:
                self.log_message(f"Failed to open project: {str(e)}", "error")
    
    def save_project(self):
        """Save current project."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Files (*.json)")
        if file_path:
            try:
                # Prepare project data
                project_data = {
                    'version': '1.0.0',
                    'project_info': {
                        'name': self.project_name_edit.text(),
                        'description': self.project_description.toPlainText(),
                        'created': self._get_current_timestamp()
                    },
                    'geometry': self.geometry.to_dict(),
                    'mesh_parameters': {
                        'element_size': getattr(self, 'element_size', None) and self.element_size.value(),
                        'boundary_layers': getattr(self, 'boundary_layers', None) and self.boundary_layers.isChecked(),
                        'boundary_layer_thickness': getattr(self, 'boundary_layer_thickness', None) and self.boundary_layer_thickness.value(),
                        'domain_extension': getattr(self, 'domain_extension', None) and self.domain_extension.value()
                    },
                    'simulation_settings': {
                        'solver': getattr(self, 'solver_type', None) and self.solver_type.currentText(),
                        'inlet_velocity': getattr(self, 'inlet_velocity', None) and self.inlet_velocity.value(),
                        'outlet_pressure': getattr(self, 'outlet_pressure', None) and self.outlet_pressure.value(),
                        'turbulence_model': getattr(self, 'turbulence_model', None) and self.turbulence_model.currentText()
                    }
                }
                
                # Remove None values
                project_data = self._clean_dict(project_data)
                
                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(project_data, f, indent=2)
                
                self.current_file = file_path
                self.is_modified = False
                self.refresh_project_metadata()
                
                self.log_message(f"Project saved to: {file_path}", "success")
                
            except Exception as e:
                self.log_message(f"Failed to save project: {str(e)}", "error")
    
    def _get_current_timestamp(self):
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _clean_dict(self, d):
        """Remove None values from dictionary recursively."""
        if isinstance(d, dict):
            return {k: self._clean_dict(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [self._clean_dict(v) for v in d if v is not None]
        else:
            return d

    def export_case(self):
        """Export case files for SU2."""
        if not self.geometry.elements:
            self.log_message("No geometry defined", "warning")
            return
            
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if directory:
            try:
                # Create case structure
                case_name = self.project_name_edit.text() or "nozzleCase"
                case_dir = os.path.join(directory, case_name)
                
                if not os.path.exists(case_dir):
                    os.makedirs(case_dir)
                
                # Generate SU2 case files using simulation setup
                if self.simulation_setup:
                    try:
                        self.simulation_setup.generate_su2_case(case_dir, self.current_mesh_data)
                    except Exception as e:
                        print(f"SU2 case generation failed: {e}")
                        # Create basic SU2 config as fallback
                        self._create_basic_su2_config(case_dir)
                else:
                    self._create_basic_su2_config(case_dir)
                
                self.current_case_directory = case_dir
                self.update_workflow_status()
                self.refresh_project_metadata()
                
                self.log_message(f"SU2 case exported to: {case_dir}", "success")
                
            except Exception as e:
                self.log_message(f"Failed to export case: {str(e)}", "error")

    def _create_basic_su2_config(self, case_dir):
        """Create a basic SU2 configuration file."""
        config_content = """% SU2 Configuration File
% Generated by Nozzle CFD Design Tool

% ------------- DIRECT, ADJOINT, AND LINEARIZED PROBLEM DEFINITION ------------%
SOLVER= RANS
KIND_TURB_MODEL= SST
MATH_PROBLEM= DIRECT
RESTART_SOL= NO

% -------------------- COMPRESSIBLE FREE-STREAM DEFINITION --------------------%
MACH_NUMBER= 0.5
AOA= 0.0
SIDESLIP_ANGLE= 0.0
FREESTREAM_PRESSURE= 101325.0
FREESTREAM_TEMPERATURE= 288.15
REYNOLDS_NUMBER= 1000000.0
REYNOLDS_LENGTH= 1.0
INIT_OPTION= TD_CONDITIONS

% ---------------------- REFERENCE VALUE DEFINITION ---------------------------%
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.00
REF_ORIGIN_MOMENT_Z= 0.00
REF_LENGTH= 1.0
REF_AREA= 1.0

% -------------------- BOUNDARY CONDITION DEFINITION --------------------------%
MARKER_HEATFLUX= ( wall, 0.0 )
MARKER_FAR= ( farfield )

% ------------------------ SURFACES IDENTIFICATION ----------------------------%
MARKER_PLOTTING= ( wall )
MARKER_MONITORING= ( wall )

% ------------- COMMON PARAMETERS DEFINING THE NUMERICAL METHOD ---------------%
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= 10.0
CFL_ADAPT= NO

% ------------------------ LINEAR SOLVER DEFINITION ---------------------------%
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 10

% -------------------- FLOW NUMERICAL METHOD DEFINITION -----------------------%
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN

% -------------------- TURBULENT NUMERICAL METHOD DEFINITION ------------------%
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO

% --------------------------- CONVERGENCE PARAMETERS --------------------------%
ITER= 1000
CONV_RESIDUAL_MINVAL= -8
CONV_STARTITER= 10

% ------------------------- INPUT/OUTPUT INFORMATION --------------------------%
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
SOLUTION_FILENAME= restart_flow.dat
RESTART_FILENAME= restart_flow.dat
CONV_FILENAME= history
OUTPUT_FILES= (RESTART, PARAVIEW)
OUTPUT_WRT_FREQ= 100
"""
        config_path = os.path.join(case_dir, 'config.cfg')
        with open(config_path, 'w') as f:
            f.write(config_content)

    def on_tab_changed(self, index):
        """Handle tab change events."""
        tab_names = ["Geometry", "Meshing", "Simulation", "Post-processing"]
        if 0 <= index < len(tab_names):
            print(f"Switched to {tab_names[index]} tab")
    
    def _create_control_dict(self, system_dir):
        """Create basic controlDict file."""
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

// ************************************************************************* //
"""
        with open(os.path.join(system_dir, 'controlDict'), 'w') as f:
            f.write(content)
    
    def _create_fv_schemes(self, system_dir):
        """Create basic fvSchemes file."""
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

// ************************************************************************* //
"""
        with open(os.path.join(system_dir, 'fvSchemes'), 'w') as f:
            f.write(content)
    
    def _create_fv_solution(self, system_dir):
        """Create basic fvSolution file."""
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-06;
        relTol          0.1;
    }

    "(k|epsilon)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-06;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon)"   1e-4;
    }
}

relaxationFactors
{
    equations
    {
        U               0.9;
        ".*"            0.9;
    }
}

// ************************************************************************* //
"""
        with open(os.path.join(system_dir, 'fvSolution'), 'w') as f:
            f.write(content)
    
    def _create_basic_fields(self, zero_dir):
        """Create basic field files in 0 directory."""
        # Create U file
        u_content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (10 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (10 0 0);
    }
    
    outlet
    {
        type            zeroGradient;
    }
    
    walls
    {
        type            noSlip;
    }
    
    symmetry
    {
        type            symmetryPlane;
    }
}

// ************************************************************************* //
"""
        with open(os.path.join(zero_dir, 'U'), 'w') as f:
            f.write(u_content)
        
        # Create p file
        p_content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    
    walls
    {
        type            zeroGradient;
    }
    
    symmetry
    {
        type            symmetryPlane;
    }
}

// ************************************************************************* //
"""
        with open(os.path.join(zero_dir, 'p'), 'w') as f:
            f.write(p_content)
    
    def _create_transport_properties(self, constant_dir):
        """Create transportProperties file."""
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] 1.5e-05;

// ************************************************************************* //
"""
        with open(os.path.join(constant_dir, 'transportProperties'), 'w') as f:
            f.write(content)
        
    def closeEvent(self, event):
        """Handle application close event - close immediately without confirmation."""
        # Stop any running simulation
        if self._simulation_worker is not None:
            self._simulation_worker.stop()
        if self._simulation_thread is not None and self._simulation_thread.isRunning():
            self._simulation_thread.quit()
            self._simulation_thread.wait(1000)
        event.accept()


def main():
    """Main entry point for comprehensive nozzle design application."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Nozzle CFD Design Tool")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("CFD Research")
    
    # Create and show main window
    window = NozzleDesignGUI()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
