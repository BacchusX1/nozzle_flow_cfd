"""
Interactive Post-Processor Widget for SU2 CFD Results

Provides interactive visualization with:
- Plot selection from available fields
- Value inspection on click
- Zoom with middle mouse scroll
- Pan with left mouse drag
"""

import os
import sys
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QSplitter, QTextEdit,
    QCheckBox, QFormLayout, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal

# Import matplotlib AFTER PySide6 to ensure proper Qt binding
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import SU2 case analyzer
from backend.postprocessing.su2_case_analyzer import SU2Case, parse_history_csv

# Import standard values for defaults
from backend.standard_values import DEFAULTS


class InteractiveCanvas(FigureCanvas):
    """
    Interactive matplotlib canvas with:
    - Middle mouse scroll to zoom
    - Left mouse drag to pan
    - Left click to inspect values
    """
    
    value_picked = Signal(float, float, object)  # x, y, value
    
    def __init__(self, parent=None, width=10, height=6, dpi=100, facecolor='#1e1e1e'):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=facecolor)
        self.ax = self.fig.add_subplot(111, facecolor='#1e1e1e')
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Store current data for value lookup
        self.current_data = None  # Will store (points_2d, node_values, triangulation)
        self.current_field_name = ""
        
        # Pan state
        self._pan_active = False
        self._pan_start = None
        self._xlim_start = None
        self._ylim_start = None
        
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
        self.ax.grid(True, alpha=0.2, color='#606060')
        
    def _on_scroll(self, event):
        """Handle scroll events for zooming."""
        if event.inaxes != self.ax:
            return
            
        # Zoom factor
        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return
            
        # Get current limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Get mouse position in data coordinates
        xdata = event.xdata
        ydata = event.ydata
        
        # Calculate new limits centered on mouse position
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
            
        if event.button == 1:  # Left mouse button
            # Start pan
            self._pan_active = True
            self._pan_start = (event.xdata, event.ydata)
            self._xlim_start = self.ax.get_xlim()
            self._ylim_start = self.ax.get_ylim()
            
    def _on_release(self, event):
        """Handle mouse release events."""
        if event.button == 1:  # Left mouse button
            if self._pan_active and self._pan_start is not None:
                # Check if it was a click (no significant movement)
                if event.inaxes == self.ax and event.xdata is not None:
                    dx = abs(event.xdata - self._pan_start[0]) if self._pan_start[0] else 0
                    dy = abs(event.ydata - self._pan_start[1]) if self._pan_start[1] else 0
                    
                    xlim = self.ax.get_xlim()
                    ylim = self.ax.get_ylim()
                    threshold_x = (xlim[1] - xlim[0]) * 0.01
                    threshold_y = (ylim[1] - ylim[0]) * 0.01
                    
                    if dx < threshold_x and dy < threshold_y:
                        # It's a click - look up value
                        self._lookup_value(event.xdata, event.ydata)
            
            self._pan_active = False
            self._pan_start = None
            
    def _on_motion(self, event):
        """Handle mouse motion events for panning."""
        if not self._pan_active or self._pan_start is None:
            return
            
        if event.inaxes != self.ax or event.xdata is None:
            return
            
        # Calculate pan offset
        dx = self._pan_start[0] - event.xdata
        dy = self._pan_start[1] - event.ydata
        
        # Apply pan
        self.ax.set_xlim(self._xlim_start[0] + dx, self._xlim_start[1] + dx)
        self.ax.set_ylim(self._ylim_start[0] + dy, self._ylim_start[1] + dy)
        
        self.draw()
        
    def _lookup_value(self, x, y):
        """Look up field value at given coordinates."""
        if self.current_data is None:
            return
            
        points_2d, node_values, triangulation = self.current_data
        
        if points_2d is None or node_values is None:
            return
            
        # Find nearest point
        distances = np.sqrt((points_2d[:, 0] - x)**2 + (points_2d[:, 1] - y)**2)
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        
        # Check if within reasonable distance
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        max_dist = 0.05 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        
        if nearest_dist > max_dist:
            return
            
        value = node_values[nearest_idx]
        
        # Emit signal with picked value
        self.value_picked.emit(x, y, value)
        
    def set_field_data(self, points_2d, node_values, triangulation, field_name):
        """Store field data for value lookup."""
        self.current_data = (points_2d, node_values, triangulation)
        self.current_field_name = field_name
        
    def clear_field_data(self):
        """Clear stored field data."""
        self.current_data = None
        self.current_field_name = ""


class InteractivePostprocessorWidget(QWidget):
    """
    Interactive post-processor widget for SU2 CFD results.
    
    Features:
    - Load and visualize SU2 case results
    - Select from available fields
    - Interactive value inspection on click
    - Zoom with middle mouse scroll
    - Pan with left mouse drag
    """
    
    def __init__(self, parent=None, theme=None, scale_factor=1.0, configure_splitter_func=None):
        super().__init__(parent)
        self.theme = theme
        self.scale_factor = scale_factor
        self._configure_splitter = configure_splitter_func
        
        # Case data
        self.case = None
        self.case_dir = ""
        self.current_field_data = None
        self.current_node_values = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the widget UI with canvas in center, controls on left, summary on right.
        
        Uses the same splitter layout pattern as the Geometry and Simulation tabs
        for consistent look and feel across the application.
        """
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        scale = float(self.scale_factor or 1.0)
        
        # === MAIN HORIZONTAL SPLITTER: Controls | Canvas + Summary ===
        top_splitter = QSplitter(Qt.Horizontal)
        if self._configure_splitter:
            self._configure_splitter(top_splitter)
        else:
            top_splitter.setChildrenCollapsible(False)
            top_splitter.setHandleWidth(max(8, int(round(10 * scale))))
        
        # === LEFT PANEL: Controls (scrollable, like other tabs) ===
        control_panel = QWidget()
        control_panel.setMinimumWidth(int(round(520 * scale)))  # Match other tabs
        control_panel.setStyleSheet("""
            QWidget {
                background: #252526;
                border-right: 1px solid #3c3c3c;
            }
        """)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
        """)
        
        controls_widget = self._create_controls_panel()
        scroll.setWidget(controls_widget)
        
        panel_layout = QVBoxLayout(control_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)
        
        # === RIGHT SIDE: Canvas with Summary below ===
        right_container = QWidget()
        right_layout = QHBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # Canvas (center)
        center_widget = self._create_canvas_panel()
        
        # Summary panel (right side of canvas)
        summary_widget = self._create_summary_panel()
        summary_widget.setMinimumWidth(int(round(280 * scale)))
        summary_widget.setMaximumWidth(int(round(400 * scale)))
        
        # Sub-splitter for canvas + summary
        canvas_summary_splitter = QSplitter(Qt.Horizontal)
        if self._configure_splitter:
            self._configure_splitter(canvas_summary_splitter)
        else:
            canvas_summary_splitter.setChildrenCollapsible(False)
            canvas_summary_splitter.setHandleWidth(max(8, int(round(10 * scale))))
        
        canvas_summary_splitter.addWidget(center_widget)
        canvas_summary_splitter.addWidget(summary_widget)
        canvas_summary_splitter.setStretchFactor(0, 1)  # Canvas stretches
        canvas_summary_splitter.setStretchFactor(1, 0)  # Summary fixed
        canvas_summary_splitter.setSizes([int(round(700 * scale)), int(round(300 * scale))])
        
        right_layout.addWidget(canvas_summary_splitter)
        
        # Add panels to main splitter
        top_splitter.addWidget(control_panel)
        top_splitter.addWidget(right_container)
        top_splitter.setStretchFactor(0, 3)  # Controls panel proportion
        top_splitter.setStretchFactor(1, 2)  # Canvas area proportion
        top_splitter.setSizes([int(round(760 * scale)), int(round(520 * scale))])

        layout.addWidget(top_splitter, 1)
        
    def _create_controls_panel(self):
        """Create the controls panel (left side) with styling matching other tabs."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Panel header (matching other tabs)
        panel_header = QLabel("Results Viewer")
        panel_header.setStyleSheet("""
            QLabel {
                color: #e0e0e0;
                font-size: 17px;
                font-weight: 700;
                padding-bottom: 10px;
                border-bottom: 3px solid #0078d4;
            }
        """)
        layout.addWidget(panel_header)
        
        # === Case Loading ===
        load_group = QGroupBox("Load Case")
        load_layout = QVBoxLayout(load_group)
        load_layout.setSpacing(8)
        load_layout.setContentsMargins(12, 20, 12, 12)
        
        # Case path - use defaults from YAML
        path_layout = QHBoxLayout()
        self.case_path_edit = QComboBox()
        self.case_path_edit.setEditable(True)
        case_dirs = getattr(DEFAULTS, 'postproc_case_directories', ['./case', './case2'])
        self.case_path_edit.addItems(case_dirs)
        self.case_path_edit.setToolTip("Path to SU2 case directory with results")
        
        btn_browse = QPushButton("...")
        btn_browse.setMaximumWidth(40)
        btn_browse.clicked.connect(self._browse_case)
        
        path_layout.addWidget(self.case_path_edit)
        path_layout.addWidget(btn_browse)
        load_layout.addLayout(path_layout)
        
        btn_load = QPushButton("Load Case")
        btn_load.clicked.connect(self._load_case)
        btn_load.setMinimumHeight(36)
        load_layout.addWidget(btn_load)
        
        layout.addWidget(load_group)
        
        # === Field Selection ===
        field_group = QGroupBox("Field Selection")
        field_layout = QFormLayout(field_group)
        field_layout.setSpacing(10)
        field_layout.setContentsMargins(12, 20, 12, 12)
        
        # Use available fields from DEFAULTS
        available_fields = getattr(DEFAULTS, 'postproc_available_fields', 
                                   ["Pressure", "Velocity", "Temperature", "Mach", "Density"])
        default_field = getattr(DEFAULTS, 'postproc_default_field', 'Pressure')
        
        self.field_combo = QComboBox()
        self.field_combo.addItems(available_fields)
        self.field_combo.setCurrentText(default_field)
        self.field_combo.currentTextChanged.connect(self._on_field_changed)
        self.field_combo.setToolTip("Select field variable to visualize")
        
        self.time_combo = QComboBox()
        self.time_combo.addItem("0")
        self.time_combo.currentTextChanged.connect(self._on_time_changed)
        self.time_combo.setToolTip("Select time step / iteration")
        
        field_layout.addRow("Field:", self.field_combo)
        field_layout.addRow("Time:", self.time_combo)
        
        layout.addWidget(field_group)
        
        # === Visualization Settings ===
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_group)
        viz_layout.setSpacing(10)
        viz_layout.setContentsMargins(12, 20, 12, 12)
        
        # Use colormap from DEFAULTS
        default_colormap = getattr(DEFAULTS, 'postproc_colormap', 'viridis')
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "plasma", "inferno", "magma",
            "jet", "coolwarm", "RdYlBu", "seismic"
        ])
        self.colormap_combo.setCurrentText(default_colormap)
        self.colormap_combo.currentTextChanged.connect(self._update_plot)
        
        # Use contour levels from DEFAULTS
        default_levels = getattr(DEFAULTS, 'postproc_contour_levels', 20)
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(5, 100)
        self.levels_spin.setValue(default_levels)
        self.levels_spin.setValue(20)
        self.levels_spin.valueChanged.connect(self._update_plot)
        
        self.show_mesh_check = QCheckBox("Show mesh edges")
        # Use show mesh edges from DEFAULTS
        default_show_mesh = getattr(DEFAULTS, 'postproc_show_mesh_edges', False)
        self.show_mesh_check.setChecked(default_show_mesh)
        self.show_mesh_check.stateChanged.connect(self._update_plot)
        
        viz_layout.addRow("Colormap:", self.colormap_combo)
        viz_layout.addRow("Levels:", self.levels_spin)
        viz_layout.addWidget(self.show_mesh_check)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self._reset_view)
        btn_save = QPushButton("Save Image")
        btn_save.clicked.connect(self._save_image)
        btn_layout.addWidget(btn_reset)
        btn_layout.addWidget(btn_save)
        viz_layout.addRow(btn_layout)
        
        layout.addWidget(viz_group)
        
        # === Value Inspector ===
        inspector_group = QGroupBox("Value Inspector")
        inspector_layout = QVBoxLayout(inspector_group)
        
        self.value_label = QLabel("Click on the plot to inspect values")
        self.value_label.setWordWrap(True)
        self.value_label.setStyleSheet("""
            QLabel {
                background: #2d2d2d;
                padding: 8px;
                border-radius: 4px;
                font-family: monospace;
            }
        """)
        inspector_layout.addWidget(self.value_label)
        
        # Coordinates display
        self.coords_label = QLabel("X: --  Y: --")
        inspector_layout.addWidget(self.coords_label)
        
        layout.addWidget(inspector_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_summary_panel(self):
        """Create the case summary panel (right side)."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # === Case Summary ===
        summary_group = QGroupBox("Case Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setText("Load a case to see summary...")
        self.summary_text.setStyleSheet("""
            QTextEdit {
                background: #2d2d2d;
                border: none;
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        layout.addStretch()
        
        return widget
        
    def _create_canvas_panel(self):
        """Create the canvas panel with interactive plot and instructions overlay."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Instructions text at top of canvas
        instructions = QLabel(
            "🖱️ <b>Left click</b>: Inspect value  |  "
            "🖱️ <b>Left drag</b>: Pan view  |  "
            "🖱️ <b>Scroll</b>: Zoom in/out"
        )
        instructions.setStyleSheet("""
            QLabel {
                background: #252525;
                color: #a0a0a0;
                padding: 6px 12px;
                font-size: 11px;
                border-bottom: 1px solid #404040;
            }
        """)
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        
        # Create interactive canvas
        self.canvas = InteractiveCanvas(self, width=10, height=6, dpi=100)
        self.canvas.value_picked.connect(self._on_value_picked)
        
        # Initial message
        self.canvas.ax.text(
            0.5, 0.5,
            'Load a case to visualize results\n\n'
            'Use the controls on the left to:\n'
            '• Load an SU2 case directory\n'
            '• Select field to visualize\n'
            '• Customize visualization',
            transform=self.canvas.ax.transAxes,
            ha='center', va='center',
            color='#808080', fontsize=12
        )
        
        layout.addWidget(self.canvas, 1)
        
        return widget
        
    def _browse_case(self):
        """Browse for case directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select SU2 Case Directory",
            os.path.expanduser("~")
        )
        if directory:
            self.case_path_edit.setCurrentText(directory)
            
    def _load_case(self):
        """Load SU2 case."""
        case_dir = self.case_path_edit.currentText().strip()
        
        if not case_dir:
            QMessageBox.warning(self, "Load Case", "Please specify a case directory.")
            return
            
        # Resolve relative paths
        if not os.path.isabs(case_dir):
            # Try relative to workspace
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            case_dir = os.path.join(workspace_root, case_dir)
            
        if not os.path.exists(case_dir):
            QMessageBox.warning(self, "Load Case", f"Directory not found: {case_dir}")
            return
            
        # Check for SU2 case files (.cfg or .su2 mesh) OR VTU files directly
        import glob
        import re
        cfg_files = glob.glob(os.path.join(case_dir, "*.cfg"))
        mesh_files = glob.glob(os.path.join(case_dir, "*.su2"))
        vtu_files = glob.glob(os.path.join(case_dir, "flow_*.vtu"))
        
        if not cfg_files and not mesh_files and not vtu_files:
            QMessageBox.warning(
                self, "Load Case",
                "This doesn't look like an SU2 case.\n"
                "Expected .cfg config file, .su2 mesh file, or flow_*.vtu results."
            )
            return
            
        try:
            # Load case using SU2 case analyzer
            if SU2Case is not None:
                self.case = SU2Case(case_dir)
                self.case.load_mesh()
                self.case.load_solution()
            else:
                QMessageBox.warning(self, "Load Case", "SU2 case analyzer module not available.")
                return
                
            self.case_dir = case_dir
            
            # Scan for available timesteps from flow_*.vtu files
            self.time_combo.clear()
            self.available_timesteps = {}  # Store mapping: display name -> file path
            
            # Find all flow_*.vtu files and extract timestep numbers
            vtu_pattern = re.compile(r'flow_(\d+)\.vtu$')
            timesteps = []
            for vtu_file in vtu_files:
                filename = os.path.basename(vtu_file)
                match = vtu_pattern.match(filename)
                if match:
                    step_num = int(match.group(1))
                    timesteps.append((step_num, vtu_file))
            
            # Sort by timestep number
            timesteps.sort(key=lambda x: x[0])
            
            # Populate time combo
            if timesteps:
                for step_num, file_path in timesteps:
                    display_name = f"Step {step_num:05d}"
                    self.time_combo.addItem(display_name)
                    self.available_timesteps[display_name] = file_path
                # Select the latest timestep by default
                self.time_combo.setCurrentIndex(self.time_combo.count() - 1)
            else:
                # Fallback if no timesteps found
                self.time_combo.addItem("latest")
                self.available_timesteps["latest"] = None
            
            # Get available fields from solution data
            available_fields = self.case.get_available_fields()
            self.field_combo.clear()
            if available_fields:
                self.field_combo.addItems(available_fields)
            else:
                # Default SU2 fields
                self.field_combo.addItems(["Pressure", "Velocity", "Temperature", "Mach", "Density"])
                
            # Update summary
            mesh_info = self.case.get_mesh_info()
            summary = f"""Case: {os.path.basename(case_dir)}
Path: {case_dir}
Mesh:
  • Points: {mesh_info.get('num_points', 'N/A'):,}
  • Elements: {mesh_info.get('num_elements', 'N/A'):,}
  • Dimension: {mesh_info.get('dimension', 'N/A')}D
Solver: {self.case.get_solver_type() or 'N/A'}
"""
            self.summary_text.setText(summary)
            
            # Trigger initial plot
            self._update_plot()
            
            QMessageBox.information(
                self, "Case Loaded",
                f"Successfully loaded SU2 case"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", f"Failed to load case:\n{str(e)}")
            
    def _on_field_changed(self, field_name):
        """Handle field selection change."""
        self._update_plot()
        
    def _on_time_changed(self, time_dir):
        """Handle time step change - reload the VTU file for selected timestep."""
        if self.case is None:
            return
            
        # Get the selected timestep file path
        selected_time = self.time_combo.currentText()
        if hasattr(self, 'available_timesteps') and selected_time in self.available_timesteps:
            vtu_path = self.available_timesteps[selected_time]
            if vtu_path:
                # Reload the solution for this specific timestep
                self.case.load_solution(vtu_path)
        
        self._update_plot()
        
    def _update_plot(self):
        """Update the visualization plot."""
        if self.case is None:
            return
            
        field_name = self.field_combo.currentText()
        
        if not field_name:
            return
            
        try:
            # Get the selected timestep file path and load it if not already loaded
            selected_time = self.time_combo.currentText()
            if hasattr(self, 'available_timesteps') and selected_time in self.available_timesteps:
                vtu_path = self.available_timesteps[selected_time]
                if vtu_path:
                    self.case.load_solution(vtu_path)
            
            # Load field data from SU2 case
            field_data = self.case.get_field_data(field_name)
            
            if field_data is None:
                self.canvas.ax.clear()
                self.canvas.ax.text(
                    0.5, 0.5,
                    f'Could not load field: {field_name}',
                    transform=self.canvas.ax.transAxes,
                    ha='center', va='center',
                    color='#ff6060', fontsize=12
                )
                self.canvas._style_axes()
                self.canvas.draw()
                return
                
            # Get node coordinates
            points_2d = self.case.get_points_2d()
            node_values = field_data
            
            # Handle vector fields - compute magnitude
            is_vector = len(node_values.shape) > 1 and node_values.shape[1] >= 2
            if is_vector:
                plot_values = np.linalg.norm(node_values, axis=1)
                field_label = f"|{field_name}|"
            else:
                plot_values = node_values.flatten()
                field_label = field_name
                
            # Create triangulation if available
            triangulation = self.case.get_triangulation()
            
            # Store for value lookup
            self.canvas.set_field_data(
                points_2d,
                plot_values,
                triangulation,
                field_name
            )
            
            # Clear and plot
            self.canvas.fig.clear()
            self.canvas.ax = self.canvas.fig.add_subplot(111, facecolor='#1e1e1e')
            
            # Get colormap and levels
            cmap = self.colormap_combo.currentText()
            levels = self.levels_spin.value()
            
            # Create contour plot
            if triangulation is not None:
                tripcolor = self.canvas.ax.tripcolor(
                    triangulation,
                    plot_values,
                    shading='gouraud',
                    cmap=cmap
                )
                
                # Draw boundary outline for clear nozzle shape
                self._draw_boundary(triangulation)
            else:
                # Fall back to scatter plot
                tripcolor = self.canvas.ax.scatter(
                    points_2d[:, 0], points_2d[:, 1],
                    c=plot_values, cmap=cmap, s=5
                )
            
            # Add colorbar
            cbar = self.canvas.fig.colorbar(tripcolor, ax=self.canvas.ax, label=field_label)
            cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
            cbar.ax.yaxis.label.set_color('#e0e0e0')
            for label in cbar.ax.yaxis.get_ticklabels():
                label.set_color('#e0e0e0')
                
            # Show mesh edges if requested
            if self.show_mesh_check.isChecked() and triangulation is not None:
                self.canvas.ax.triplot(
                    triangulation,
                    'k-', linewidth=0.1, alpha=0.3
                )
                
            # Add statistics text
            stats_text = (
                f"Min: {plot_values.min():.3e}\n"
                f"Max: {plot_values.max():.3e}\n"
                f"Mean: {plot_values.mean():.3e}"
            )
            self.canvas.ax.text(
                0.02, 0.98, stats_text,
                transform=self.canvas.ax.transAxes,
                va='top', ha='left', color='#e0e0e0',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#1e1e1e', alpha=0.8, edgecolor='#404040')
            )
            
            # Style and labels
            self.canvas.ax.set_title(f'{field_label}', color='#e0e0e0', fontsize=12)
            self.canvas._style_axes()
            self.canvas.ax.set_aspect('equal')
            
            self.canvas.draw()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            self.canvas.ax.clear()
            self.canvas.ax.text(
                0.5, 0.5,
                f'Error loading field:\n{str(e)}',
                transform=self.canvas.ax.transAxes,
                ha='center', va='center',
                color='#ff6060', fontsize=10
            )
            self.canvas._style_axes()
            self.canvas.draw()
    
    def _draw_boundary(self, triangulation):
        """Draw mesh boundary outline for clear nozzle shape visualization."""
        from collections import Counter
        
        # Find boundary edges (edges that appear exactly once in the mesh)
        edge_count = Counter()
        for tri in triangulation.triangles:
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]]))
            ]
            for edge in edges:
                edge_count[edge] += 1
        
        # Extract boundary edges (appear only once)
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        
        # Draw each boundary edge
        for edge in boundary_edges:
            x = [triangulation.x[edge[0]], triangulation.x[edge[1]]]
            y = [triangulation.y[edge[0]], triangulation.y[edge[1]]]
            self.canvas.ax.plot(x, y, 'k-', linewidth=0.8, alpha=0.8)
            
    def _on_value_picked(self, x, y, value):
        """Handle value pick from canvas."""
        # Format value based on type
        if isinstance(value, np.ndarray):
            if len(value) == 3:
                val_str = f"({value[0]:.4e}, {value[1]:.4e}, {value[2]:.4e})"
                mag = np.linalg.norm(value)
                val_str += f"\nMagnitude: {mag:.4e}"
            else:
                val_str = str(value)
        else:
            val_str = f"{value:.6e}"
            
        # Update labels
        self.value_label.setText(f"Value: {val_str}")
        self.coords_label.setText(f"X: {x:.6f}  Y: {y:.6f}")
        
        # Add marker on plot
        # Remove previous marker if any
        for artist in self.canvas.ax.get_children():
            if hasattr(artist, '_is_value_marker'):
                artist.remove()
                
        # Add new marker
        marker = self.canvas.ax.plot(
            x, y, 'o',
            markersize=10,
            markerfacecolor='none',
            markeredgecolor='white',
            markeredgewidth=2,
            alpha=0.9
        )[0]
        marker._is_value_marker = True
        
        # Add text annotation
        text = self.canvas.ax.annotate(
            f'{value:.3e}',
            xy=(x, y),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#303030', alpha=0.9, edgecolor='white'),
            arrowprops=dict(arrowstyle='->', color='white', alpha=0.7)
        )
        text._is_value_marker = True
        
        self.canvas.draw()
        
    def _reset_view(self):
        """Reset view to show full data."""
        if self.case is None:
            return
            
        points_2d = self.case.get_points_2d()
        if points_2d is None or len(points_2d) == 0:
            return
            
        # Get data bounds
        x = points_2d[:, 0]
        y = points_2d[:, 1]
        
        margin = 0.05
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        
        self.canvas.ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
        self.canvas.ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
        
        self.canvas.draw()
        
    def _save_image(self):
        """Save current plot as image."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Image",
            "postprocessing_plot.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if filename:
            try:
                self.canvas.fig.savefig(filename, dpi=300, facecolor='#1e1e1e', edgecolor='none')
                QMessageBox.information(self, "Saved", f"Image saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{str(e)}")
