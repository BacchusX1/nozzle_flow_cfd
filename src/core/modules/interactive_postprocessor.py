"""
Interactive Post-Processor Widget for OpenFOAM CFD Results

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
    QCheckBox, QFormLayout, QFrame
)
from PySide6.QtCore import Qt, Signal

# Import matplotlib AFTER PySide6 to ensure proper Qt binding
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import case analyzer functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
try:
    from case_analyzer import OpenFOAMCase, read_foam_file, parse_field_data
except ImportError:
    # Fallback - define minimal versions
    OpenFOAMCase = None


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
    Interactive post-processor widget for OpenFOAM CFD results.
    
    Features:
    - Load and visualize OpenFOAM case results
    - Select from available fields and time steps
    - Interactive value inspection on click
    - Zoom with middle mouse scroll
    - Pan with left mouse drag
    """
    
    def __init__(self, parent=None, theme=None):
        super().__init__(parent)
        self.theme = theme
        
        # Case data
        self.case = None
        self.case_dir = ""
        self.current_field_data = None
        self.current_node_values = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Left panel - controls
        controls_widget = self._create_controls_panel()
        controls_widget.setMaximumWidth(350)
        controls_widget.setMinimumWidth(280)
        
        # Right panel - interactive canvas
        canvas_widget = self._create_canvas_panel()
        
        # Add to splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(controls_widget)
        splitter.addWidget(canvas_widget)
        splitter.setSizes([320, 800])
        
        layout.addWidget(splitter)
        
    def _create_controls_panel(self):
        """Create the controls panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # === Case Loading ===
        load_group = QGroupBox("Load Case")
        load_layout = QVBoxLayout(load_group)
        
        # Case path
        path_layout = QHBoxLayout()
        self.case_path_edit = QComboBox()
        self.case_path_edit.setEditable(True)
        self.case_path_edit.addItems(["./case", "./case2"])
        
        btn_browse = QPushButton("...")
        btn_browse.setMaximumWidth(40)
        btn_browse.clicked.connect(self._browse_case)
        
        path_layout.addWidget(self.case_path_edit)
        path_layout.addWidget(btn_browse)
        load_layout.addLayout(path_layout)
        
        btn_load = QPushButton("📂 Load Case")
        btn_load.clicked.connect(self._load_case)
        load_layout.addWidget(btn_load)
        
        layout.addWidget(load_group)
        
        # === Field Selection ===
        field_group = QGroupBox("Field Selection")
        field_layout = QFormLayout(field_group)
        
        self.field_combo = QComboBox()
        self.field_combo.addItems(["U", "p", "T", "k", "omega", "nut", "epsilon"])
        self.field_combo.currentTextChanged.connect(self._on_field_changed)
        
        self.time_combo = QComboBox()
        self.time_combo.addItem("0")
        self.time_combo.currentTextChanged.connect(self._on_time_changed)
        
        field_layout.addRow("Field:", self.field_combo)
        field_layout.addRow("Time:", self.time_combo)
        
        layout.addWidget(field_group)
        
        # === Visualization Settings ===
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_group)
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "plasma", "inferno", "magma",
            "jet", "coolwarm", "RdYlBu", "seismic"
        ])
        self.colormap_combo.currentTextChanged.connect(self._update_plot)
        
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(5, 100)
        self.levels_spin.setValue(20)
        self.levels_spin.valueChanged.connect(self._update_plot)
        
        self.show_mesh_check = QCheckBox("Show mesh edges")
        self.show_mesh_check.setChecked(False)
        self.show_mesh_check.stateChanged.connect(self._update_plot)
        
        viz_layout.addRow("Colormap:", self.colormap_combo)
        viz_layout.addRow("Levels:", self.levels_spin)
        viz_layout.addWidget(self.show_mesh_check)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_reset = QPushButton("🔄 Reset View")
        btn_reset.clicked.connect(self._reset_view)
        btn_save = QPushButton("💾 Save Image")
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
        
        # === Case Summary ===
        summary_group = QGroupBox("Case Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        self.summary_text.setText("Load a case to see summary...")
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        # === Instructions ===
        help_group = QGroupBox("Controls")
        help_layout = QVBoxLayout(help_group)
        help_label = QLabel(
            "🖱️ <b>Left click</b>: Inspect value\n"
            "🖱️ <b>Left drag</b>: Pan view\n"
            "🖱️ <b>Scroll</b>: Zoom in/out"
        )
        help_label.setWordWrap(True)
        help_layout.addWidget(help_label)
        layout.addWidget(help_group)
        
        layout.addStretch()
        
        return widget
        
    def _create_canvas_panel(self):
        """Create the canvas panel with interactive plot."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create interactive canvas
        self.canvas = InteractiveCanvas(self, width=10, height=6, dpi=100)
        self.canvas.value_picked.connect(self._on_value_picked)
        
        # Initial message
        self.canvas.ax.text(
            0.5, 0.5,
            'Load a case to visualize results\n\n'
            'Use the controls on the left to:\n'
            '• Load an OpenFOAM case\n'
            '• Select field and time step\n'
            '• Customize visualization',
            transform=self.canvas.ax.transAxes,
            ha='center', va='center',
            color='#808080', fontsize=12
        )
        
        layout.addWidget(self.canvas)
        
        return widget
        
    def _browse_case(self):
        """Browse for case directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select OpenFOAM Case Directory",
            os.path.expanduser("~")
        )
        if directory:
            self.case_path_edit.setCurrentText(directory)
            
    def _load_case(self):
        """Load OpenFOAM case."""
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
            
        # Check for OpenFOAM structure
        constant_dir = os.path.join(case_dir, 'constant')
        system_dir = os.path.join(case_dir, 'system')
        
        if not os.path.exists(constant_dir) or not os.path.exists(system_dir):
            QMessageBox.warning(
                self, "Load Case",
                "This doesn't look like an OpenFOAM case.\n"
                "Expected 'constant/' and 'system/' directories."
            )
            return
            
        try:
            # Load case using case_analyzer
            if OpenFOAMCase is not None:
                self.case = OpenFOAMCase(case_dir)
                self.case.load_mesh()
                self.case.prepare_2d_surface()
            else:
                QMessageBox.warning(self, "Load Case", "Case analyzer module not available.")
                return
                
            self.case_dir = case_dir
            
            # Get available time directories
            time_dirs = self.case.get_time_dirs()
            
            # Update time combo
            self.time_combo.clear()
            for t_val, t_dir in time_dirs:
                self.time_combo.addItem(t_dir)
            
            # Select latest time
            if time_dirs:
                self.time_combo.setCurrentText(time_dirs[-1][1])
                
            # Get available fields from latest time
            if time_dirs:
                latest_dir = os.path.join(case_dir, time_dirs[-1][1])
                fields = []
                for f in os.listdir(latest_dir):
                    f_path = os.path.join(latest_dir, f)
                    if os.path.isfile(f_path) and not f.startswith('.'):
                        fields.append(f)
                        
                self.field_combo.clear()
                self.field_combo.addItems(sorted(fields))
                
            # Update summary
            summary = f"""Case: {os.path.basename(case_dir)}
Path: {case_dir}
Mesh:
  • Points: {len(self.case.points):,}
  • Faces: {len(self.case.faces):,}
  • Cells: {self.case.n_cells:,}
Time steps: {len(time_dirs)}
"""
            self.summary_text.setText(summary)
            
            # Trigger initial plot
            self._update_plot()
            
            QMessageBox.information(
                self, "Case Loaded",
                f"Successfully loaded case with {self.case.n_cells:,} cells"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Load Error", f"Failed to load case:\n{str(e)}")
            
    def _on_field_changed(self, field_name):
        """Handle field selection change."""
        self._update_plot()
        
    def _on_time_changed(self, time_dir):
        """Handle time step change."""
        self._update_plot()
        
    def _update_plot(self):
        """Update the visualization plot."""
        if self.case is None:
            return
            
        field_name = self.field_combo.currentText()
        time_dir = self.time_combo.currentText()
        
        if not field_name or not time_dir:
            return
            
        try:
            # Load field data
            field_data = self.case.load_field(time_dir, field_name)
            
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
                
            # Interpolate to nodes
            node_values = self.case.interpolate_to_nodes(field_data)
            
            # Handle vector fields - compute magnitude
            is_vector = len(node_values.shape) > 1 and node_values.shape[1] == 3
            if is_vector:
                plot_values = np.linalg.norm(node_values, axis=1)
                field_label = f"|{field_name}|"
            else:
                plot_values = node_values
                field_label = field_name
                
            # Get the subset of values for the triangulation
            if hasattr(self.case, 'surface_point_ids') and self.case.surface_point_ids is not None:
                plot_values_subset = plot_values[self.case.surface_point_ids]
            elif hasattr(self.case, 'unique_indices') and self.case.unique_indices is not None:
                plot_values_subset = plot_values[self.case.unique_indices]
            else:
                plot_values_subset = plot_values
                
            # Store for value lookup
            self.canvas.set_field_data(
                self.case.points_2d,
                plot_values_subset,
                self.case.triangulation,
                field_name
            )
            
            # Clear and plot
            self.canvas.fig.clear()
            self.canvas.ax = self.canvas.fig.add_subplot(111, facecolor='#1e1e1e')
            
            # Get colormap and levels
            cmap = self.colormap_combo.currentText()
            levels = self.levels_spin.value()
            
            # Create contour plot
            tripcolor = self.canvas.ax.tripcolor(
                self.case.triangulation,
                plot_values_subset,
                shading='gouraud',
                cmap=cmap
            )
            
            # Add colorbar
            cbar = self.canvas.fig.colorbar(tripcolor, ax=self.canvas.ax, label=field_label)
            cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
            cbar.ax.yaxis.label.set_color('#e0e0e0')
            for label in cbar.ax.yaxis.get_ticklabels():
                label.set_color('#e0e0e0')
                
            # Show mesh edges if requested
            if self.show_mesh_check.isChecked():
                self.canvas.ax.triplot(
                    self.case.triangulation,
                    'k-', linewidth=0.1, alpha=0.3
                )
                
            # Add statistics text
            stats_text = (
                f"Min: {plot_values_subset.min():.3e}\n"
                f"Max: {plot_values_subset.max():.3e}\n"
                f"Mean: {plot_values_subset.mean():.3e}"
            )
            self.canvas.ax.text(
                0.02, 0.98, stats_text,
                transform=self.canvas.ax.transAxes,
                va='top', ha='left', color='#e0e0e0',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#1e1e1e', alpha=0.8, edgecolor='#404040')
            )
            
            # Style and labels
            self.canvas.ax.set_title(f'{field_label} at t = {time_dir}', color='#e0e0e0', fontsize=12)
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
        if self.case is None or self.case.points_2d is None:
            return
            
        # Get data bounds
        x = self.case.points_2d[:, 0]
        y = self.case.points_2d[:, 1]
        
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
