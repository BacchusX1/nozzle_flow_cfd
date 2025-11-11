"""
Post-processing Module

Handles visualization and analysis of CFD simulation results,
including field plots, contours, streamlines, and data extraction.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection


class FieldType(Enum):
    """Available field types for visualization."""
    VELOCITY_MAGNITUDE = "velocity_magnitude"
    VELOCITY_X = "velocity_x"
    VELOCITY_Y = "velocity_y"
    PRESSURE = "pressure"
    TURBULENT_KINETIC_ENERGY = "k"
    TURBULENT_DISSIPATION = "epsilon"
    WALL_SHEAR_STRESS = "wall_shear"
    VORTICITY = "vorticity"


class PlotType(Enum):
    """Available plot types."""
    CONTOUR = "contour"
    STREAMLINES = "streamlines"
    VECTOR = "vector"
    LINE_PLOT = "line_plot"
    SURFACE_PLOT = "surface"


@dataclass
class VisualizationSettings:
    """Settings for visualization."""
    field_type: FieldType = FieldType.VELOCITY_MAGNITUDE
    plot_type: PlotType = PlotType.CONTOUR
    
    # Contour settings
    num_levels: int = 20
    colormap: str = "jet"
    show_colorbar: bool = True
    filled_contours: bool = True
    
    # Streamline settings
    streamline_density: float = 1.0
    streamline_color: str = "black"
    streamline_width: float = 1.0
    
    # Vector settings
    vector_scale: float = 1.0
    vector_density: int = 10
    vector_color: str = "blue"
    
    # General settings
    show_mesh: bool = False
    show_boundaries: bool = True
    show_geometry: bool = True
    figure_size: Tuple[float, float] = (12, 8)
    dpi: int = 100


@dataclass
class ProbeData:
    """Data from a probe point."""
    location: Tuple[float, float]
    time_series: Dict[str, List[float]]
    time_values: List[float]


class ResultsProcessor:
    """Main results processing and visualization class."""
    
    def __init__(self):
        self.case_directory = ""
        self.results_data = {}
        self.mesh_data = None
        self.geometry_data = None
        self.settings = VisualizationSettings()
        self.probes: List[ProbeData] = []
        
    def load_case(self, case_directory: str):
        """Load OpenFOAM case results."""
        self.case_directory = case_directory
        
        # Load mesh data
        self._load_mesh_data()
        
        # Load field data
        self._load_field_data()
        
        # Load geometry data if available
        self._load_geometry_data()
        
    def _load_mesh_data(self):
        """Load mesh data from case."""
        mesh_file = os.path.join(self.case_directory, "mesh_data.json")
        if os.path.exists(mesh_file):
            with open(mesh_file, 'r') as f:
                self.mesh_data = json.load(f)
        else:
            # Try to read from OpenFOAM mesh files
            self._read_openfoam_mesh()
            
    def _load_field_data(self):
        """Load field data from time directories."""
        # Look for time directories
        time_dirs = []
        for item in os.listdir(self.case_directory):
            try:
                time_val = float(item)
                time_dirs.append((time_val, item))
            except ValueError:
                continue
                
        time_dirs.sort()
        
        # Load latest time step
        if time_dirs:
            latest_time = time_dirs[-1][1]
            self._load_time_step_data(latest_time)
            
    def _load_time_step_data(self, time_dir: str):
        """Load field data from specific time directory."""
        time_path = os.path.join(self.case_directory, time_dir)
        
        # Load available fields
        field_files = ['U', 'p', 'k', 'epsilon']
        self.results_data[time_dir] = {}
        
        for field in field_files:
            field_path = os.path.join(time_path, field)
            if os.path.exists(field_path):
                field_data = self._read_openfoam_field(field_path)
                self.results_data[time_dir][field] = field_data
                
    def _read_openfoam_field(self, field_path: str):
        """Read OpenFOAM field file (simplified)."""
        # This is a simplified reader - in practice would need proper OpenFOAM parser
        # For now, return dummy data
        return {
            'internal_field': np.random.random((100, 3)),  # Dummy velocity field
            'boundary_fields': {}
        }
        
    def _read_openfoam_mesh(self):
        """Read OpenFOAM mesh files (simplified)."""
        # This would read points, faces, cells from polyMesh
        # For now, create dummy mesh
        self.mesh_data = {
            'vertices': np.random.random((100, 2)),
            'elements': [[i, i+1, i+2] for i in range(0, 97, 3)],
            'element_type': 'triangle'
        }
        
    def _load_geometry_data(self):
        """Load original geometry data."""
        geometry_file = os.path.join(self.case_directory, "geometry_data.json")
        if os.path.exists(geometry_file):
            with open(geometry_file, 'r') as f:
                self.geometry_data = json.load(f)
                
    def create_visualization(self, settings: VisualizationSettings = None) -> plt.Figure:
        """Create visualization based on settings."""
        if settings:
            self.settings = settings
            
        fig, ax = plt.subplots(figsize=self.settings.figure_size, dpi=self.settings.dpi)
        
        # Get field data
        field_data = self._get_field_data(self.settings.field_type)
        
        if field_data is None:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
            
        # Create visualization based on plot type
        if self.settings.plot_type == PlotType.CONTOUR:
            self._create_contour_plot(ax, field_data)
        elif self.settings.plot_type == PlotType.STREAMLINES:
            self._create_streamline_plot(ax, field_data)
        elif self.settings.plot_type == PlotType.VECTOR:
            self._create_vector_plot(ax, field_data)
        elif self.settings.plot_type == PlotType.LINE_PLOT:
            self._create_line_plot(ax, field_data)
            
        # Add geometry overlay
        if self.settings.show_geometry:
            self._add_geometry_overlay(ax)
            
        # Add mesh overlay
        if self.settings.show_mesh:
            self._add_mesh_overlay(ax)
            
        # Set labels and title
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(f"{self.settings.field_type.value.replace('_', ' ').title()}")
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig
        
    def _get_field_data(self, field_type: FieldType):
        """Get field data for visualization."""
        if not self.results_data:
            return self._generate_dummy_field_data(field_type)
            
        # Get latest time step
        latest_time = list(self.results_data.keys())[-1]
        time_data = self.results_data[latest_time]
        
        if field_type == FieldType.VELOCITY_MAGNITUDE:
            if 'U' in time_data:
                velocity = time_data['U']['internal_field']
                return np.sqrt(velocity[:, 0]**2 + velocity[:, 1]**2)
        elif field_type == FieldType.VELOCITY_X:
            if 'U' in time_data:
                return time_data['U']['internal_field'][:, 0]
        elif field_type == FieldType.VELOCITY_Y:
            if 'U' in time_data:
                return time_data['U']['internal_field'][:, 1]
        elif field_type == FieldType.PRESSURE:
            if 'p' in time_data:
                return time_data['p']['internal_field']
                
        return self._generate_dummy_field_data(field_type)
        
    def _generate_dummy_field_data(self, field_type: FieldType):
        """Generate dummy field data for demonstration."""
        if not self.mesh_data:
            # Create dummy mesh
            x = np.linspace(0, 2, 50)
            y = np.linspace(-0.5, 0.5, 25)
            X, Y = np.meshgrid(x, y)
            
            # Generate field based on type
            if field_type == FieldType.VELOCITY_MAGNITUDE:
                data = np.sqrt((1 - X)**2 + Y**2) + 0.1 * np.random.random(X.shape)
            elif field_type == FieldType.PRESSURE:
                data = -0.5 * (X**2 + Y**2) + 2 * X
            elif field_type == FieldType.VELOCITY_X:
                data = 1 - X + 0.1 * Y
            elif field_type == FieldType.VELOCITY_Y:
                data = 0.1 * np.sin(2 * np.pi * X) * np.exp(-Y**2)
            else:
                data = np.random.random(X.shape)
                
            return {'X': X, 'Y': Y, 'data': data, 'coordinates': (X, Y)}
            
        # Use mesh coordinates
        vertices = np.array(self.mesh_data['vertices'])
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        
        # Generate field values at vertices
        if field_type == FieldType.VELOCITY_MAGNITUDE:
            values = np.sqrt((1 - x_coords)**2 + y_coords**2)
        elif field_type == FieldType.PRESSURE:
            values = -0.5 * (x_coords**2 + y_coords**2) + 2 * x_coords
        else:
            values = np.random.random(len(vertices))
            
        return {'vertices': vertices, 'values': values}
        
    def _create_contour_plot(self, ax, field_data):
        """Create contour plot."""
        if 'X' in field_data and 'Y' in field_data:
            # Structured data
            X, Y, data = field_data['X'], field_data['Y'], field_data['data']
            
            if self.settings.filled_contours:
                contour = ax.contourf(X, Y, data, levels=self.settings.num_levels, 
                                    cmap=self.settings.colormap)
            else:
                contour = ax.contour(X, Y, data, levels=self.settings.num_levels, 
                                   cmap=self.settings.colormap)
                                   
            if self.settings.show_colorbar:
                plt.colorbar(contour, ax=ax, label=self.settings.field_type.value)
                
        elif 'vertices' in field_data:
            # Unstructured data - create triangulation
            vertices = field_data['vertices']
            values = field_data['values']
            
            # Simple scatter plot for now
            scatter = ax.scatter(vertices[:, 0], vertices[:, 1], c=values, 
                               cmap=self.settings.colormap, s=20)
            
            if self.settings.show_colorbar:
                plt.colorbar(scatter, ax=ax, label=self.settings.field_type.value)
                
    def _create_streamline_plot(self, ax, field_data):
        """Create streamline plot."""
        # Get velocity components
        u_data = self._get_field_data(FieldType.VELOCITY_X)
        v_data = self._get_field_data(FieldType.VELOCITY_Y)
        
        if u_data and v_data and 'X' in u_data and 'X' in v_data:
            X, Y = u_data['X'], u_data['Y']
            U, V = u_data['data'], v_data['data']
            
            streamlines = ax.streamplot(X, Y, U, V, 
                                      density=self.settings.streamline_density,
                                      color=self.settings.streamline_color,
                                      linewidth=self.settings.streamline_width)
                                      
        # Add background contour
        if field_data and 'X' in field_data:
            contour = ax.contourf(field_data['X'], field_data['Y'], field_data['data'], 
                                alpha=0.3, cmap=self.settings.colormap)
            if self.settings.show_colorbar:
                plt.colorbar(contour, ax=ax, label=self.settings.field_type.value)
                
    def _create_vector_plot(self, ax, field_data):
        """Create vector plot."""
        u_data = self._get_field_data(FieldType.VELOCITY_X)
        v_data = self._get_field_data(FieldType.VELOCITY_Y)
        
        if u_data and v_data and 'X' in u_data:
            X, Y = u_data['X'], u_data['Y']
            U, V = u_data['data'], v_data['data']
            
            # Subsample for clarity
            step = max(1, len(X[0]) // self.settings.vector_density)
            X_sub = X[::step, ::step]
            Y_sub = Y[::step, ::step]
            U_sub = U[::step, ::step]
            V_sub = V[::step, ::step]
            
            ax.quiver(X_sub, Y_sub, U_sub, V_sub, 
                     scale=self.settings.vector_scale,
                     color=self.settings.vector_color)
                     
        # Add background contour
        if field_data and 'X' in field_data:
            contour = ax.contourf(field_data['X'], field_data['Y'], field_data['data'], 
                                alpha=0.3, cmap=self.settings.colormap)
            if self.settings.show_colorbar:
                plt.colorbar(contour, ax=ax, label=self.settings.field_type.value)
                
    def _create_line_plot(self, ax, field_data):
        """Create line plot along a specified line."""
        # For now, create a simple x-direction line plot
        if field_data and 'X' in field_data:
            X, Y, data = field_data['X'], field_data['Y'], field_data['data']
            
            # Extract data along centerline (y=0 or closest)
            center_idx = np.argmin(np.abs(Y[:, 0]))
            x_line = X[center_idx, :]
            data_line = data[center_idx, :]
            
            ax.plot(x_line, data_line, 'b-', linewidth=2)
            ax.set_xlabel("X [m]")
            ax.set_ylabel(self.settings.field_type.value)
            ax.grid(True)
            
    def _add_geometry_overlay(self, ax):
        """Add geometry overlay to plot."""
        if not self.geometry_data:
            return
            
        # Draw nozzle geometry
        if 'upper_wall' in self.geometry_data:
            upper_wall = self.geometry_data['upper_wall']
            ax.plot([p[0] for p in upper_wall], [p[1] for p in upper_wall], 
                   'k-', linewidth=2, label='Upper wall')
                   
        if 'lower_wall' in self.geometry_data:
            lower_wall = self.geometry_data['lower_wall']
            ax.plot([p[0] for p in lower_wall], [p[1] for p in lower_wall], 
                   'k-', linewidth=2, label='Lower wall')
                   
    def _add_mesh_overlay(self, ax):
        """Add mesh overlay to plot."""
        if not self.mesh_data:
            return
            
        vertices = np.array(self.mesh_data['vertices'])
        elements = self.mesh_data['elements']
        
        # Draw mesh edges
        for elem in elements:
            if len(elem) >= 3:
                polygon = vertices[elem]
                # Close the polygon
                polygon = np.vstack([polygon, polygon[0]])
                ax.plot(polygon[:, 0], polygon[:, 1], 'k-', alpha=0.3, linewidth=0.5)
                
    def extract_line_data(self, start_point: Tuple[float, float], 
                         end_point: Tuple[float, float], 
                         field_type: FieldType, num_points: int = 100):
        """Extract field data along a line."""
        # Create line points
        x = np.linspace(start_point[0], end_point[0], num_points)
        y = np.linspace(start_point[1], end_point[1], num_points)
        
        # Interpolate field data at line points
        field_data = self._get_field_data(field_type)
        
        if field_data and 'X' in field_data:
            from scipy.interpolate import griddata
            
            # Flatten grid data
            X_flat = field_data['X'].flatten()
            Y_flat = field_data['Y'].flatten()
            data_flat = field_data['data'].flatten()
            
            # Interpolate
            line_values = griddata((X_flat, Y_flat), data_flat, (x, y), method='linear')
            
            return {
                'x': x,
                'y': y,
                'values': line_values,
                'field_type': field_type
            }
            
        return None
        
    def add_probe(self, location: Tuple[float, float]):
        """Add a probe point for time series data."""
        probe = ProbeData(
            location=location,
            time_series={},
            time_values=[]
        )
        self.probes.append(probe)
        return len(self.probes) - 1  # Return probe index
        
    def calculate_derived_quantities(self):
        """Calculate derived quantities like wall shear stress."""
        # This would calculate additional fields from primary fields
        # For now, return dummy data
        return {
            'wall_shear_stress': np.random.random(100),
            'vorticity': np.random.random(100),
            'pressure_coefficient': np.random.random(100)
        }
        
    def export_data(self, filename: str, field_type: FieldType):
        """Export field data to file."""
        field_data = self._get_field_data(field_type)
        
        if field_data:
            export_data = {
                'field_type': field_type.value,
                'data': field_data
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
    def generate_report(self, output_file: str):
        """Generate analysis report."""
        report = {
            'case_directory': self.case_directory,
            'mesh_info': self.mesh_data.get('mesh_info', {}) if self.mesh_data else {},
            'available_fields': list(self.results_data.keys()) if self.results_data else [],
            'derived_quantities': self.calculate_derived_quantities(),
            'probe_data': [probe.location for probe in self.probes]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
