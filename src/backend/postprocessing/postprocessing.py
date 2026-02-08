"""
Post-processing Module

Handles visualization and analysis of SU2 CFD simulation results,
including field plots, contours, streamlines, and data extraction.
"""

import os
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False


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
    """Main results processing and visualization class for SU2 output."""
    
    def __init__(self):
        self.case_directory = ""
        self.results_data = {}
        self.mesh_data = None
        self.geometry_data = None
        self.settings = VisualizationSettings()
        self.probes: List[ProbeData] = []
        self.history_data = None  # SU2 convergence history
        
    def load_case(self, case_directory: str):
        """Load SU2 case results."""
        self.case_directory = case_directory
        
        # Load mesh data
        self._load_mesh_data()
        
        # Load field data from SU2 output
        self._load_field_data()
        
        # Load geometry data if available
        self._load_geometry_data()
        
        # Load convergence history
        self._load_history_data()
        
    def _load_mesh_data(self):
        """Load mesh data from SU2 case."""
        # Try JSON mesh data first
        mesh_file = os.path.join(self.case_directory, "mesh_data.json")
        if os.path.exists(mesh_file):
            with open(mesh_file, 'r') as f:
                self.mesh_data = json.load(f)
            return
            
        # Try to read from SU2 mesh file
        self._read_su2_mesh()
            
    def _load_field_data(self):
        """Load field data from SU2 output files."""
        import glob
        
        # Priority 1: Surface CSV file (reliable SU2 output)
        surface_csv = os.path.join(self.case_directory, "surface_flow.csv")
        if os.path.exists(surface_csv):
            self._load_su2_surface_csv(surface_csv)
            
        # Priority 2: VTU files (Paraview format from SU2)
        vtu_files = glob.glob(os.path.join(self.case_directory, "flow*.vtu"))
        if vtu_files:
            # Sort to get latest (highest iteration number)
            vtu_files.sort()
            self._load_vtu_solution(vtu_files[-1])
            return
            
        # Priority 3: Restart CSV if exists
        restart_file = os.path.join(self.case_directory, "restart_flow.csv")
        if os.path.exists(restart_file):
            self._load_su2_restart_csv(restart_file)
            return
            
        # Priority 4: Legacy VTK format
        vtk_files = glob.glob(os.path.join(self.case_directory, "*.vtk"))
        if vtk_files:
            self._load_vtk_solution(vtk_files[-1])
            return
            
    def _load_su2_restart_csv(self, filepath: str):
        """Load SU2 restart CSV file."""
        if HAS_PANDAS:
            df = pd.read_csv(filepath)
            self.results_data['latest'] = {}
            
            # Map SU2 field names to internal names
            field_mapping = {
                'Momentum_x': 'rhoU_x',
                'Momentum_y': 'rhoU_y',
                'Momentum_z': 'rhoU_z',
                'Velocity_x': 'U_x',
                'Velocity_y': 'U_y',
                'Velocity_z': 'U_z',
                'Pressure': 'p',
                'Temperature': 'T',
                'Density': 'rho',
                'Mach': 'Ma',
                'TKE': 'k',
                'Omega': 'omega',
                'Dissipation': 'epsilon',
                'Nu_Tilde': 'nuTilde'
            }
            
            for su2_name, internal_name in field_mapping.items():
                if su2_name in df.columns:
                    self.results_data['latest'][internal_name] = {
                        'internal_field': df[su2_name].values
                    }
                    
            # Reconstruct velocity vector if components available
            if 'Velocity_x' in df.columns:
                u_x = df['Velocity_x'].values if 'Velocity_x' in df.columns else np.zeros(len(df))
                u_y = df['Velocity_y'].values if 'Velocity_y' in df.columns else np.zeros(len(df))
                u_z = df.get('Velocity_z', pd.Series(np.zeros(len(df)))).values
                
                velocity = np.column_stack([u_x, u_y, u_z])
                self.results_data['latest']['U'] = {
                    'internal_field': velocity,
                    'boundary_fields': {}
                }
                
            # Load coordinates if available
            if 'x' in df.columns and 'y' in df.columns:
                coords = np.column_stack([df['x'].values, df['y'].values])
                self.results_data['latest']['_coordinates'] = coords
        else:
            # Fall back to csv module
            self._load_csv_without_pandas(filepath)
            
    def _load_csv_without_pandas(self, filepath: str):
        """Load CSV file without pandas."""
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows:
            return
            
        self.results_data['latest'] = {}
        
        # Extract coordinates
        if 'x' in rows[0] and 'y' in rows[0]:
            x = np.array([float(row.get('x', 0)) for row in rows])
            y = np.array([float(row.get('y', 0)) for row in rows])
            self.results_data['latest']['_coordinates'] = np.column_stack([x, y])
            
        # Extract velocity
        if 'Velocity_x' in rows[0]:
            u_x = np.array([float(row.get('Velocity_x', 0)) for row in rows])
            u_y = np.array([float(row.get('Velocity_y', 0)) for row in rows])
            u_z = np.array([float(row.get('Velocity_z', 0)) for row in rows])
            self.results_data['latest']['U'] = {
                'internal_field': np.column_stack([u_x, u_y, u_z]),
                'boundary_fields': {}
            }
            
        # Extract pressure
        if 'Pressure' in rows[0]:
            p = np.array([float(row.get('Pressure', 0)) for row in rows])
            self.results_data['latest']['p'] = {'internal_field': p}
            
        # Extract temperature
        if 'Temperature' in rows[0]:
            T = np.array([float(row.get('Temperature', 0)) for row in rows])
            self.results_data['latest']['T'] = {'internal_field': T}
            
    def _load_su2_surface_csv(self, filepath: str):
        """Load SU2 surface output CSV file."""
        if HAS_PANDAS:
            df = pd.read_csv(filepath)
            self.results_data['surface'] = {}
            
            # Store all columns as fields
            for col in df.columns:
                if col not in ['x', 'y', 'z', 'PointID', 'GlobalIndex']:
                    self.results_data['surface'][col] = df[col].values
                    
            # Store coordinates
            if 'x' in df.columns:
                self.results_data['surface']['_coordinates'] = np.column_stack([
                    df['x'].values,
                    df['y'].values if 'y' in df.columns else np.zeros(len(df))
                ])
    
    def _load_vtk_solution(self, filepath: str):
        """Load legacy VTK solution file."""
        self._load_vtu_solution(filepath)
        
    def _load_vtu_solution(self, filepath: str):
        """Load VTU/VTK solution file using meshio or basic XML parsing."""
        self.results_data['latest'] = {}
        
        # Try meshio first (best option)
        if HAS_MESHIO:
            try:
                mesh = meshio.read(filepath)
                
                # Store point coordinates
                points = mesh.points
                self.results_data['latest']['_coordinates'] = points[:, :2]
                
                # Map SU2 field names to internal names
                field_mapping = {
                    'Momentum': 'rhoU',
                    'Velocity': 'U',
                    'Pressure': 'p',
                    'Temperature': 'T',
                    'Density': 'rho',
                    'Mach': 'Ma',
                    'Turb_Kin_Energy': 'k',
                    'Omega': 'omega',
                }
                
                # Load point data
                for su2_name, data in mesh.point_data.items():
                    internal_name = field_mapping.get(su2_name, su2_name)
                    self.results_data['latest'][internal_name] = {
                        'internal_field': data,
                        'boundary_fields': {}
                    }
                    
                # Build velocity field if momentum available
                if 'Momentum' in mesh.point_data and 'Density' in mesh.point_data:
                    rho = mesh.point_data['Density']
                    momentum = mesh.point_data['Momentum']
                    velocity = momentum / rho[:, np.newaxis] if len(momentum.shape) > 1 else momentum / rho
                    self.results_data['latest']['U'] = {
                        'internal_field': velocity,
                        'boundary_fields': {}
                    }
                elif 'Velocity' in mesh.point_data:
                    self.results_data['latest']['U'] = {
                        'internal_field': mesh.point_data['Velocity'],
                        'boundary_fields': {}
                    }
                    
                return
            except Exception as e:
                print(f"Warning: meshio failed to load {filepath}: {e}")
        
        # Fallback: Basic XML parsing for VTU files
        if filepath.endswith('.vtu'):
            try:
                self._parse_vtu_xml(filepath)
                return
            except Exception as e:
                print(f"Warning: VTU XML parsing failed: {e}")
                
        # Legacy VTK parser
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            if 'POINTS' in content:
                lines = content.split('\n')
                points_idx = next(i for i, line in enumerate(lines) if 'POINTS' in line)
                num_points = int(lines[points_idx].split()[1])
                
                points = []
                idx = points_idx + 1
                while len(points) < num_points:
                    values = lines[idx].split()
                    for i in range(0, len(values), 3):
                        if len(points) < num_points:
                            points.append([
                                float(values[i]),
                                float(values[i+1]) if i+1 < len(values) else 0,
                                float(values[i+2]) if i+2 < len(values) else 0
                            ])
                    idx += 1
                    
                self.results_data['latest']['_coordinates'] = np.array(points)[:, :2]
                
        except Exception as e:
            print(f"Warning: Could not parse VTK file: {e}")
            
    def _parse_vtu_xml(self, filepath: str):
        """Parse VTU file using XML parsing (fallback if meshio unavailable)."""
        import xml.etree.ElementTree as ET
        import base64
        import struct
        
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Find the Piece element
        piece = root.find('.//Piece')
        if piece is None:
            return
            
        # Get points
        points_elem = piece.find('.//Points/DataArray')
        if points_elem is not None:
            points_text = points_elem.text.strip()
            # Try to parse as text (ASCII format)
            try:
                values = [float(v) for v in points_text.split()]
                num_points = len(values) // 3
                points = np.array(values).reshape(num_points, 3)
                self.results_data['latest']['_coordinates'] = points[:, :2]
            except:
                pass
                
        # Get point data
        for data_array in piece.findall('.//PointData/DataArray'):
            name = data_array.get('Name', 'unknown')
            try:
                values_text = data_array.text.strip()
                values = np.array([float(v) for v in values_text.split()])
                
                # Check if vector (3 components)
                num_components = int(data_array.get('NumberOfComponents', 1))
                if num_components > 1:
                    values = values.reshape(-1, num_components)
                    
                self.results_data['latest'][name] = {
                    'internal_field': values,
                    'boundary_fields': {}
                }
            except:
                pass
            
    def _load_history_data(self):
        """Load SU2 convergence history."""
        history_file = os.path.join(self.case_directory, "history.csv")
        if os.path.exists(history_file):
            if HAS_PANDAS:
                self.history_data = pd.read_csv(history_file)
            else:
                with open(history_file, 'r') as f:
                    reader = csv.DictReader(f)
                    self.history_data = list(reader)
                    
    def _read_su2_mesh(self):
        """Read SU2 mesh file."""
        # Look for .su2 mesh file
        import glob
        mesh_files = glob.glob(os.path.join(self.case_directory, "*.su2"))
        
        if not mesh_files:
            # Create dummy mesh
            self.mesh_data = {
                'vertices': np.random.random((100, 2)),
                'elements': [[i, i+1, i+2] for i in range(0, 97, 3)],
                'element_type': 'triangle'
            }
            return
            
        mesh_file = mesh_files[0]
        
        vertices = []
        elements = []
        
        with open(mesh_file, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('NPOIN='):
                num_points = int(line.split('=')[1].split()[0])
                for j in range(num_points):
                    i += 1
                    parts = lines[i].strip().split()
                    x, y = float(parts[0]), float(parts[1])
                    vertices.append([x, y])
                    
            elif line.startswith('NELEM='):
                num_elems = int(line.split('=')[1].split()[0])
                for j in range(num_elems):
                    i += 1
                    parts = lines[i].strip().split()
                    elem_type = int(parts[0])
                    # Triangle: 5, Quad: 9
                    if elem_type == 5:  # Triangle
                        elements.append([int(parts[1]), int(parts[2]), int(parts[3])])
                    elif elem_type == 9:  # Quad
                        elements.append([int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])])
                        
            i += 1
            
        self.mesh_data = {
            'vertices': np.array(vertices),
            'elements': elements,
            'element_type': 'mixed'
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
        """Create contour plot using mesh triangulation for proper geometry boundaries."""
        import matplotlib.tri as mtri
        
        # Priority 1: Use mesh-based triangulation (respects geometry boundaries)
        if self.mesh_data and 'vertices' in self.mesh_data:
            vertices = np.array(self.mesh_data['vertices'])
            elements = self.mesh_data.get('elements', [])
            
            # Get field values
            if 'values' in field_data:
                values = field_data['values']
            elif 'data' in field_data:
                # Structured data - flatten
                values = field_data['data'].flatten()
            else:
                values = None
                
            if values is not None and len(values) == len(vertices):
                # Convert elements to triangles
                triangles = []
                for elem in elements:
                    if len(elem) == 3:
                        triangles.append(elem)
                    elif len(elem) == 4:
                        # Split quad into two triangles
                        triangles.append((elem[0], elem[1], elem[2]))
                        triangles.append((elem[0], elem[2], elem[3]))
                        
                if triangles:
                    try:
                        triang = mtri.Triangulation(
                            vertices[:, 0], vertices[:, 1], triangles=triangles
                        )
                        
                        if self.settings.filled_contours:
                            contour = ax.tricontourf(triang, values, 
                                                    levels=self.settings.num_levels,
                                                    cmap=self.settings.colormap)
                        else:
                            contour = ax.tricontour(triang, values,
                                                   levels=self.settings.num_levels,
                                                   cmap=self.settings.colormap)
                                                   
                        if self.settings.show_colorbar:
                            plt.colorbar(contour, ax=ax, label=self.settings.field_type.value)
                        return
                    except Exception as e:
                        print(f"Triangulation failed: {e}, falling back to scatter")
                        
                # Fallback: Delaunay triangulation
                try:
                    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1])
                    if self.settings.filled_contours:
                        contour = ax.tricontourf(triang, values,
                                                levels=self.settings.num_levels,
                                                cmap=self.settings.colormap)
                    else:
                        contour = ax.tricontour(triang, values,
                                               levels=self.settings.num_levels,
                                               cmap=self.settings.colormap)
                    if self.settings.show_colorbar:
                        plt.colorbar(contour, ax=ax, label=self.settings.field_type.value)
                    return
                except Exception as e:
                    print(f"Delaunay triangulation failed: {e}")
                    
        # Fallback 2: Structured grid data
        if 'X' in field_data and 'Y' in field_data:
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
            # Unstructured data with vertices - use triangulation
            vertices = field_data['vertices']
            values = field_data['values']
            
            try:
                triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1])
                if self.settings.filled_contours:
                    contour = ax.tricontourf(triang, values, 
                                            levels=self.settings.num_levels,
                                            cmap=self.settings.colormap)
                else:
                    contour = ax.tricontour(triang, values,
                                           levels=self.settings.num_levels,
                                           cmap=self.settings.colormap)
                if self.settings.show_colorbar:
                    plt.colorbar(contour, ax=ax, label=self.settings.field_type.value)
            except Exception:
                # Ultimate fallback: scatter plot
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
