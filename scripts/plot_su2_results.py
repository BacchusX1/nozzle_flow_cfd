#!/usr/bin/env python3
"""
SU2 Results Plotter

Generates properly bounded contour plots from SU2 simulation results.
Uses mesh triangulation to ensure plots respect the nozzle geometry boundaries.

Usage:
    python scripts/plot_su2_results.py --case /path/to/case
    python scripts/plot_su2_results.py --case case/ --field Pressure --output results.png

    e.g. python scripts/plot_su2_results.py --case case/ --all

Author: Nozzle CFD Design Tool
"""

import os
import sys
import argparse
import glob
import csv
import struct
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection


class SU2ResultsPlotter:
    """Plot SU2 CFD results with proper geometry-bounded contours."""
    
    def __init__(self, case_dir: str):
        """Initialize plotter with case directory.
        
        Args:
            case_dir: Path to SU2 case directory containing mesh and results
        """
        self.case_dir = Path(case_dir).resolve()
        self.mesh_nodes: Optional[np.ndarray] = None
        self.mesh_elements: List[Tuple] = []
        self.triangles: List[Tuple[int, int, int]] = []
        self.field_data: Dict[str, np.ndarray] = {}
        self.history_data: Dict[str, np.ndarray] = {}
        self.config: Dict[str, str] = {}
        
    def load_mesh(self) -> bool:
        """Load mesh from SU2 mesh file.
        
        Returns:
            True if mesh loaded successfully
        """
        mesh_files = list(self.case_dir.glob("*.su2"))
        if not mesh_files:
            print(f"Warning: No .su2 mesh file found in {self.case_dir}")
            return False
            
        mesh_file = mesh_files[0]
        print(f"Loading mesh: {mesh_file.name}")
        
        nodes = []
        elements = []
        ndim = 2
        
        with open(mesh_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('NDIME='):
                ndim = int(line.split('=')[1].strip())
                
            elif line.startswith('NPOIN='):
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
                        elif elem_type == 9:  # Quadrilateral
                            elements.append((int(parts[1]), int(parts[2]), 
                                           int(parts[3]), int(parts[4])))
                    i += 1
                continue
                    
            i += 1
            
        self.mesh_nodes = np.array(nodes)
        self.mesh_elements = elements
        
        # Convert to triangles for plotting
        self._triangulate()
        
        print(f"  Loaded {len(nodes)} nodes, {len(elements)} elements, "
              f"{len(self.triangles)} triangles")
        return True
        
    def _triangulate(self):
        """Convert mesh elements to triangles."""
        self.triangles = []
        for elem in self.mesh_elements:
            if len(elem) == 3:
                self.triangles.append(elem)
            elif len(elem) == 4:
                # Split quad into two triangles
                self.triangles.append((elem[0], elem[1], elem[2]))
                self.triangles.append((elem[0], elem[2], elem[3]))
                
    def load_solution_vtu(self, vtu_file: Optional[str] = None) -> bool:
        """Load solution from VTU file with binary appended data support.
        
        Args:
            vtu_file: Path to VTU file. If None, searches for flow*.vtu
            
        Returns:
            True if solution loaded successfully
        """
        if vtu_file is None:
            # Prefer volume mesh (flow.vtu) over surface mesh (surface_flow.vtu)
            vtu_files = sorted(self.case_dir.glob("flow.vtu"))
            if not vtu_files:
                # Fall back to any flow*.vtu but exclude surface_flow
                vtu_files = [f for f in self.case_dir.glob("flow*.vtu") 
                            if 'surface' not in f.name.lower()]
            if not vtu_files:
                vtu_files = sorted(self.case_dir.glob("flow*.vtu"))
            if not vtu_files:
                return False
            vtu_file = vtu_files[-1]  # Get latest
        else:
            vtu_file = Path(vtu_file)
            
        print(f"Loading solution: {vtu_file.name}")
        
        # Try binary VTU parser first (handles SU2 output format)
        if self._parse_vtu_binary(vtu_file):
            return True
                
        # Fallback: basic XML parsing
        return self._parse_vtu_basic(vtu_file)
    
    def _parse_vtu_binary(self, vtu_file: Path) -> bool:
        """Parse VTU file with binary appended data (SU2 format).
        
        Args:
            vtu_file: Path to VTU file
            
        Returns:
            True if parsed successfully
        """
        try:
            with open(vtu_file, 'rb') as f:
                content = f.read()
            
            # Check for appended data section
            if b'<AppendedData' not in content:
                return False  # Not a binary VTU, fall back to XML parser
                
            # Parse XML header to get array info
            xml_end = content.find(b'<AppendedData')
            header = content[:xml_end].decode('utf-8', errors='ignore')
            
            # Determine header type (UInt32 or UInt64)
            if 'header_type="UInt64"' in header:
                header_dtype = '<Q'  # 8 bytes
                header_size = 8
            else:
                header_dtype = '<I'  # 4 bytes
                header_size = 4
            
            # Find binary data start
            marker = b'<AppendedData encoding="raw">'
            pos = content.find(marker)
            if pos < 0:
                return False
            underscore_pos = content.find(b'_', pos)
            data_start = underscore_pos + 1
            binary_data = content[data_start:]
            
            # Helper function to extract offset from header
            def get_offset(name: str) -> Optional[int]:
                pattern = rf'Name="{name}"[^>]*offset="(\d+)"'
                match = re.search(pattern, header)
                return int(match.group(1)) if match else None
            
            # Helper to get type info - search for DataArray containing the name
            def get_type_info(name: str) -> Tuple[Optional[str], int]:
                # Find the full DataArray element for this field
                pattern = rf'<DataArray[^>]*Name="{name}"[^>]*/>'
                match = re.search(pattern, header)
                if not match:
                    return None, 1
                
                element = match.group(0)
                
                # Extract type from the element
                type_match = re.search(r'type="([^"]+)"', element)
                dtype_str = type_match.group(1) if type_match else None
                
                # Extract NumberOfComponents from the element
                ncomp_match = re.search(r'NumberOfComponents\s*=\s*"(\d+)"', element)
                ncomp = int(ncomp_match.group(1)) if ncomp_match else 1
                
                return dtype_str, ncomp
            
            # VTK type mapping
            type_map = {
                'Float32': np.float32,
                'Float64': np.float64,
                'Int32': np.int32,
                'Int64': np.int64,
                'UInt8': np.uint8,
                'UInt32': np.uint32,
            }
            
            def read_array(offset: int, dtype_str: str) -> np.ndarray:
                dtype = type_map.get(dtype_str, np.float32)
                block_size = struct.unpack(header_dtype, 
                    binary_data[offset:offset+header_size])[0]
                return np.frombuffer(
                    binary_data[offset+header_size:offset+header_size+block_size], 
                    dtype=dtype
                )
            
            # Read points (always at offset 0 for Points DataArray)
            pts = read_array(0, 'Float32').reshape(-1, 3)
            self.mesh_nodes = pts[:, :2]
            
            # Read connectivity
            conn_offset = get_offset('connectivity')
            offs_offset = get_offset('offsets')
            types_offset = get_offset('types')
            
            if conn_offset is None or offs_offset is None or types_offset is None:
                print("  Warning: Missing cell data, falling back to Delaunay")
                self.triangles = []
                return len(self.mesh_nodes) > 0
            
            conn = read_array(conn_offset, 'Int32')
            offsets = read_array(offs_offset, 'Int32')
            cell_types = read_array(types_offset, 'UInt8')
            
            # Build triangles from cells
            self.triangles = []
            prev_off = 0
            for i, off in enumerate(offsets):
                cell = conn[prev_off:off]
                cell_type = cell_types[i]
                if cell_type == 5:  # Triangle
                    self.triangles.append(tuple(cell))
                elif cell_type == 9:  # Quad - split into 2 triangles
                    self.triangles.append((cell[0], cell[1], cell[2]))
                    self.triangles.append((cell[0], cell[2], cell[3]))
                prev_off = off
            
            # Read all point data fields
            # Find all DataArray elements in PointData
            point_data_match = re.search(r'<PointData>(.*?)</PointData>', header, re.DOTALL)
            if point_data_match:
                point_data_section = point_data_match.group(1)
                # Match DataArray with any closing (handles newlines)
                for match in re.finditer(r'<DataArray[^>]*Name="([^"]+)"[^>]*/>', point_data_section):
                    field_name = match.group(1)
                    offset = get_offset(field_name)
                    dtype_str, ncomp = get_type_info(field_name)
                    
                    if offset is not None and dtype_str:
                        data = read_array(offset, dtype_str)
                        if ncomp > 1:
                            data = data.reshape(-1, ncomp)
                            self.field_data[field_name] = data
                            # Also store magnitude for vector fields
                            self.field_data[f"{field_name}_Magnitude"] = np.linalg.norm(data, axis=1)
                        else:
                            self.field_data[field_name] = data
            
            print(f"  Loaded {len(self.mesh_nodes)} nodes, {len(self.triangles)} triangles")
            print(f"  Fields: {list(self.field_data.keys())}")
            return True
            
        except Exception as e:
            print(f"  Binary VTU parse error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def _parse_vtu_basic(self, vtu_file: Path) -> bool:
        """Basic VTU parsing without meshio."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(vtu_file)
            root = tree.getroot()
            
            piece = root.find('.//Piece')
            if piece is None:
                return False
                
            # ALWAYS parse points from VTU to ensure mesh matches field data
            points_elem = piece.find('.//Points/DataArray')
            if points_elem is not None and points_elem.text:
                values = [float(v) for v in points_elem.text.strip().split()]
                n_points = len(values) // 3
                self.mesh_nodes = np.array(values).reshape(n_points, 3)[:, :2]
                self.triangles = []  # Will use Delaunay if no cells found
                
            # Parse cells if available
            cells_elem = piece.find('.//Cells')
            if cells_elem is not None:
                connectivity = cells_elem.find("DataArray[@Name='connectivity']")
                offsets = cells_elem.find("DataArray[@Name='offsets']")
                types = cells_elem.find("DataArray[@Name='types']")
                
                if connectivity is not None and connectivity.text:
                    conn = [int(v) for v in connectivity.text.strip().split()]
                    if offsets is not None and offsets.text:
                        offs = [int(v) for v in offsets.text.strip().split()]
                        prev_off = 0
                        for off in offs:
                            cell = conn[prev_off:off]
                            if len(cell) == 3:
                                self.triangles.append(tuple(cell))
                            elif len(cell) == 4:
                                self.triangles.append((cell[0], cell[1], cell[2]))
                                self.triangles.append((cell[0], cell[2], cell[3]))
                            prev_off = off
                    
            # Parse point data
            for data_array in piece.findall('.//PointData/DataArray'):
                name = data_array.get('Name', 'unknown')
                if data_array.text:
                    try:
                        values = np.array([float(v) for v in data_array.text.strip().split()])
                        n_comp = int(data_array.get('NumberOfComponents', 1))
                        if n_comp > 1:
                            values = values.reshape(-1, n_comp)
                            self.field_data[name] = values
                            self.field_data[f"{name}_Magnitude"] = np.linalg.norm(values, axis=1)
                        else:
                            self.field_data[name] = values
                    except:
                        pass
                        
            return len(self.field_data) > 0
            
        except Exception as e:
            print(f"  VTU parse error: {e}")
            return False
            
    def load_surface_csv(self, csv_file: Optional[str] = None) -> bool:
        """Load surface solution from CSV file.
        
        Args:
            csv_file: Path to surface CSV. If None, searches for surface_flow.csv
            
        Returns:
            True if loaded successfully
        """
        if csv_file is None:
            csv_files = list(self.case_dir.glob("surface_flow*.csv"))
            if not csv_files:
                return False
            csv_file = csv_files[-1]
        else:
            csv_file = Path(csv_file)
            
        print(f"Loading surface data: {csv_file.name}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
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
                self.field_data[key] = np.array(values)
                
            # Extract coordinates from CSV for surface plotting
            x_col = None
            y_col = None
            for col in header:
                if col.lower() in ('x', 'x-coordinate', 'points:0'):
                    x_col = col
                elif col.lower() in ('y', 'y-coordinate', 'points:1'):
                    y_col = col
                    
            if x_col and y_col:
                x = self.field_data[x_col]
                y = self.field_data[y_col]
                # Update mesh_nodes with CSV coordinates (for surface plotting)
                self.mesh_nodes = np.column_stack([x, y])
                self.triangles = []  # Will use Delaunay triangulation
                print(f"  Loaded {len(self.mesh_nodes)} surface points")
                
            print(f"  Fields: {[k for k in self.field_data.keys() if k not in (x_col, y_col, 'PointID')]}")
            return True
            
        except Exception as e:
            print(f"  CSV parse error: {e}")
            return False
            
    def load_history(self) -> bool:
        """Load convergence history from history.csv.
        
        Returns:
            True if loaded successfully
        """
        history_file = self.case_dir / "history.csv"
        if not history_file.exists():
            return False
            
        print(f"Loading history: {history_file.name}")
        
        try:
            with open(history_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                header = [col.strip().strip('"') for col in next(reader)]
                
                data = {col: [] for col in header}
                for row in reader:
                    if not row or row[0].startswith('#'):
                        continue
                    for i, val in enumerate(row):
                        if i < len(header):
                            try:
                                data[header[i]].append(float(val))
                            except ValueError:
                                data[header[i]].append(np.nan)
                                
            self.history_data = {k: np.array(v) for k, v in data.items()}
            return True
            
        except Exception as e:
            print(f"  History parse error: {e}")
            return False
            
    def get_triangulation(self) -> Optional[mtri.Triangulation]:
        """Get matplotlib triangulation for plotting.
        
        Returns:
            Triangulation object or None
        """
        if self.mesh_nodes is None or len(self.mesh_nodes) < 3:
            return None
            
        if not self.triangles:
            # Fallback to Delaunay
            try:
                return mtri.Triangulation(self.mesh_nodes[:, 0], self.mesh_nodes[:, 1])
            except:
                return None
                
        try:
            return mtri.Triangulation(
                self.mesh_nodes[:, 0],
                self.mesh_nodes[:, 1],
                triangles=self.triangles
            )
        except Exception as e:
            print(f"Warning: Triangulation error: {e}")
            # Fallback to Delaunay
            try:
                return mtri.Triangulation(self.mesh_nodes[:, 0], self.mesh_nodes[:, 1])
            except:
                return None
                
    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        """Get field data by name (case-insensitive, partial match).
        
        Args:
            field_name: Field name to search for
            
        Returns:
            Field data array or None
        """
        # Exact match
        if field_name in self.field_data:
            return self.field_data[field_name]
            
        # Case-insensitive match
        field_lower = field_name.lower()
        for key, value in self.field_data.items():
            if key.lower() == field_lower:
                return value
                
        # Partial match
        for key, value in self.field_data.items():
            if field_lower in key.lower():
                return value
                
        return None
        
    def plot_contour(self, field_name: str, ax=None, 
                     levels: int = 20, cmap: str = 'jet',
                     show_colorbar: bool = True,
                     show_mesh: bool = False) -> plt.Figure:
        """Create contour plot of a field.
        
        Args:
            field_name: Name of field to plot
            ax: Matplotlib axes (created if None)
            levels: Number of contour levels
            cmap: Colormap name
            show_colorbar: Whether to show colorbar
            show_mesh: Whether to overlay mesh
            
        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        else:
            fig = ax.get_figure()
            
        # Get triangulation
        triang = self.get_triangulation()
        if triang is None:
            ax.text(0.5, 0.5, "No mesh data available", ha='center', va='center',
                   transform=ax.transAxes)
            return fig
            
        # Get field data
        field = self.get_field(field_name)
        if field is None:
            ax.text(0.5, 0.5, f"Field '{field_name}' not found\n"
                   f"Available: {list(self.field_data.keys())}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
            
        # Handle vector fields (use magnitude)
        if len(field.shape) > 1:
            field = np.linalg.norm(field, axis=1)
            
        # Ensure field matches mesh nodes
        if len(field) != len(self.mesh_nodes):
            ax.text(0.5, 0.5, f"Field size mismatch: {len(field)} vs {len(self.mesh_nodes)} nodes",
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Use tripcolor for cell-based coloring (respects mesh boundaries exactly)
        # This is the key fix - tripcolor only colors actual mesh cells,
        # unlike tricontourf which can fill the convex hull
        if self.triangles:
            # Use tripcolor with shading='gouraud' for smooth interpolation
            tripcolor = ax.tripcolor(triang, field, shading='gouraud', 
                                     cmap=cmap, edgecolors='none')
            
            if show_colorbar:
                cbar = plt.colorbar(tripcolor, ax=ax, shrink=0.8)
                # Add units to colorbar label for common fields
                label = field_name
                units = self._get_field_units(field_name)
                if units:
                    label = f'{field_name} [{units}]'
                cbar.set_label(label)
        else:
            # Fallback to tricontourf if no explicit triangulation
            contourf = ax.tricontourf(triang, field, levels=levels, cmap=cmap)
            if show_colorbar:
                cbar = plt.colorbar(contourf, ax=ax, shrink=0.8)
                cbar.set_label(field_name)
            
        # Add mesh overlay
        if show_mesh:
            ax.triplot(triang, 'k-', linewidth=0.3, alpha=0.3)
        
        # Draw boundary outline for clearer visualization
        self._draw_boundary(ax)
            
        # Set equal aspect and labels
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'{field_name}')
        
        return fig
    
    def _get_field_units(self, field_name: str) -> Optional[str]:
        """Get units for common field names."""
        units_map = {
            'pressure': 'Pa',
            'temperature': 'K',
            'density': 'kg/m³',
            'velocity': 'm/s',
            'velocity_magnitude': 'm/s',
            'mach': '',
            'momentum': 'kg/(m²·s)',
            'momentum_magnitude': 'kg/(m²·s)',
            'energy': 'J/m³',
        }
        return units_map.get(field_name.lower(), None)
    
    def _draw_boundary(self, ax):
        """Draw the mesh boundary outline on the plot."""
        if self.mesh_nodes is None or not self.triangles:
            return
        
        # Find boundary edges (edges that appear in only one triangle)
        from collections import Counter
        edge_count = Counter()
        
        for tri in self.triangles:
            # Each triangle has 3 edges
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]])),
            ]
            for edge in edges:
                edge_count[edge] += 1
        
        # Boundary edges appear exactly once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        # Draw boundary edges
        for edge in boundary_edges:
            x = [self.mesh_nodes[edge[0], 0], self.mesh_nodes[edge[1], 0]]
            y = [self.mesh_nodes[edge[0], 1], self.mesh_nodes[edge[1], 1]]
            ax.plot(x, y, 'k-', linewidth=0.8, alpha=0.8)
        
    def plot_convergence(self, ax=None) -> plt.Figure:
        """Plot convergence history.
        
        Args:
            ax: Matplotlib axes (created if None)
            
        Returns:
            Matplotlib figure
        """
        if not self.history_data:
            self.load_history()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        else:
            fig = ax.get_figure()
            
        if not self.history_data:
            ax.text(0.5, 0.5, "No history data available", ha='center', va='center',
                   transform=ax.transAxes)
            return fig
            
        # Find iteration column
        iter_col = None
        for key in ['Inner_Iter', 'Outer_Iter', 'Time_Iter', 'Iteration']:
            if key in self.history_data:
                iter_col = key
                break
                
        if iter_col is None:
            iters = np.arange(len(list(self.history_data.values())[0]))
        else:
            iters = self.history_data[iter_col]
            
        # Plot residuals
        plotted = 0
        for key, values in self.history_data.items():
            if 'rms' in key.lower() or 'res' in key.lower():
                if len(values) == len(iters):
                    ax.semilogy(iters, np.abs(values), label=key, linewidth=1.5)
                    plotted += 1
                    if plotted >= 6:  # Limit to 6 residuals
                        break
                        
        if plotted == 0:
            ax.text(0.5, 0.5, "No residual data found", ha='center', va='center',
                   transform=ax.transAxes)
            return fig
            
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual (log scale)')
        ax.set_title('Convergence History')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def plot_all(self, output_dir: Optional[str] = None, 
                 fields: Optional[List[str]] = None) -> List[str]:
        """Generate all plots and save to output directory.
        
        Args:
            output_dir: Directory to save plots (default: case_dir/plots)
            fields: List of fields to plot (default: auto-detect)
            
        Returns:
            List of saved file paths
        """
        if output_dir is None:
            output_dir = self.case_dir / "plots"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        
        # Convergence plot
        fig = self.plot_convergence()
        path = output_dir / "convergence.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(str(path))
        print(f"  Saved: {path.name}")
        
        # Field plots
        if fields is None:
            # Auto-detect common fields
            fields = []
            for pattern in ['Pressure', 'Mach', 'Temperature', 'Density', 
                           'Velocity_Magnitude', 'Momentum_Magnitude']:
                if self.get_field(pattern) is not None:
                    fields.append(pattern)
            
            # Also check for 'Velocity' vector and add magnitude if not already present
            if 'Velocity_Magnitude' not in fields and self.get_field('Velocity') is not None:
                fields.append('Velocity')
                    
        for field_name in fields:
            try:
                fig = self.plot_contour(field_name)
                # Clean filename
                safe_name = field_name.replace('/', '_').replace('\\', '_')
                path = output_dir / f"{safe_name}.png"
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(str(path))
                print(f"  Saved: {path.name}")
            except Exception as e:
                print(f"  Error plotting {field_name}: {e}")
                
        return saved_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot SU2 CFD results with proper geometry boundaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_su2_results.py --case case/
  python plot_su2_results.py --case case/ --field Pressure --output pressure.png
  python plot_su2_results.py --case case/ --all
        """
    )
    
    parser.add_argument('--case', '-c', default='.', 
                       help='Path to SU2 case directory (default: current)')
    parser.add_argument('--field', '-f', default='Pressure',
                       help='Field to plot (default: Pressure)')
    parser.add_argument('--output', '-o', 
                       help='Output file path (default: field_name.png)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Generate all standard plots')
    parser.add_argument('--levels', '-l', type=int, default=20,
                       help='Number of contour levels (default: 20)')
    parser.add_argument('--cmap', default='jet',
                       help='Colormap (default: jet)')
    parser.add_argument('--mesh', action='store_true',
                       help='Show mesh overlay')
    parser.add_argument('--show', action='store_true',
                       help='Show plot interactively')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = SU2ResultsPlotter(args.case)
    
    # Load data - try VTU first (contains both mesh and fields)
    vtu_loaded = plotter.load_solution_vtu()
    
    if not vtu_loaded:
        # VTU failed, try CSV + mesh
        plotter.load_mesh()  # Load .su2 mesh
        plotter.load_surface_csv()  # Load surface CSV (will override mesh with CSV coords)
        
    plotter.load_history()
    
    if args.all:
        # Generate all plots
        saved = plotter.plot_all()
        print(f"\nGenerated {len(saved)} plots")
        
    else:
        # Single field plot
        fig = plotter.plot_contour(
            args.field,
            levels=args.levels,
            cmap=args.cmap,
            show_mesh=args.mesh
        )
        
        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"Saved: {args.output}")
        elif args.show:
            plt.show()
        else:
            # Default output
            output = Path(args.case) / "plots" / f"{args.field}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=150, bbox_inches='tight')
            print(f"Saved: {output}")
            
        plt.close(fig)


if __name__ == "__main__":
    main()
