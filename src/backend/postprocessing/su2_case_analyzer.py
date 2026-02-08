"""
SU2 Case Analyzer Module

Parses SU2 output files (CSV history, VTU solution files) for post-processing.
Handles convergence monitoring, field visualization, and flow analysis.
"""

import os
import re
import csv
import json
import glob
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


def parse_history_csv(filepath: str) -> Dict[str, np.ndarray]:
    """Parse SU2 history.csv convergence file.

    Args:
        filepath: Path to history.csv file

    Returns:
        Dictionary mapping column names to numpy arrays of values
    """
    data = defaultdict(list)

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            # SU2 history files use quotes around column names
            reader = csv.reader(f)

            # First line is header
            header = next(reader)
            # Clean up column names (remove quotes and whitespace)
            columns = [col.strip().strip('"') for col in header]

            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                for i, val in enumerate(row):
                    if i < len(columns):
                        try:
                            data[columns[i]].append(float(val))
                        except ValueError:
                            data[columns[i]].append(np.nan)

        return {k: np.array(v) for k, v in data.items()}

    except Exception as e:
        print(f"Error parsing history CSV: {e}")
        return {}


def parse_surface_csv(filepath: str) -> Dict[str, Any]:
    """Parse SU2 surface output CSV file.

    Args:
        filepath: Path to surface CSV file

    Returns:
        Dictionary with 'points' (Nx3 array) and field data arrays
    """
    data = {"points": [], "fields": defaultdict(list)}

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader)
            columns = [col.strip().strip('"') for col in header]

            # Find coordinate columns
            x_idx = next((i for i, c in enumerate(columns) if c.lower() in ("x", "x-coordinate")), None)
            y_idx = next((i for i, c in enumerate(columns) if c.lower() in ("y", "y-coordinate")), None)
            z_idx = next((i for i, c in enumerate(columns) if c.lower() in ("z", "z-coordinate")), None)

            for row in reader:
                if not row:
                    continue

                # Extract point coordinates
                x = float(row[x_idx]) if x_idx is not None else 0.0
                y = float(row[y_idx]) if y_idx is not None else 0.0
                z = float(row[z_idx]) if z_idx is not None else 0.0
                data["points"].append((x, y, z))

                # Extract field values
                for i, col in enumerate(columns):
                    if i not in (x_idx, y_idx, z_idx):
                        try:
                            data["fields"][col].append(float(row[i]))
                        except (ValueError, IndexError):
                            data["fields"][col].append(np.nan)

        data["points"] = np.array(data["points"])
        data["fields"] = {k: np.array(v) for k, v in data["fields"].items()}
        return data

    except Exception as e:
        print(f"Error parsing surface CSV: {e}")
        return {"points": np.array([]), "fields": {}}


class SU2Case:
    """Analyzer for SU2 CFD case results."""

    def __init__(self, case_dir: str):
        self.case_dir = Path(case_dir).resolve()
        self.config_file = "config.cfg"
        self.config = {}
        self.history_data = {}
        self.solution_data = {}
        self.mesh_data = None

    def load_config(self) -> Dict[str, str]:
        """Load and parse SU2 configuration file.

        Returns:
            Dictionary of config key-value pairs
        """
        config_path = self.case_dir / self.config_file
        if not config_path.exists():
            return {}

        config = {}
        try:
            with open(config_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith("%"):
                        continue
                    # Parse KEY= VALUE format
                    if "=" in line:
                        key, value = line.split("=", 1)
                        config[key.strip()] = value.strip()

            self.config = config
            return config

        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def get_solver_type(self) -> Optional[str]:
        """Get solver type from config."""
        if not self.config:
            self.load_config()
        return self.config.get("SOLVER")

    def get_mesh_filename(self) -> Optional[str]:
        """Get mesh filename from config."""
        if not self.config:
            self.load_config()
        return self.config.get("MESH_FILENAME")

    def load_mesh(self) -> Optional[Dict[str, Any]]:
        """Load mesh data from SU2 mesh file."""
        mesh_file = self.get_mesh_filename()
        if not mesh_file:
            return None

        mesh_path = self.case_dir / mesh_file
        if not mesh_path.exists():
            return None

        try:
            from backend.meshing.su2_mesh_converter import SU2MeshConverter
            converter = SU2MeshConverter()
            self.mesh_data = converter.read_su2_mesh(str(mesh_path))
            return self.mesh_data
        except ImportError:
            # Inline minimal mesh reader
            return self._read_mesh_minimal(str(mesh_path))

    def _read_mesh_minimal(self, filepath: str) -> Dict[str, Any]:
        """Minimal SU2 mesh reader."""
        nodes = []
        elements = []
        ndim = 2

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("NDIME="):
                ndim = int(line.split("=")[1].strip())
                i += 1

            elif line.startswith("NPOIN="):
                n_points = int(line.split("=")[1].strip().split()[0])
                i += 1
                for _ in range(n_points):
                    if i >= len(lines):
                        break
                    parts = lines[i].strip().split()
                    if ndim == 2 and len(parts) >= 2:
                        nodes.append((float(parts[0]), float(parts[1])))
                    elif len(parts) >= 3:
                        nodes.append((float(parts[0]), float(parts[1]), float(parts[2])))
                    i += 1

            elif line.startswith("NELEM="):
                n_elem = int(line.split("=")[1].strip())
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
                            elements.append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])))
                    i += 1
            else:
                i += 1

        self.mesh_data = {
            "nodes": nodes,
            "elements": elements,
            "ndim": ndim,
        }
        return self.mesh_data

    def load_solution(self, vtu_path: str = None) -> bool:
        """Load solution data from available output files.
        
        Args:
            vtu_path: Optional specific VTU file path to load. If None, loads latest.
        
        Returns:
            True if solution loaded successfully
        """
        # If specific file provided, load it directly
        if vtu_path and os.path.exists(vtu_path):
            return self._load_vtu_binary(Path(vtu_path))
        
        # Priority 1: VTU files (best for visualization - contains mesh + fields)
        vtu_files = list(self.case_dir.glob("flow.vtu"))
        if not vtu_files:
            vtu_files = [f for f in self.case_dir.glob("flow*.vtu") 
                        if 'surface' not in f.name.lower()]
        if vtu_files:
            if self._load_vtu_binary(vtu_files[-1]):
                return True
        
        # Priority 2: CSV files
        for pattern in ["restart_flow.csv", "flow*.csv", "surface_flow.csv", 
                        "solution.csv"]:
            files = list(self.case_dir.glob(pattern))
            if files:
                self.solution_data = parse_surface_csv(str(files[0]))
                return bool(self.solution_data)
        return False

    def _load_vtu_binary(self, vtu_file: Path) -> bool:
        """Load VTU file with binary appended data (SU2 format).
        
        Args:
            vtu_file: Path to VTU file
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(vtu_file, 'rb') as f:
                content = f.read()
            
            # Check for appended data section
            if b'<AppendedData' not in content:
                return False
                
            # Parse XML header
            xml_end = content.find(b'<AppendedData')
            header = content[:xml_end].decode('utf-8', errors='ignore')
            
            # Determine header type
            if 'header_type="UInt64"' in header:
                header_dtype = '<Q'
                header_size = 8
            else:
                header_dtype = '<I'
                header_size = 4
            
            # Find binary data start
            marker = b'<AppendedData encoding="raw">'
            pos = content.find(marker)
            if pos < 0:
                return False
            underscore_pos = content.find(b'_', pos)
            data_start = underscore_pos + 1
            binary_data = content[data_start:]
            
            # Helper to extract offset
            def get_offset(name: str) -> Optional[int]:
                pattern = rf'Name="{name}"[^>]*offset="(\d+)"'
                match = re.search(pattern, header)
                return int(match.group(1)) if match else None
            
            # Helper to get type info
            def get_type_info(name: str) -> Tuple[Optional[str], int]:
                pattern = rf'<DataArray[^>]*Name="{name}"[^>]*/>' 
                match = re.search(pattern, header)
                if not match:
                    return None, 1
                element = match.group(0)
                type_match = re.search(r'type="([^"]+)"', element)
                dtype_str = type_match.group(1) if type_match else None
                ncomp_match = re.search(r'NumberOfComponents\s*=\s*"(\d+)"', element)
                ncomp = int(ncomp_match.group(1)) if ncomp_match else 1
                return dtype_str, ncomp
            
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
            
            # Read points
            pts = read_array(0, 'Float32').reshape(-1, 3)
            
            # Read connectivity
            conn_offset = get_offset('connectivity')
            offs_offset = get_offset('offsets')
            types_offset = get_offset('types')
            
            triangles = []
            if conn_offset and offs_offset and types_offset:
                conn = read_array(conn_offset, 'Int32')
                offsets = read_array(offs_offset, 'Int32')
                cell_types = read_array(types_offset, 'UInt8')
                
                prev_off = 0
                for i, off in enumerate(offsets):
                    cell = conn[prev_off:off]
                    cell_type = cell_types[i]
                    if cell_type == 5:  # Triangle
                        triangles.append(tuple(cell))
                    elif cell_type == 9:  # Quad
                        triangles.append((cell[0], cell[1], cell[2]))
                        triangles.append((cell[0], cell[2], cell[3]))
                    prev_off = off
            
            # Store mesh data
            self.mesh_data = {
                'nodes': [(p[0], p[1]) for p in pts],
                'elements': triangles,
                'ndim': 2,
                'triangles': triangles,
            }
            
            # Read field data
            point_data_match = re.search(r'<PointData>(.*?)</PointData>', header, re.DOTALL)
            fields = {}
            if point_data_match:
                section = point_data_match.group(1)
                for match in re.finditer(r'<DataArray[^>]*Name="([^"]+)"[^>]*/>', section):
                    field_name = match.group(1)
                    offset = get_offset(field_name)
                    dtype_str, ncomp = get_type_info(field_name)
                    
                    if offset is not None and dtype_str:
                        data = read_array(offset, dtype_str)
                        if ncomp > 1:
                            data = data.reshape(-1, ncomp)
                            fields[field_name] = data
                            fields[f"{field_name}_Magnitude"] = np.linalg.norm(data, axis=1)
                        else:
                            fields[field_name] = data
            
            self.solution_data = {
                'points': pts[:, :2],
                'fields': fields,
            }
            
            print(f"  Loaded VTU: {len(pts)} nodes, {len(triangles)} triangles")
            print(f"  Fields: {list(fields.keys())}")
            return True
            
        except Exception as e:
            print(f"VTU parse error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_available_fields(self) -> List[str]:
        """Get list of available field names for visualization."""
        if not self.solution_data:
            self.load_solution()
            
        fields = list(self.solution_data.get("fields", {}).keys())
        
        # Add commonly expected fields if not present
        default_fields = ["Pressure", "Velocity", "Temperature", "Density", "Mach"]
        for field in default_fields:
            if field not in fields:
                # Check for variations
                for existing in fields:
                    if field.lower() in existing.lower():
                        break
                else:
                    fields.append(field)
        
        return sorted(set(fields))

    def get_field_data(self, field_name: str) -> Optional[np.ndarray]:
        """Get field data array for visualization.
        
        Args:
            field_name: Name of field to retrieve
            
        Returns:
            Numpy array of field values at mesh nodes, or None
        """
        if not self.solution_data:
            self.load_solution()
            
        fields = self.solution_data.get("fields", {})
        
        # Direct match
        if field_name in fields:
            return fields[field_name]
            
        # Case-insensitive match
        field_lower = field_name.lower()
        for key, value in fields.items():
            if key.lower() == field_lower:
                return value
                
        # Partial match
        for key, value in fields.items():
            if field_lower in key.lower():
                return value
                
        # Handle velocity magnitude
        if field_name.lower() in ("velocity", "velocity_magnitude", "|velocity|"):
            vel_x = None
            vel_y = None
            for key, value in fields.items():
                if "velocity" in key.lower():
                    if "x" in key.lower():
                        vel_x = value
                    elif "y" in key.lower():
                        vel_y = value
            if vel_x is not None and vel_y is not None:
                return np.sqrt(vel_x**2 + vel_y**2)
                
        return None

    def get_points_2d(self) -> Optional[np.ndarray]:
        """Get 2D point coordinates for visualization.
        
        Returns:
            Nx2 numpy array of point coordinates
        """
        # Try solution data first (has point coordinates)
        if self.solution_data:
            points = self.solution_data.get("points")
            if points is not None and len(points) > 0:
                if points.shape[1] >= 2:
                    return points[:, :2]
                    
        # Fall back to mesh data
        if self.mesh_data:
            nodes = self.mesh_data.get("nodes", [])
            if nodes:
                arr = np.array(nodes)
                if arr.shape[1] >= 2:
                    return arr[:, :2]
                    
        return None

    def get_triangulation(self):
        """Get matplotlib triangulation for plotting.
        
        Returns:
            matplotlib.tri.Triangulation object or None
        """
        import matplotlib.tri as mtri
        
        points_2d = self.get_points_2d()
        if points_2d is None or len(points_2d) < 3:
            return None
            
        if self.mesh_data:
            elements = self.mesh_data.get("elements", [])
            
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
                    return mtri.Triangulation(
                        points_2d[:, 0], 
                        points_2d[:, 1], 
                        triangles=triangles
                    )
                except Exception:
                    pass
                    
        # Fall back to Delaunay triangulation
        try:
            return mtri.Triangulation(points_2d[:, 0], points_2d[:, 1])
        except Exception:
            return None

    def get_mesh_info(self) -> Dict[str, Any]:
        """Get mesh statistics.
        
        Returns:
            Dictionary with mesh information
        """
        info = {
            "num_points": 0,
            "num_elements": 0,
            "dimension": 2
        }
        
        if self.mesh_data:
            info["num_points"] = len(self.mesh_data.get("nodes", []))
            info["num_elements"] = len(self.mesh_data.get("elements", []))
            info["dimension"] = self.mesh_data.get("ndim", 2)
        elif self.solution_data:
            points = self.solution_data.get("points")
            if points is not None:
                info["num_points"] = len(points)
                
        return info

    def load_history(self) -> Dict[str, np.ndarray]:
        """Load convergence history data.

        Returns:
            Dictionary of convergence data arrays
        """
        # Try common history file names
        history_files = ["history.csv", "history.dat", "convergence.csv"]

        for name in history_files:
            path = self.case_dir / name
            if path.exists():
                self.history_data = parse_history_csv(str(path))
                return self.history_data

        return {}

    def get_iterations(self) -> Optional[np.ndarray]:
        """Get iteration numbers from history."""
        if not self.history_data:
            self.load_history()

        # Common column names for iterations
        for key in ["Iteration", "Inner_Iter", "Outer_Iter", "Time_Iter"]:
            if key in self.history_data:
                return self.history_data[key]

        return None

    def get_residual(self, field: str = "rms[Rho]") -> Optional[np.ndarray]:
        """Get residual data for a field.

        Args:
            field: Field name (e.g., 'rms[Rho]', 'rms[RhoU]', 'rms[RhoE]')
        """
        if not self.history_data:
            self.load_history()

        # Try exact match first
        if field in self.history_data:
            return self.history_data[field]

        # Try case-insensitive match
        field_lower = field.lower()
        for key in self.history_data:
            if key.lower() == field_lower:
                return self.history_data[key]

        return None

    def get_available_residuals(self) -> List[str]:
        """Get list of available residual fields."""
        if not self.history_data:
            self.load_history()

        return [k for k in self.history_data.keys() if "rms" in k.lower() or "res" in k.lower()]

    def get_force_coefficients(self) -> Dict[str, np.ndarray]:
        """Get aerodynamic force coefficients from history.

        Returns:
            Dictionary with Cd, Cl, Cm etc.
        """
        if not self.history_data:
            self.load_history()

        coeffs = {}
        coeff_patterns = ["CD", "CL", "CSF", "CMx", "CMy", "CMz", "CFx", "CFy", "CFz",
                          "Cd", "Cl", "Drag", "Lift", "Moment"]

        for key in self.history_data:
            for pattern in coeff_patterns:
                if pattern.lower() in key.lower():
                    coeffs[key] = self.history_data[key]
                    break

        return coeffs

    def get_flow_properties(self) -> Dict[str, float]:
        """Get final flow properties from last iteration."""
        if not self.history_data:
            self.load_history()

        props = {}
        for key, values in self.history_data.items():
            if len(values) > 0:
                props[key] = float(values[-1])

        return props

    def load_solution_file(self, filename: str = None) -> Dict[str, Any]:
        """Load solution data from CSV or VTU file.

        Args:
            filename: Solution filename. If None, searches for common names.
        """
        if filename:
            path = self.case_dir / filename
            if path.exists():
                if filename.endswith(".csv"):
                    self.solution_data = parse_surface_csv(str(path))
                    return self.solution_data
                # TODO: Add VTU parser if needed

        # Search for solution files
        for pattern in ["flow*.csv", "surface*.csv", "restart*.csv"]:
            files = list(self.case_dir.glob(pattern))
            if files:
                self.solution_data = parse_surface_csv(str(files[0]))
                return self.solution_data

        return {}

    def get_output_files(self) -> Dict[str, List[str]]:
        """Get categorized list of output files.

        Returns:
            Dictionary with categories: 'restart', 'surface', 'volume', 'history'
        """
        files = {
            "restart": [],
            "surface": [],
            "volume": [],
            "history": [],
            "log": [],
        }

        for f in os.listdir(self.case_dir):
            f_lower = f.lower()
            if "restart" in f_lower:
                files["restart"].append(f)
            elif "surface" in f_lower:
                files["surface"].append(f)
            elif "flow" in f_lower or "volume" in f_lower:
                files["volume"].append(f)
            elif "history" in f_lower or "convergence" in f_lower:
                files["history"].append(f)
            elif f.startswith("log."):
                files["log"].append(f)

        return files

    def get_time_steps(self) -> List[str]:
        """Get list of available time/iteration steps (for unsteady).

        For SU2, this typically means restart files with iteration numbers.
        """
        restart_pattern = re.compile(r"restart_(\d+)\.dat")
        steps = []

        for f in os.listdir(self.case_dir):
            match = restart_pattern.match(f)
            if match:
                steps.append(match.group(1))

        return sorted(steps, key=int)

    def compute_mass_flow(self, boundary: str = "inlet") -> Optional[float]:
        """Compute mass flow rate through a boundary.

        Note: Requires surface output data with density and velocity.
        """
        if not self.solution_data:
            self.load_solution_file()

        fields = self.solution_data.get("fields", {})

        # Look for density and velocity fields
        rho = None
        u = None

        for key in fields:
            if "density" in key.lower() or key.lower() == "rho":
                rho = fields[key]
            elif "velocity" in key.lower() and "x" in key.lower():
                u = fields[key]

        if rho is not None and u is not None:
            # Simplified: just integrate rho * u (assumes unit area)
            return float(np.sum(rho * u))

        return None

    def generate_report(self, output_file: str = None) -> str:
        """Generate text summary report of case results.

        Args:
            output_file: Optional file path to save report

        Returns:
            Report text
        """
        lines = ["=" * 60]
        lines.append("SU2 CASE ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"Case Directory: {self.case_dir}")
        lines.append("")

        # Config summary
        if not self.config:
            self.load_config()

        lines.append("CONFIGURATION:")
        lines.append("-" * 40)
        for key in ["SOLVER", "MATH_PROBLEM", "MACH_NUMBER", "REYNOLDS_NUMBER",
                    "FREESTREAM_PRESSURE", "FREESTREAM_TEMPERATURE", "KIND_TURB_MODEL"]:
            if key in self.config:
                lines.append(f"  {key}: {self.config[key]}")
        lines.append("")

        # Convergence summary
        if not self.history_data:
            self.load_history()

        if self.history_data:
            lines.append("CONVERGENCE:")
            lines.append("-" * 40)
            iters = self.get_iterations()
            if iters is not None:
                lines.append(f"  Total iterations: {int(iters[-1])}")

            residuals = self.get_available_residuals()
            for res in residuals[:5]:  # First 5 residuals
                values = self.history_data[res]
                if len(values) > 0:
                    lines.append(f"  {res}: {values[-1]:.6e}")
            lines.append("")

            # Force coefficients
            coeffs = self.get_force_coefficients()
            if coeffs:
                lines.append("FORCE COEFFICIENTS (final):")
                lines.append("-" * 40)
                for name, values in coeffs.items():
                    if len(values) > 0:
                        lines.append(f"  {name}: {values[-1]:.6f}")
                lines.append("")

        lines.append("=" * 60)

        report = "\n".join(lines)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report)

        return report


def analyze_case(case_dir: str, output_dir: str = None) -> None:
    """Analyze an SU2 case and generate output plots.

    Args:
        case_dir: Path to SU2 case directory
        output_dir: Output directory for plots (default: case_dir/postProcessing)
    """
    import matplotlib.pyplot as plt

    case = SU2Case(case_dir)
    case.load_config()
    case.load_history()

    if output_dir is None:
        output_dir = os.path.join(case_dir, "postProcessing")
    os.makedirs(output_dir, exist_ok=True)

    # Convergence plot
    iters = case.get_iterations()
    residuals = case.get_available_residuals()

    if iters is not None and residuals:
        plt.figure(figsize=(10, 6))
        for res in residuals[:6]:
            values = case.history_data[res]
            if len(values) == len(iters):
                plt.semilogy(iters, np.abs(values), label=res)

        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.title("Convergence History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "convergence.png"), dpi=150)
        plt.close()

    # Force coefficients plot
    coeffs = case.get_force_coefficients()
    if iters is not None and coeffs:
        plt.figure(figsize=(10, 6))
        for name, values in coeffs.items():
            if len(values) == len(iters):
                plt.plot(iters, values, label=name)

        plt.xlabel("Iteration")
        plt.ylabel("Coefficient")
        plt.title("Force Coefficients History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "coefficients.png"), dpi=150)
        plt.close()

    # Generate report
    report = case.generate_report(os.path.join(output_dir, "report.txt"))
    print(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SU2 Case Analyzer")
    parser.add_argument("--case", default=".", help="Path to case directory")
    parser.add_argument("--out", help="Output directory")

    args = parser.parse_args()
    analyze_case(args.case, args.out)
