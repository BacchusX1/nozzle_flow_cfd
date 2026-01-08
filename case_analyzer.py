"""
OpenFOAM Case Analyzer for 2D Nozzle Flows.
Implements mesh visualization, field plotting, and flow analysis.
"""

import os
import sys
import re
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from collections import defaultdict
from pathlib import Path

# ==============================================================================
# OpenFOAM Parsers (Robust Minimal Implementation)
# ==============================================================================

def remove_comments(text):
    """Remove C++ style comments from OpenFOAM files."""
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text

def parse_foam_header(content):
    """Parse the FoamFile header."""
    header = {}
    match = re.search(r'FoamFile\s*\{(.*?)\}', content, re.DOTALL)
    if match:
        body = match.group(1)
        for line in body.split(';'):
            parts = line.strip().split()
            if len(parts) >= 2:
                header[parts[0]] = parts[1].strip('"')
    return header

def read_foam_file(filepath):
    """
    Read an OpenFOAM file and return raw content, removing header and comments.
    """
    if not os.path.exists(filepath):
        return None, ""
        
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    clean_content = remove_comments(content)
    header = parse_foam_header(clean_content)
    
    if '}' in clean_content:
        _, body = clean_content.split('}', 1)
        lines = body.splitlines()
        start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('//') or not line.strip():
                continue
            start = i
            break
        body = '\n'.join(lines[start:])
    else:
        body = clean_content

    return header, body

def parse_vector_data(text, n_items=None):
    """Parse a list of vectors e.g. (0 0 0) (1 2 3) ..."""
    if '(' in text:
        start = text.find('(')
        end = text.rfind(')')
        inner = text[start+1:end]
        inner = inner.replace('(', ' ').replace(')', ' ')
        try:
            data = np.fromstring(inner, sep=' ')
            return data.reshape(-1, 3)
        except ValueError:
            return np.array([]) 
    return np.array([])

def parse_scalar_data(text, n_items=None):
    """Parse scalar list."""
    if '(' in text:
        start = text.find('(')
        end = text.rfind(')')
        inner = text[start+1:end]
        try:
            data = np.fromstring(inner, sep=' ')
            return data
        except ValueError:
            return np.array([])
    return np.array([])

def parse_faces_list(text):
    """Parse face definitions."""
    faces = []
    # Optimization: Iteratively find pattern strictly
    pattern = re.compile(r'(\d+)\s*\(([\d\s]+)\)')
    matches = pattern.finditer(text)
    for m in matches:
        indices = np.fromstring(m.group(2), sep=' ', dtype=int)
        faces.append(indices)
    return faces

def parse_boundary(text):
    """Parse polyMesh/boundary file."""
    boundaries = {}
    current_patch = None
    lines = text.splitlines()
    depth = 0
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.isdigit() or line.startswith('boundary') or line.startswith('//'): continue
            
        if '{' in line:
            depth += 1
            if depth == 1:
                parts = line.split('{')
                name = parts[0].strip()
                if name: 
                    current_patch = name
                    boundaries[current_patch] = {}
            continue
            
        if '}' in line:
            depth -= 1
            if depth == 0: current_patch = None
            continue
            
        if current_patch and depth == 1:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                val = parts[1].strip(';')
                boundaries[current_patch][key] = val
                
        if depth == 0 and not '}' in line and not '(' in line and not ')' in line:
            current_patch = line
            boundaries[current_patch] = {}

    return boundaries

def parse_field_data(text):
    """
    Parse internalField and boundaryField.
    Returns dictionary with 'internal' and 'boundary'.
    """
    data = {'internal': None, 'boundary': {}}
    
    # Internal Field
    if 'internalField' in text:
        uni_match = re.search(r'internalField\s+uniform\s+([^;]+);', text)
        if uni_match:
            val_str = uni_match.group(1).replace('(', '').replace(')', '')
            vals = [float(x) for x in val_str.split()]
            data['internal'] = {'type': 'uniform', 'value': vals[0] if len(vals)==1 else np.array(vals)}
        else:
            # Non-uniform
            # locate "internalField nonuniform List<...>"
            match = re.search(r'internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(', text)
            if match:
                n_items = int(match.group(1))
                list_start = match.end() - 1
                # Find matching parenthesis
                cnt = 0
                list_end = -1
                for i in range(list_start, len(text)):
                    if text[i] == '(': cnt += 1
                    elif text[i] == ')': cnt -= 1
                    if cnt == 0:
                        list_end = i
                        break
                
                if list_end != -1:
                    list_content = text[list_start:list_end+1]
                    # Check if vector or scalar
                    if list_content[1:].strip().startswith('('):
                        data['internal'] = {'type': 'nonuniform', 'value': parse_vector_data(list_content)}
                    else:
                        data['internal'] = {'type': 'nonuniform', 'value': parse_scalar_data(list_content)}

    return data

# ==============================================================================
# Mesh and Case Analysis Classes
# ==============================================================================

class OpenFOAMCase:
    def __init__(self, case_dir):
        self.case_dir = Path(case_dir).resolve()
        self.mesh_dir = self.case_dir / 'constant' / 'polyMesh'
        self.points = None
        self.faces = None
        self.owner = None
        self.neighbour = None
        self.boundary = None
        self.triangulation = None
        self.point_to_cells = None
        
    def load_mesh(self):
        print(f"Loading mesh from {self.mesh_dir}...")
        _, pts_txt = read_foam_file(self.mesh_dir / 'points')
        self.points = parse_vector_data(pts_txt)
        
        _, faces_txt = read_foam_file(self.mesh_dir / 'faces')
        self.faces = parse_faces_list(faces_txt)
        
        _, own_txt = read_foam_file(self.mesh_dir / 'owner')
        self.owner = parse_scalar_data(own_txt).astype(int)
        
        _, nei_txt = read_foam_file(self.mesh_dir / 'neighbour')
        self.neighbour = parse_scalar_data(nei_txt).astype(int)
        
        _, bnd_txt = read_foam_file(self.mesh_dir / 'boundary')
        self.boundary = parse_boundary(bnd_txt)
        
        self.n_cells = max(self.owner.max(), self.neighbour.max() if len(self.neighbour) > 0 else 0) + 1
        print(f"Mesh loaded: {len(self.points)} points, {len(self.faces)} faces, {self.n_cells} cells")
        
    def prepare_2d_surface(self):
        """Prepare 2D surface triangulation.

        IMPORTANT: For a converging-diverging nozzle, naive Delaunay triangulation
        fills the convex hull and creates triangles outside the physical domain
        (appearing like a bounding box overlay). To avoid that, we prefer using
        the actual topology of the 2D 'empty' patch (typically frontAndBack).
        """

        # Determine the "flat" dimension (2D extrusion thickness)
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        dims = maxs - mins
        flat_dim = int(np.argmin(dims))
        keep_dims = [0, 1, 2]
        keep_dims.remove(flat_dim)
        idx_x, idx_y = keep_dims

        # Pick an empty patch to represent the 2D surface connectivity
        empty_patches = [name for name, props in self.boundary.items() if props.get('type') == 'empty']
        patch_name = None
        if empty_patches:
            # Prefer common name if present
            patch_name = 'frontAndBack' if 'frontAndBack' in empty_patches else empty_patches[0]

        self.surface_point_ids = None
        self.points_2d = None
        self.unique_indices = None

        if patch_name and ('startFace' in self.boundary.get(patch_name, {})) and ('nFaces' in self.boundary.get(patch_name, {})):
            start_face = int(self.boundary[patch_name]['startFace'])
            n_faces = int(self.boundary[patch_name]['nFaces'])
            patch_faces = self.faces[start_face:start_face + n_faces]

            # frontAndBack usually contains BOTH planes (min and max along flat_dim).
            # Select one plane (the min plane) to avoid overlapping triangles.
            flat_min = mins[flat_dim]
            flat_max = maxs[flat_dim]
            tol = max(1e-12, 1e-9 * max(1.0, float(flat_max - flat_min)))

            faces_min_plane = []
            faces_max_plane = []
            for f in patch_faces:
                coords = self.points[f, flat_dim]
                mean_val = float(np.mean(coords))
                if abs(mean_val - flat_min) <= tol:
                    faces_min_plane.append(f)
                elif abs(mean_val - flat_max) <= tol:
                    faces_max_plane.append(f)

            surface_faces = faces_min_plane if faces_min_plane else faces_max_plane
            if not surface_faces:
                surface_faces = patch_faces

            # Build a compact point list and remap indices
            point_ids = np.unique(np.concatenate(surface_faces))
            local_index = {int(pid): i for i, pid in enumerate(point_ids)}
            points2d = self.points[point_ids][:, [idx_x, idx_y]]

            # Fan-triangulate each polygonal face (quads -> 2 triangles)
            triangles = []
            for f in surface_faces:
                if len(f) < 3:
                    continue
                i0 = local_index[int(f[0])]
                for k in range(1, len(f) - 1):
                    i1 = local_index[int(f[k])]
                    i2 = local_index[int(f[k + 1])]
                    triangles.append((i0, i1, i2))

            self.surface_point_ids = point_ids
            self.points_2d = points2d
            self.triangulation = mtri.Triangulation(points2d[:, 0], points2d[:, 1], np.asarray(triangles, dtype=int))
            print(
                f"Created surface triangulation from patch '{patch_name}': "
                f"{len(self.triangulation.triangles)} triangles, {len(point_ids)} nodes"
            )
            return

        # Fallback: Delaunay on unique projected points (may include outside-of-domain triangles)
        pts_2d_raw = self.points[:, [idx_x, idx_y]]
        self.points_2d, self.unique_indices = np.unique(pts_2d_raw, axis=0, return_index=True)
        self.triangulation = mtri.Triangulation(self.points_2d[:, 0], self.points_2d[:, 1])
        print(
            f"Created fallback Delaunay triangulation with {len(self.triangulation.triangles)} elements "
            f"(from {len(self.points_2d)} unique nodes)"
        )

    def get_time_dirs(self):
        dirs = []
        for d in os.listdir(self.case_dir):
            path = self.case_dir / d
            if d.startswith('.'): continue 
            if not path.is_dir(): continue
            if d in ['0', 'constant', 'system', 'postProcessing', 'processor0']: 
                # check if d is a float
                try: 
                    float(d)
                except:
                    continue
            
            try:
                val = float(d)
                dirs.append((val, d))
            except ValueError:
                continue
        return sorted(dirs, key=lambda x: x[0])

    def load_field(self, time_dir, field_name):
        fpath = self.case_dir / time_dir / field_name
        _, content = read_foam_file(fpath)
        data = parse_field_data(content)
        
        val_block = data.get('internal')
        if val_block:
            if val_block['type'] == 'uniform':
                return np.tile(val_block['value'], (self.n_cells, 1)) if isinstance(val_block['value'], np.ndarray) else np.full(self.n_cells, val_block['value'])
            return val_block['value']
        return None

    def interpolate_to_nodes(self, cell_values):
        """Map cell values to nodes."""
        if self.point_to_cells is None:
            print("Building connectivity map...")
            self.point_to_cells = defaultdict(list)
            
            # Map Point -> Face -> Cell
            points_to_faces = defaultdict(list)
            for f_idx, face in enumerate(self.faces):
                for p_idx in face:
                    points_to_faces[p_idx].append(f_idx)
            
            # Iterate faces to assign cells to points
            for p_idx, faces in points_to_faces.items():
                seen_cells = set()
                for f_idx in faces:
                    # Check owner
                    if f_idx < len(self.owner):
                        c = self.owner[f_idx]
                        if c not in seen_cells:
                            self.point_to_cells[p_idx].append(c)
                            seen_cells.add(c)
                    # Check neighbour
                    if f_idx < len(self.neighbour):
                        c = self.neighbour[f_idx]
                        if c not in seen_cells:
                            self.point_to_cells[p_idx].append(c)
                            seen_cells.add(c)

        n_points = len(self.points)
        is_vector = (len(cell_values.shape) > 1 and cell_values.shape[1] == 3)
        node_vals = np.zeros((n_points, 3) if is_vector else n_points)
        
        # Vectorize? Difficult with variable list lengths. 
        # Loop is OK for <100k points in Python for offline analysis.
        for p_idx in range(n_points):
            cells = self.point_to_cells.get(p_idx, [])
            if not cells: continue
            
            valid_cells = [c for c in cells if c < len(cell_values)]
            if valid_cells:
                vals = cell_values[valid_cells]
                node_vals[p_idx] = np.mean(vals, axis=0)
                
        return node_vals

def compute_flow_summary(case, time_dir, output_dir):
    """
    Compute mass/volumetric flow from phi.
    """
    fpath = case.case_dir / time_dir / 'phi'
    if not fpath.exists():
        return
        
    print("Computing flow rates from phi...")
    _, content = read_foam_file(fpath)
    
    # 1. Get Patch Sizes from Boundary File
    boundary_faces = {} 
    _, bcontent = read_foam_file(case.mesh_dir / 'boundary')
    n_internal_faces = len(case.neighbour)
    current_start = n_internal_faces
    
    matches = re.finditer(r'([a-zA-Z0-9_]+)\s*\{[^\}]*\}', bcontent)
    for m in matches:
        name = m.group(1)
        block = m.group(0)
        nf = int(re.search(r'nFaces\s+(\d+);', block).group(1))
        sf_match = re.search(r'startFace\s+(\d+);', block)
        sf = int(sf_match.group(1)) if sf_match else current_start
        
        boundary_faces[name] = (sf, nf)
        current_start = sf + nf
    
    # 2. Extract Boundary Field Block for phi
    flows = {}
    bf_match = re.search(r'boundaryField\s*\{', content)
    if bf_match:
        bf_content = content[bf_match.end():]
        # Robustly extract full block considering nesting
        depth = 1
        end_idx = 0
        for i, char in enumerate(bf_content):
            if char == '{': depth += 1
            elif char == '}': depth -= 1
            if depth == 0:
                end_idx = i
                break
        bf_content = bf_content[:end_idx]
        
        # 3. Find patch values
        for patch, (start, count) in boundary_faces.items():
            # Regex to find 'patchName { ... }'
            # Note: patchName can be complex, usually words
            p_match = re.search(fr'\b{patch}\b\s*\{{', bf_content)
            if p_match:
                # Extract this patch's block
                p_start = p_match.end()
                p_depth = 1
                p_end = 0
                for i in range(p_start, len(bf_content)):
                    if bf_content[i] == '{': p_depth += 1
                    elif bf_content[i] == '}': p_depth -= 1
                    if p_depth == 0:
                        p_end = i
                        break
                p_block = bf_content[p_start:p_end]
                
                # Check value
                if 'scalar' in p_block and 'nonuniform' in p_block:
                     # Parse list
                     match = re.search(r'(\d+)\s*\(', p_block)
                     if match:
                         list_start = match.end() - 1
                         cnt = 0
                         for k in range(list_start, len(p_block)):
                             if p_block[k] == '(': cnt+=1
                             elif p_block[k] == ')': cnt-=1
                             if cnt == 0:
                                 l_str = p_block[list_start:k+1]
                                 vals = parse_scalar_data(l_str)
                                 flows[patch] = np.sum(vals)
                                 break
                elif 'uniform' in p_block:
                    val_match = re.search(r'value\s+uniform\s+([^;]+);', p_block)
                    if val_match:
                        # For phi, uniform value usually means uniform flux? 
                        # Or uniform velocity? If phi is flux, it's m3/s.
                        # If uniform 0, sum is 0.
                        val = float(val_match.group(1))
                        flows[patch] = val * count 

    # Save
    with open(os.path.join(output_dir, 'nozzle_flow.json'), 'w') as f:
        json.dump(flows, f, indent=2)
    
    line = ["FLOW SUMMARY (Volumetric if incompressible)", "======================================="]
    for p, f in flows.items():
        line.append(f"Patch {p}: {f:.6e}")
    
    with open(os.path.join(output_dir, 'nozzle_flow.txt'), 'w') as f:
        f.write('\n'.join(line))
    print("Flow summary saved.")

def analyze_case(case_dir, output_dir=None, time=None, fields=['U', 'p']):
    case = OpenFOAMCase(case_dir)
    case.load_mesh()
    case.prepare_2d_surface()
    
    if output_dir is None:
        output_dir = os.path.join(case_dir, 'postProcessing', 'caseAnalyzer')
    os.makedirs(output_dir, exist_ok=True)
    
    times = case.get_time_dirs()
    if not times:
        print("No time directories found!")
        return
        
    if time is None:
        selected_time_val, selected_time_dir = times[-1]
    else:
        selected_time_dir = str(time)
        selected_time_val = float(time)
        
    print(f"Analyzing time: {selected_time_dir}")
    
    # Mesh Plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    # With ~O(1e5) nodes and ~O(1e5-1e6) edges, the plot will inevitably look
    # "gray" when zoomed out. The goal here is to make vertices/edges visible
    # when zooming in, while keeping the overall rendering performant.
    plt.triplot(
        case.triangulation,
        color='k',
        linewidth=0.15,
        alpha=0.35,
        rasterized=True,
    )
    # Overlay vertices
    if getattr(case, 'points_2d', None) is not None:
        ax.scatter(
            case.points_2d[:, 0],
            case.points_2d[:, 1],
            s=0.15,
            c='k',
            alpha=0.5,
            linewidths=0,
            rasterized=True,
        )
    plt.title('Computational Mesh')
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'mesh.png'), dpi=300)
    plt.close()
    
    # Fields
    for field_name in fields:
        print(f"Processing field: {field_name}")
        val = case.load_field(selected_time_dir, field_name)
        if val is None: 
            print(f"Field {field_name} not found.")
            continue
        
        node_vals = case.interpolate_to_nodes(val)

        # Reduce to plotted node set
        # - If we built a surface triangulation from the empty patch, use only those points.
        # - Otherwise, fall back to unique projected points (Delaunay fallback).
        if getattr(case, 'surface_point_ids', None) is not None:
            node_vals = node_vals[case.surface_point_ids]
        elif getattr(case, 'unique_indices', None) is not None:
            node_vals = node_vals[case.unique_indices]
        
        # Plot
        if len(node_vals.shape) == 1:
            plt.figure(figsize=(12, 6))
            plt.tripcolor(case.triangulation, node_vals, shading='gouraud', cmap='viridis')
            plt.colorbar(label=field_name)
            plt.title(f'{field_name} at t={selected_time_val}')
            plt.axis('equal')
            plt.savefig(os.path.join(output_dir, f'field_{field_name}_surface.png'), dpi=300)
            plt.close()
        else:
            mag = np.linalg.norm(node_vals, axis=1)
            plt.figure(figsize=(12, 6))
            plt.tripcolor(case.triangulation, mag, shading='gouraud', cmap='plasma')
            plt.colorbar(label=f'|{field_name}|')
            plt.title(f'{field_name} Magnitude at t={selected_time_val}')
            plt.axis('equal')
            plt.savefig(os.path.join(output_dir, f'field_{field_name}mag_surface.png'), dpi=300)
            plt.close()
            
            for i, comp in enumerate(['x', 'y', 'z']):
                plt.figure(figsize=(12, 6))
                plt.tripcolor(case.triangulation, node_vals[:, i], shading='gouraud', cmap='coolwarm')
                plt.colorbar(label=f'{field_name}{comp}')
                plt.title(f'{field_name}{comp} Component')
                plt.axis('equal')
                plt.savefig(os.path.join(output_dir, f'field_{field_name}{comp}_surface.png'), dpi=300)
                plt.close()

    compute_flow_summary(case, selected_time_dir, output_dir)
    
    # Settings Summary
    summary = {}
    cd_path = case.case_dir / 'system' / 'controlDict'
    if cd_path.exists():
        h, content = read_foam_file(cd_path)
        summary['controlDict'] = h
    
    # Parse log
    # Find any log file in case dir
    log_files = glob.glob(str(case.case_dir / 'log.*'))
    if not log_files:
        log_files = glob.glob(str(case.case_dir / 'case/log.*'))
        
    if log_files:
        summary['logs'] = log_files
        with open(log_files[0], 'r', errors='ignore') as f:
            # Read last lines
            lines = f.readlines()
            summary['log_tail'] = [l.strip() for l in lines[-10:]]

    with open(os.path.join(output_dir, 'settings_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save a readable report
    with open(os.path.join(output_dir, 'settings_summary.md'), 'w') as f:
        f.write("# Case Summary\n\n")
        f.write("## Control Dict\n")
        if 'controlDict' in summary:
            for k, v in summary['controlDict'].items():
                f.write(f"- **{k}**: {v}\n")
        f.write("\n## Logs\n")
        if 'log_tail' in summary:
            f.write("Last lines:\n```\n")
            f.write('\n'.join(summary['log_tail']))
            f.write("\n```\n")

    print(f"Analysis complete. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenFOAM Case Analyzer")
    parser.add_argument(
        "--case",
        default=".",
        help="Path to case directory (default: current directory)",
    )
    parser.add_argument("--out", help="Output directory")
    parser.add_argument("--time", help="Specific time directory")
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import matplotlib
        import numpy
    except ImportError:
        print("Error: numpy and matplotlib are required.")
        sys.exit(1)
        
    # If run without args, assume current working directory is the case.
    case_dir = os.path.abspath(args.case)
    required = [os.path.join(case_dir, "system"), os.path.join(case_dir, "constant")]
    if not all(os.path.isdir(p) for p in required):
        print(
            "Error: This does not look like an OpenFOAM case directory.\n"
            "Expected to find 'system/' and 'constant/' in: " + case_dir + "\n\n"
            "Run from inside your case directory, or pass --case /path/to/case"
        )
        sys.exit(2)

    analyze_case(case_dir, args.out, args.time)
