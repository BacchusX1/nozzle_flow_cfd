#!/usr/bin/env python3
"""
OpenFOAM Case Visualization Script
Visualizes the mesh and latest solution fields from an OpenFOAM case.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata

def parse_openfoam_field(filepath):
    """
    Parse an OpenFOAM field file and extract the internal field data.
    
    Args:
        filepath: Path to the OpenFOAM field file
        
    Returns:
        numpy array of field values
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the internalField section
    start_idx = None
    field_type = None
    for i, line in enumerate(lines):
        if 'internalField' in line:
            if 'List<vector>' in line:
                field_type = 'vector'
            elif 'List<scalar>' in line:
                field_type = 'scalar'
            start_idx = i
            break
    
    if start_idx is None:
        raise ValueError(f"Could not find internalField in {filepath}")
    
    # Find the number of values
    n_values = None
    for i in range(start_idx, min(start_idx + 5, len(lines))):
        line = lines[i].strip()
        if line.isdigit():
            n_values = int(line)
            start_idx = i + 1
            break
    
    if n_values is None:
        raise ValueError(f"Could not determine field size in {filepath}")
    
    # Find opening parenthesis
    while start_idx < len(lines) and '(' not in lines[start_idx]:
        start_idx += 1
    start_idx += 1
    
    # Parse the data
    data = []
    current_idx = start_idx
    
    if field_type == 'scalar':
        while len(data) < n_values and current_idx < len(lines):
            line = lines[current_idx].strip()
            if line and not line.startswith(')') and not line.startswith('//'):
                try:
                    data.append(float(line))
                except ValueError:
                    pass
            current_idx += 1
        return np.array(data)
    
    elif field_type == 'vector':
        while len(data) < n_values and current_idx < len(lines):
            line = lines[current_idx].strip()
            if line and line.startswith('(') and not line.startswith('//'):
                # Extract vector components
                line = line.strip('()')
                try:
                    components = [float(x) for x in line.split()]
                    if len(components) == 3:
                        data.append(components)
                except ValueError:
                    pass
            current_idx += 1
        return np.array(data)
    
    return np.array(data)


def parse_openfoam_mesh(case_dir):
    """
    Parse OpenFOAM mesh from polyMesh directory to get cell centers.
    
    Args:
        case_dir: Path to the OpenFOAM case directory
        
    Returns:
        Dictionary containing points and cell centers
    """
    mesh_dir = os.path.join(case_dir, 'constant', 'polyMesh')
    
    # Parse points
    points_file = os.path.join(mesh_dir, 'points')
    with open(points_file, 'r') as f:
        lines = f.readlines()
    
    # Find number of points
    n_points = None
    start_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit() and n_points is None:
            n_points = int(line)
            start_idx = i + 1
            break
    
    # Find opening parenthesis
    while start_idx < len(lines) and '(' not in lines[start_idx]:
        start_idx += 1
    start_idx += 1
    
    # Parse points
    points = []
    current_idx = start_idx
    while len(points) < n_points and current_idx < len(lines):
        line = lines[current_idx].strip()
        if line and line.startswith('('):
            line = line.strip('()')
            try:
                coords = [float(x) for x in line.split()]
                if len(coords) == 3:
                    points.append(coords)
            except ValueError:
                pass
        current_idx += 1
    
    points = np.array(points)
    
    # Try to parse cell centers from C file (if paraFoam was run)
    # Otherwise we'll use a different approach
    cell_centers = None
    
    return {
        'points': points,
        'cell_centers': cell_centers,
        'n_points': len(points)
    }


def get_latest_time_directory(case_dir):
    """
    Find the latest time directory in the OpenFOAM case.
    
    Args:
        case_dir: Path to the OpenFOAM case directory
        
    Returns:
        Path to the latest time directory
    """
    time_dirs = []
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path):
            try:
                time_val = float(item)
                time_dirs.append((time_val, item_path))
            except ValueError:
                continue
    
    if not time_dirs:
        raise ValueError("No time directories found in case")
    
    time_dirs.sort(key=lambda x: x[0])
    return time_dirs[-1][1], time_dirs[-1][0]


def compute_cell_centers(case_dir, n_cells):
    """
    Compute actual cell centers from OpenFOAM mesh by parsing owner, faces, and points.
    This properly reconstructs cell geometry.
    """
    mesh_dir = os.path.join(case_dir, 'constant', 'polyMesh')
    
    # Parse points
    points_file = os.path.join(mesh_dir, 'points')
    with open(points_file, 'r') as f:
        lines = f.readlines()
    
    # Find and parse points
    n_points = None
    start_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit() and n_points is None:
            n_points = int(line)
            start_idx = i + 1
            break
    
    while start_idx < len(lines) and '(' not in lines[start_idx]:
        start_idx += 1
    start_idx += 1
    
    points = []
    current_idx = start_idx
    while len(points) < n_points and current_idx < len(lines):
        line = lines[current_idx].strip()
        if line and line.startswith('('):
            line = line.strip('()')
            try:
                coords = [float(x) for x in line.split()]
                if len(coords) == 3:
                    points.append(coords)
            except ValueError:
                pass
        current_idx += 1
    points = np.array(points)
    
    # Parse faces
    faces_file = os.path.join(mesh_dir, 'faces')
    with open(faces_file, 'r') as f:
        lines = f.readlines()
    
    n_faces = None
    start_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit() and n_faces is None:
            n_faces = int(line)
            start_idx = i + 1
            break
    
    while start_idx < len(lines) and '(' not in lines[start_idx]:
        start_idx += 1
    start_idx += 1
    
    faces = []
    current_idx = start_idx
    while len(faces) < n_faces and current_idx < len(lines):
        line = lines[current_idx].strip()
        if line and '(' in line:
            # Extract face vertex indices
            line = line.split('(')[1].split(')')[0]
            try:
                vertex_indices = [int(x) for x in line.split()]
                faces.append(vertex_indices)
            except ValueError:
                pass
        current_idx += 1
    
    # Parse owner
    owner_file = os.path.join(mesh_dir, 'owner')
    with open(owner_file, 'r') as f:
        lines = f.readlines()
    
    n_owner = None
    start_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit() and n_owner is None:
            n_owner = int(line)
            start_idx = i + 1
            break
    
    while start_idx < len(lines) and '(' not in lines[start_idx]:
        start_idx += 1
    start_idx += 1
    
    owner = []
    current_idx = start_idx
    while len(owner) < n_owner and current_idx < len(lines):
        line = lines[current_idx].strip()
        if line and not line.startswith(')') and not line.startswith('//'):
            try:
                owner.append(int(line))
            except ValueError:
                pass
        current_idx += 1
    owner = np.array(owner)
    
    # Compute cell centers by averaging face centers
    print(f"Computing cell centers from {len(faces)} faces and {len(owner)} owners...")
    
    # Initialize cell centers array
    cell_centers = np.zeros((n_cells, 3))
    cell_face_count = np.zeros(n_cells)
    
    # For each face, compute its center and add to the owner cell
    for face_idx, face_vertices in enumerate(faces[:len(owner)]):
        if face_idx < len(owner):
            cell_id = owner[face_idx]
            if cell_id < n_cells and len(face_vertices) > 0:
                # Compute face center
                face_coords = points[face_vertices]
                face_center = face_coords.mean(axis=0)
                
                # Add to cell center accumulator
                cell_centers[cell_id] += face_center
                cell_face_count[cell_id] += 1
    
    # Average to get actual cell centers
    for i in range(n_cells):
        if cell_face_count[i] > 0:
            cell_centers[i] /= cell_face_count[i]
    
    print(f"Computed {n_cells} cell centers")
    
    return cell_centers


def get_boundary_lines(case_dir, thin_dim=2):
    """
    Extract actual boundary faces from OpenFOAM mesh for plotting.
    Returns boundary line segments.
    """
    mesh_dir = os.path.join(case_dir, 'constant', 'polyMesh')
    
    # Parse points
    points_file = os.path.join(mesh_dir, 'points')
    with open(points_file, 'r') as f:
        lines = f.readlines()
    
    n_points = None
    start_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit() and n_points is None:
            n_points = int(line)
            start_idx = i + 1
            break
    
    while start_idx < len(lines) and '(' not in lines[start_idx]:
        start_idx += 1
    start_idx += 1
    
    points = []
    current_idx = start_idx
    while len(points) < n_points and current_idx < len(lines):
        line = lines[current_idx].strip()
        if line and line.startswith('('):
            line = line.strip('()')
            try:
                coords = [float(x) for x in line.split()]
                if len(coords) == 3:
                    points.append(coords)
            except ValueError:
                pass
        current_idx += 1
    points = np.array(points)
    
    # Parse faces
    faces_file = os.path.join(mesh_dir, 'faces')
    with open(faces_file, 'r') as f:
        lines = f.readlines()
    
    n_faces = None
    start_idx = None
    for i, line in enumerate(lines):
        line = line.strip()
        if line.isdigit() and n_faces is None:
            n_faces = int(line)
            start_idx = i + 1
            break
    
    while start_idx < len(lines) and '(' not in lines[start_idx]:
        start_idx += 1
    start_idx += 1
    
    faces = []
    current_idx = start_idx
    while len(faces) < n_faces and current_idx < len(lines):
        line = lines[current_idx].strip()
        if line and '(' in line:
            line = line.split('(')[1].split(')')[0]
            try:
                vertex_indices = [int(x) for x in line.split()]
                faces.append(vertex_indices)
            except ValueError:
                pass
        current_idx += 1
    
    # Parse boundary file
    boundary_file = os.path.join(mesh_dir, 'boundary')
    with open(boundary_file, 'r') as f:
        lines = f.readlines()
    
    # Extract boundary patch information
    boundary_patches = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for patch definitions (not comments or parentheses)
        if line and not line.startswith('//') and not line.startswith('/*'):
            # Check if next lines contain 'type' and 'nFaces'
            if i + 5 < len(lines):
                next_lines = ''.join(lines[i:i+10])
                if 'nFaces' in next_lines and 'startFace' in next_lines:
                    # Extract patch info
                    patch_name = line.strip()
                    for j in range(i, min(i+10, len(lines))):
                        if 'nFaces' in lines[j]:
                            nFaces = int(lines[j].split()[1].strip(';'))
                        if 'startFace' in lines[j]:
                            startFace = int(lines[j].split()[1].strip(';'))
                        if 'type' in lines[j]:
                            patch_type = lines[j].split()[1].strip(';')
                    
                    # Only include wall patches (not empty or processor)
                    if 'empty' not in patch_type and 'processor' not in patch_type:
                        boundary_patches.append({
                            'name': patch_name,
                            'type': patch_type,
                            'nFaces': nFaces,
                            'startFace': startFace
                        })
        i += 1
    
    print(f"Found {len(boundary_patches)} boundary patches:")
    for patch in boundary_patches:
        print(f"  {patch['name']}: {patch['nFaces']} faces starting at {patch['startFace']}")
    
    # Extract boundary face vertices
    boundary_lines = []
    for patch in boundary_patches:
        start = patch['startFace']
        end = start + patch['nFaces']
        for face_idx in range(start, min(end, len(faces))):
            face = faces[face_idx]
            if len(face) >= 2:
                # For each edge in the face, add as a line segment
                for i in range(len(face)):
                    p1_idx = face[i]
                    p2_idx = face[(i + 1) % len(face)]
                    if p1_idx < len(points) and p2_idx < len(points):
                        boundary_lines.append((points[p1_idx], points[p2_idx]))
    
    return boundary_lines


def visualize_case(case_dir):
    """
    Visualize the OpenFOAM mesh with velocity field from last time step.
    
    Args:
        case_dir: Path to the OpenFOAM case directory
    """
    print(f"Analyzing OpenFOAM case: {case_dir}")
    print("=" * 60)
    
    # Get latest time directory
    latest_time_dir, time_value = get_latest_time_directory(case_dir)
    print(f"Latest time: {time_value}")
    
    # Parse velocity field first to know how many cells we have
    print("\nParsing velocity field...")
    U_path = os.path.join(latest_time_dir, 'U')
    if not os.path.exists(U_path):
        print(f"Error: Velocity field not found at {U_path}")
        return
    
    U_data = parse_openfoam_field(U_path)
    print(f"Loaded velocity field: shape {U_data.shape}")
    
    # Calculate velocity magnitude
    if len(U_data.shape) == 2 and U_data.shape[1] == 3:
        U_mag = np.linalg.norm(U_data, axis=1)
    else:
        print("Error: Unexpected velocity field format")
        return
    
    n_cells = len(U_mag)
    print(f"Number of cells: {n_cells}")
    
    # Compute actual cell centers
    print("\nComputing cell centers...")
    cell_centers = compute_cell_centers(case_dir, n_cells)
    print(f"Cell centers shape: {cell_centers.shape}")
    
    # Analyze mesh dimensions to detect 2D case
    x_range = cell_centers[:, 0].max() - cell_centers[:, 0].min()
    y_range = cell_centers[:, 1].max() - cell_centers[:, 1].min()
    z_range = cell_centers[:, 2].max() - cell_centers[:, 2].min()
    
    print(f"\nMesh dimensions:")
    print(f"  X range: {x_range:.6f} m")
    print(f"  Y range: {y_range:.6f} m")
    print(f"  Z range: {z_range:.6f} m")
    
    # Detect which dimension is thin (2D mesh)
    ranges = [x_range, y_range, z_range]
    thin_dim = np.argmin(ranges)
    dim_names = ['X', 'Y', 'Z']
    
    is_2d = ranges[thin_dim] < 0.01 * max(ranges)
    
    if is_2d:
        print(f"Detected 2D mesh (thin in {dim_names[thin_dim]} direction)")
    else:
        print("Detected 3D mesh")
    
    # Create visualization
    print("\nCreating visualization...")
    
    from scipy.interpolate import griddata
    
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    
    # Determine which coordinates to use for plotting
    if is_2d:
        if thin_dim == 0:  # X is thin, plot Y-Z
            x_data = cell_centers[:, 1]
            y_data = cell_centers[:, 2]
            xlabel = 'Y [m]'
            ylabel = 'Z [m]'
        elif thin_dim == 1:  # Y is thin, plot X-Z
            x_data = cell_centers[:, 0]
            y_data = cell_centers[:, 2]
            xlabel = 'X [m]'
            ylabel = 'Z [m]'
        else:  # Z is thin, plot X-Y
            x_data = cell_centers[:, 0]
            y_data = cell_centers[:, 1]
            xlabel = 'X [m]'
            ylabel = 'Y [m]'
    else:
        # For 3D mesh, show X-Y projection
        x_data = cell_centers[:, 0]
        y_data = cell_centers[:, 1]
        xlabel = 'X [m]'
        ylabel = 'Y [m]'
    
    # Create simple scatter plot with color coding
    print("Creating scatter plot...")
    
    # Add boundary lines
    print("Extracting boundary patches...")
    boundary_lines = get_boundary_lines(case_dir, thin_dim)
    
    # Create scatter plot of velocity magnitude at cell centers
    scatter = ax.scatter(x_data, y_data, c=U_mag, s=2, cmap='jet', alpha=0.9)
    
    # Plot boundary lines on top
    print(f"Plotting {len(boundary_lines)} boundary segments...")
    for p1, p2 in boundary_lines:
        if is_2d:
            if thin_dim == 0:  # Y-Z plot
                ax.plot([p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=1.0, alpha=0.8)
            elif thin_dim == 1:  # X-Z plot
                ax.plot([p1[0], p2[0]], [p1[2], p2[2]], 'k-', linewidth=1.0, alpha=0.8)
            else:  # X-Y plot
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.0, alpha=0.8)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.0, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Velocity Magnitude [m/s]')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Velocity Magnitude at t = {time_value} s\n{n_cells} cells', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(case_dir, f'velocity_field_t{time_value}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("SOLUTION SUMMARY")
    print("=" * 60)
    print(f"Time: {time_value} s")
    print(f"Number of cells: {n_cells}")
    print(f"\nVelocity magnitude:")
    print(f"  Min: {U_mag.min():.4f} m/s")
    print(f"  Max: {U_mag.max():.4f} m/s")
    print(f"  Mean: {U_mag.mean():.4f} m/s")


def main():
    """Main entry point."""
    # Get case directory
    if len(sys.argv) > 1:
        case_dir = sys.argv[1]
    else:
        # Use current directory or parent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(current_dir, 'constant', 'polyMesh')):
            case_dir = current_dir
        else:
            case_dir = current_dir
    
    if not os.path.exists(os.path.join(case_dir, 'constant', 'polyMesh')):
        print(f"Error: No polyMesh found in {case_dir}")
        print("Usage: python check_case.py [case_directory]")
        sys.exit(1)
    
    try:
        visualize_case(case_dir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
