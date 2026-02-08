"""
SU2 Mesh Converter Module

Converts internal mesh data to SU2 native mesh format (.su2).
Handles 2D triangular and quadrilateral elements.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# SU2 element type codes
SU2_ELEMENT_TYPES = {
    "line": 3,        # 2-node line (for boundaries)
    "triangle": 5,    # 3-node triangle
    "quad": 9,        # 4-node quadrilateral
    "tetra": 10,      # 4-node tetrahedron
    "hexa": 12,       # 8-node hexahedron
    "prism": 13,      # 6-node prism
    "pyramid": 14,    # 5-node pyramid
}


class SU2MeshConverter:
    """Converts mesh data to SU2 format."""

    def __init__(self):
        self.nodes: List[Tuple[float, float]] = []
        self.elements: List[Tuple[int, ...]] = []
        self.element_type: str = "triangle"
        self.boundaries: Dict[str, List[Tuple[int, int]]] = {}
        self.ndim: int = 2

    def load_from_mesh_data(self, mesh_data: Dict[str, Any]) -> None:
        """Load mesh from internal mesh data dictionary.

        Args:
            mesh_data: Dictionary with 'nodes'/'vertices', 'elements', 'element_type',
                      and optionally 'boundaries'.
        """
        # Get nodes/vertices - avoid 'or' with numpy arrays
        nodes = mesh_data.get("nodes")
        if nodes is None:
            nodes = mesh_data.get("vertices", [])
        self.nodes = [(float(n[0]), float(n[1])) for n in nodes]

        # Get elements
        self.elements = [tuple(int(i) for i in elem) for elem in mesh_data.get("elements", [])]

        # Get element type
        self.element_type = mesh_data.get("element_type", "triangle")
        if self.element_type in ("quad", "quadrilateral"):
            self.element_type = "quad"
        elif self.element_type in ("tri", "triangle"):
            self.element_type = "triangle"

        # Determine dimensionality
        if self.nodes:
            sample = self.nodes[0]
            self.ndim = len(sample) if len(sample) <= 3 else 2

        # Get boundaries if provided (check both 'boundaries' and 'boundary_elements')
        raw_boundaries = mesh_data.get("boundaries")
        if raw_boundaries is None:
            raw_boundaries = mesh_data.get("boundary_elements", {})
        
        # Normalize boundary names for SU2 compatibility
        self.boundaries = {}
        for name, edges in raw_boundaries.items():
            # Convert lists to tuples if needed
            edge_tuples = [tuple(e) if isinstance(e, list) else e for e in edges]
            
            # Map internal names to SU2 standard names
            if name in ('wall_upper', 'wall_lower'):
                # Combine wall_upper and wall_lower into 'wall'
                if 'wall' not in self.boundaries:
                    self.boundaries['wall'] = []
                self.boundaries['wall'].extend(edge_tuples)
            elif name == 'centerline':
                # Map centerline to symmetry
                self.boundaries['symmetry'] = edge_tuples
            else:
                self.boundaries[name] = edge_tuples

    def detect_boundaries_from_geometry(self, geometry) -> None:
        """Detect boundary edges from geometry data.

        For a 2D nozzle, typically:
        - inlet: left boundary (x = x_min)
        - outlet: right boundary (x = x_max)
        - wall: top and bottom curved surfaces
        - symmetry: centerline (if applicable)
        """
        if not self.nodes or not self.elements:
            return

        nodes_arr = np.array(self.nodes)
        x_min, x_max = nodes_arr[:, 0].min(), nodes_arr[:, 0].max()
        y_min, y_max = nodes_arr[:, 1].min(), nodes_arr[:, 1].max()

        # Tolerance for boundary detection
        tol = (x_max - x_min) * 0.001

        # Find boundary edges (edges that appear only once)
        edge_count = {}
        for elem in self.elements:
            n_nodes = len(elem)
            for i in range(n_nodes):
                n1, n2 = elem[i], elem[(i + 1) % n_nodes]
                edge = (min(n1, n2), max(n1, n2))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        # Classify boundary edges
        inlet_edges = []
        outlet_edges = []
        wall_edges = []
        symmetry_edges = []

        for edge in boundary_edges:
            n1, n2 = edge
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2

            # Inlet: left side
            if abs(x1 - x_min) < tol and abs(x2 - x_min) < tol:
                inlet_edges.append(edge)
            # Outlet: right side
            elif abs(x1 - x_max) < tol and abs(x2 - x_max) < tol:
                outlet_edges.append(edge)
            # Symmetry: bottom (y ≈ 0 or y_min)
            elif abs(y1 - y_min) < tol and abs(y2 - y_min) < tol:
                if geometry and getattr(geometry, 'is_symmetric', False):
                    symmetry_edges.append(edge)
                else:
                    wall_edges.append(edge)
            # Wall: everything else on the boundary
            else:
                wall_edges.append(edge)

        self.boundaries = {
            "inlet": inlet_edges,
            "outlet": outlet_edges,
            "wall": wall_edges,
        }
        if symmetry_edges:
            self.boundaries["symmetry"] = symmetry_edges

    def write_su2_mesh(self, filepath: str) -> None:
        """Write mesh in SU2 native format.

        Args:
            filepath: Output file path (should end in .su2)
        
        Handles mixed element types (triangles + quads) from boundary layer meshes.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            # Dimension
            f.write(f"NDIME= {self.ndim}\n")

            # Elements
            n_elem = len(self.elements)
            f.write(f"NELEM= {n_elem}\n")

            for idx, elem in enumerate(self.elements):
                # Determine element type based on number of nodes
                n_nodes = len(elem)
                if n_nodes == 3:
                    elem_code = 5  # Triangle
                elif n_nodes == 4:
                    elem_code = 9  # Quad
                else:
                    # Fallback to stored element type
                    elem_code = SU2_ELEMENT_TYPES.get(self.element_type, 5)
                
                node_str = " ".join(str(n) for n in elem)
                f.write(f"{elem_code} {node_str} {idx}\n")

            # Nodes
            n_nodes = len(self.nodes)
            f.write(f"NPOIN= {n_nodes}\n")
            for idx, node in enumerate(self.nodes):
                if self.ndim == 2:
                    f.write(f"{node[0]:.15e} {node[1]:.15e} {idx}\n")
                else:
                    z = node[2] if len(node) > 2 else 0.0
                    f.write(f"{node[0]:.15e} {node[1]:.15e} {z:.15e} {idx}\n")

            # Markers (boundaries)
            n_markers = len(self.boundaries)
            f.write(f"NMARK= {n_markers}\n")

            for marker_name, edges in self.boundaries.items():
                f.write(f"MARKER_TAG= {marker_name}\n")
                f.write(f"MARKER_ELEMS= {len(edges)}\n")
                for edge in edges:
                    # Line element (code 3) for 2D boundaries
                    f.write(f"3 {edge[0]} {edge[1]}\n")

    def read_su2_mesh(self, filepath: str) -> Dict[str, Any]:
        """Read mesh from SU2 format.

        Args:
            filepath: Input file path

        Returns:
            Dictionary with mesh data
        """
        nodes = []
        elements = []
        boundaries = {}
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
                    parts = lines[i].strip().split()
                    if ndim == 2:
                        nodes.append((float(parts[0]), float(parts[1])))
                    else:
                        nodes.append((float(parts[0]), float(parts[1]), float(parts[2])))
                    i += 1

            elif line.startswith("NELEM="):
                n_elem = int(line.split("=")[1].strip())
                i += 1
                for _ in range(n_elem):
                    parts = lines[i].strip().split()
                    elem_type = int(parts[0])
                    # Determine number of nodes based on element type
                    if elem_type == 5:  # Triangle
                        elem = (int(parts[1]), int(parts[2]), int(parts[3]))
                    elif elem_type == 9:  # Quad
                        elem = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                    elif elem_type == 3:  # Line
                        elem = (int(parts[1]), int(parts[2]))
                    else:
                        # Generic: read until index
                        elem = tuple(int(p) for p in parts[1:-1])
                    elements.append(elem)
                    i += 1

            elif line.startswith("NMARK="):
                n_markers = int(line.split("=")[1].strip())
                i += 1
                for _ in range(n_markers):
                    # MARKER_TAG
                    tag_line = lines[i].strip()
                    if tag_line.startswith("MARKER_TAG="):
                        marker_name = tag_line.split("=")[1].strip()
                        i += 1
                        # MARKER_ELEMS
                        elem_line = lines[i].strip()
                        n_marker_elems = int(elem_line.split("=")[1].strip())
                        i += 1
                        marker_edges = []
                        for _ in range(n_marker_elems):
                            parts = lines[i].strip().split()
                            # Skip element type code, get node indices
                            edge = (int(parts[1]), int(parts[2]))
                            marker_edges.append(edge)
                            i += 1
                        boundaries[marker_name] = marker_edges
                    else:
                        i += 1
            else:
                i += 1

        # Determine element type from first element
        elem_type = "triangle"
        if elements:
            first_elem = elements[0]
            if len(first_elem) == 4:
                elem_type = "quad"
            elif len(first_elem) == 3:
                elem_type = "triangle"

        return {
            "nodes": nodes,
            "elements": elements,
            "element_type": elem_type,
            "boundaries": boundaries,
            "ndim": ndim,
        }

    def convert_from_gmsh(self, msh_filepath: str, su2_filepath: str) -> bool:
        """Convert Gmsh mesh to SU2 format.

        Args:
            msh_filepath: Input Gmsh .msh file
            su2_filepath: Output SU2 .su2 file

        Returns:
            True if conversion successful
        """
        try:
            # Try using gmsh Python API if available
            import gmsh
            gmsh.initialize()
            gmsh.open(msh_filepath)

            # Get nodes
            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            n_nodes = len(node_tags)

            # Create node mapping (gmsh uses 1-based, SU2 uses 0-based)
            node_map = {tag: idx for idx, tag in enumerate(node_tags)}

            # Reshape coordinates
            coords = np.array(coords).reshape(-1, 3)
            self.nodes = [(coords[i, 0], coords[i, 1]) for i in range(n_nodes)]

            # Get 2D elements (triangles and quads)
            elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=2)

            self.elements = []
            for et, nodes_list in zip(elem_types, elem_nodes):
                nodes_per_elem = gmsh.model.mesh.getElementProperties(et)[3]
                nodes_list = np.array(nodes_list).reshape(-1, nodes_per_elem)
                for elem_nodes in nodes_list:
                    elem = tuple(node_map[n] for n in elem_nodes)
                    self.elements.append(elem)
                    if nodes_per_elem == 3:
                        self.element_type = "triangle"
                    elif nodes_per_elem == 4:
                        self.element_type = "quad"

            # Get physical groups for boundaries
            phys_groups = gmsh.model.getPhysicalGroups(dim=1)
            self.boundaries = {}

            for dim, tag in phys_groups:
                name = gmsh.model.getPhysicalName(dim, tag)
                if not name:
                    name = f"boundary_{tag}"

                # Get entities in this physical group
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                edges = []
                for entity in entities:
                    _, _, entity_nodes = gmsh.model.mesh.getElements(dim, entity)
                    if entity_nodes:
                        entity_nodes = np.array(entity_nodes[0]).reshape(-1, 2)
                        for edge_nodes in entity_nodes:
                            edge = (node_map[edge_nodes[0]], node_map[edge_nodes[1]])
                            edges.append(edge)
                self.boundaries[name] = edges

            gmsh.finalize()

            # Write SU2 mesh
            self.write_su2_mesh(su2_filepath)
            return True

        except ImportError:
            # Fallback: simple MSH parser for basic Gmsh format
            return self._convert_from_gmsh_fallback(msh_filepath, su2_filepath)
        except Exception as e:
            print(f"Gmsh conversion error: {e}")
            return False

    def _convert_from_gmsh_fallback(self, msh_filepath: str, su2_filepath: str) -> bool:
        """Fallback Gmsh parser for simple mesh files."""
        try:
            with open(msh_filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Very basic MSH 2.x parser
            lines = content.split("\n")
            i = 0
            nodes = []
            elements = []

            while i < len(lines):
                line = lines[i].strip()

                if line == "$Nodes":
                    i += 1
                    n_nodes = int(lines[i].strip())
                    i += 1
                    for _ in range(n_nodes):
                        parts = lines[i].strip().split()
                        nodes.append((float(parts[1]), float(parts[2])))
                        i += 1

                elif line == "$Elements":
                    i += 1
                    n_elem = int(lines[i].strip())
                    i += 1
                    for _ in range(n_elem):
                        parts = lines[i].strip().split()
                        elem_type = int(parts[1])
                        n_tags = int(parts[2])
                        node_start = 3 + n_tags
                        # Type 2 = triangle, type 3 = quad
                        if elem_type == 2:
                            elem = tuple(int(parts[node_start + j]) - 1 for j in range(3))
                            elements.append(elem)
                            self.element_type = "triangle"
                        elif elem_type == 3:
                            elem = tuple(int(parts[node_start + j]) - 1 for j in range(4))
                            elements.append(elem)
                            self.element_type = "quad"
                        i += 1
                else:
                    i += 1

            self.nodes = nodes
            self.elements = elements
            self.write_su2_mesh(su2_filepath)
            return True

        except Exception as e:
            print(f"Fallback Gmsh conversion error: {e}")
            return False
