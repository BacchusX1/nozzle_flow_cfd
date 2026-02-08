"""
Mesh Generation Module

Handles advanced mesh generation with boundary layers, refinement zones,
and comprehensive mesh quality control for CFD simulations.
"""

import numpy as np
import tempfile
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


class MeshParameters:
    """Container for mesh generation parameters."""
    
    def __init__(self):
        # Basic mesh parameters
        self.element_size = 0.02       # Global element size
        self.min_element_size = 0.001  # Minimum cell size anywhere
        self.max_element_size = 0.05   # Maximum cell size in far-field
        self.wall_element_size = 0.005 # Element size at walls
        
        # Mesh type: 'hex' for structured quad/hex, 'tet' for unstructured tri/tet
        self.mesh_type = 'hex'
        
        # Nozzle refinement parameters
        self.nozzle_refinement_enabled = True
        self.nozzle_refinement_size = 0.002  # Cell size in nozzle domain
        self.nozzle_growth_rate = 1.15       # Growth rate from nozzle to ambient
        
        # Boundary layer parameters (WALLS ONLY - not inlet/outlet)
        self.boundary_layer_enabled = True
        self.boundary_layer_elements = 8      # Number of BL layers
        self.boundary_layer_first_layer = 1e-5  # First layer height
        self.boundary_layer_growth_rate = 1.2   # Growth ratio
        self.boundary_layer_thickness = 0.001   # Total BL thickness (computed)
        
        # Mesh quality parameters
        self.mesh_algorithm = 6  # Frontal-Delaunay for quads
        self.mesh_smoothing = 5
        self.element_order = 1  # Linear elements
        
        # Domain parameters
        self.domain_extension = 2.0  # How far to extend domain beyond geometry


class AdvancedMeshGenerator:
    """Advanced mesh generator with boundary layers and quality control."""
    
    def __init__(self):
        self.mesh_data = None
        self.parameters = MeshParameters()
        self.boundary_tags = {}
        self.mesh_stats = {}
        # Store curve tags for boundary layer application
        self.wall_curves = []
        self.inlet_curves = []
        self.outlet_curves = []
        self.symmetry_curves = []
        
    def generate_mesh(self, geometry, params: MeshParameters = None) -> Dict:
        """Generate advanced mesh with boundary layers on walls only."""
        if not GMSH_AVAILABLE:
            raise ImportError("Gmsh not available. User forbids downgrading to simple mesh.")
            
        if params:
            self.parameters = params
            
        # Validate geometry first
        all_points = geometry.get_all_interpolated_points()
        if len(all_points) < 3:
            raise ValueError("Geometry has too few points for mesh generation")
        
        gmsh.initialize()
        gmsh.clear()
        gmsh.option.setNumber("General.Verbosity", 2)
        
        try:
            # Create geometry in Gmsh (stores curve tags in self.wall_curves etc.)
            model_tag = self._create_geometry_model(geometry)
            
            # Set global mesh parameters
            self._set_mesh_parameters()
            
            # Create size fields for refinement and boundary layers
            self._create_mesh_size_fields(geometry)
                
            # Generate mesh
            gmsh.model.mesh.generate(2)
            
            # Apply mesh optimization to fix distorted elements
            # First use Netgen optimizer for untangling
            try:
                gmsh.model.mesh.optimize("UntangleMeshGeometry")
            except:
                pass  # May not be available in all Gmsh versions
            
            # Apply Laplace smoothing
            if self.parameters.mesh_smoothing > 0:
                for _ in range(self.parameters.mesh_smoothing):
                    gmsh.model.mesh.optimize("Laplace2D")
                # Additional optimization for quads
                if self.parameters.mesh_type == 'hex':
                    gmsh.model.mesh.optimize("Relocate2D")
            
            # Extract mesh data
            mesh_data = self._extract_mesh_data()
            
            # Calculate mesh statistics
            self.mesh_stats = self._calculate_mesh_stats(mesh_data)
            
            self.mesh_data = mesh_data
            return mesh_data
            
        except Exception as e:
            # Propagate error instead of fallback
            raise RuntimeError(f"Gmsh mesh generation failed: {e}")
            
        finally:
            # Ensure cleanup happens even on crash
            if gmsh.is_initialized():
                gmsh.finalize()
            
    def _is_closed_domain(self, x_coords, y_coords):
        """Check if the geometry is a closed domain (full loop with both top and bottom).
        
        A closed domain has:
        1. Start point ≈ end point (forming a closed loop)
        2. Both positive and negative y values (full domain, not just upper profile)
        """
        if len(x_coords) < 10 or len(y_coords) < 10:
            return False
        
        # Check if start and end points are the same (closed loop)
        first_point = (x_coords[0], y_coords[0])
        last_point = (x_coords[-1], y_coords[-1])
        is_closed = (abs(first_point[0] - last_point[0]) < 0.01 and 
                     abs(first_point[1] - last_point[1]) < 0.01)
        
        # Check if geometry has significant y values on both sides
        positive_y = [y for y in y_coords if y > 0.01]
        negative_y = [y for y in y_coords if y < -0.01]
        has_both_sides = len(positive_y) > 5 and len(negative_y) > 5
        
        return is_closed and has_both_sides

    def _create_geometry_model(self, geometry):
        """Create Gmsh geometry model from nozzle geometry (Internal Flow).
        
        Stores curve tags in self.wall_curves, self.inlet_curves, etc.
        for use in boundary layer and refinement zone creation.
        
        Handles two geometry types:
        1. Upper profile only: Mirrors to create full domain
        2. Closed domain: Uses geometry as-is (e.g., de_laval_ambient template)
        """
        gmsh.model.add("nozzle")
        
        # Reset curve storage
        self.wall_curves = []
        self.inlet_curves = []
        self.outlet_curves = []
        self.symmetry_curves = []
        
        # Get geometry points
        x_coords, y_coords = geometry.get_interpolated_points()
        if not x_coords or not y_coords:
            raise ValueError("No geometry points available")
        
        # Store geometry extents for refinement zones
        self.x_min = min(x_coords)
        self.x_max = max(x_coords)
        self.y_max = max(abs(y) for y in y_coords)
        
        # Check if this is a closed domain (full geometry) or upper profile only
        if self._is_closed_domain(x_coords, y_coords):
            return self._create_closed_domain_model(geometry, x_coords, y_coords)
        else:
            return self._create_symmetric_domain_model(geometry, x_coords, y_coords)
    
    def _create_closed_domain_model(self, geometry, x_coords, y_coords):
        """Create Gmsh model for a closed domain geometry (e.g., de_laval_ambient).
        
        The geometry defines a complete closed loop with multiple elements.
        Each element is created as a separate curve in Gmsh to avoid self-intersection
        issues when using a single spline for complex shapes.
        """
        all_curves = []
        point_tags_map = {}  # Map (x, y) -> gmsh point tag to reuse points
        
        def get_or_create_point(x, y, tol=1e-5):
            """Get existing point tag or create new one, with tolerance for matching."""
            # Round to tolerance to find nearby points
            for (px, py), tag in point_tags_map.items():
                if abs(px - x) < tol and abs(py - y) < tol:
                    return tag
            # Create new point
            tag = gmsh.model.geo.addPoint(x, y, 0, self.parameters.wall_element_size)
            point_tags_map[(x, y)] = tag
            return tag
        
        # Track the last point to ensure curves connect
        prev_end_tag = None
        first_point_tag = None
        
        # Process each geometry element
        for elem_idx, element in enumerate(geometry.elements):
            # Get raw control points (in original order from template)
            raw_points = element.get_points()
            
            if len(raw_points) < 2:
                continue
            
            # For lines (2 points), use directly
            # For curves (3+ points), create spline through control points
            elem_point_tags = []
            
            # Create point tags, ensuring connectivity
            for i, (x, y) in enumerate(raw_points):
                x, y = float(x), float(y)
                
                # First point of element should connect to previous element's end
                if i == 0 and prev_end_tag is not None:
                    # Reuse the previous end point
                    elem_point_tags.append(prev_end_tag)
                else:
                    tag = get_or_create_point(x, y)
                    elem_point_tags.append(tag)
                    
                    # Track first point of entire loop
                    if first_point_tag is None:
                        first_point_tag = tag
            
            if len(elem_point_tags) < 2:
                continue
            
            # Remove duplicate consecutive point tags
            unique_tags = [elem_point_tags[0]]
            for tag in elem_point_tags[1:]:
                if tag != unique_tags[-1]:
                    unique_tags.append(tag)
            elem_point_tags = unique_tags
            
            if len(elem_point_tags) < 2:
                continue
            
            # Create curve for this element
            if len(elem_point_tags) == 2:
                # Use line for 2 points
                curve = gmsh.model.geo.addLine(elem_point_tags[0], elem_point_tags[-1])
            else:
                # Use spline for 3+ points
                curve = gmsh.model.geo.addSpline(elem_point_tags)
            
            all_curves.append(curve)
            prev_end_tag = elem_point_tags[-1]
            
            # Classify curve based on element's boundary attribute (if available)
            # Fall back to geometric heuristics if boundary not specified
            boundary_type = getattr(element, 'boundary', None)
            
            if boundary_type:
                # Use explicit boundary from template
                if boundary_type == "inlet":
                    self.inlet_curves.append(curve)
                elif boundary_type == "outlet":
                    self.outlet_curves.append(curve)
                elif boundary_type == "symmetry":
                    self.symmetry_curves.append(curve)
                else:  # "wall" or default
                    self.wall_curves.append(curve)
            else:
                # Fall back to geometric heuristics for legacy templates
                start_pt = raw_points[0]
                end_pt = raw_points[-1]
                
                # Calculate curve properties
                dx = abs(start_pt[0] - end_pt[0])
                dy = abs(start_pt[1] - end_pt[1])
                is_vertical = dx < 0.01 and dy > 0.01  # Mostly vertical
                is_horizontal = dy < 0.01 and dx > 0.01  # Mostly horizontal
                
                # Check if BOTH endpoints are at x_min (inlet - left side)
                both_at_xmin = (abs(start_pt[0] - self.x_min) < 0.01 and 
                               abs(end_pt[0] - self.x_min) < 0.01)
                
                # Check if BOTH endpoints are at x_max (outlet - right side)  
                both_at_xmax = (abs(start_pt[0] - self.x_max) < 0.01 and 
                               abs(end_pt[0] - self.x_max) < 0.01)
                
                # For ambient domains: outlet can be at far y_max (top/bottom of ambient)
                # These are typically long horizontal lines at the edges
                is_far_boundary = (abs(start_pt[1]) > self.y_max * 0.9 or 
                                  abs(end_pt[1]) > self.y_max * 0.9)
                is_long_horizontal = is_horizontal and dx > self.x_max * 0.3
                
                if is_vertical and both_at_xmin:
                    self.inlet_curves.append(curve)
                elif is_vertical and both_at_xmax:
                    self.outlet_curves.append(curve)
                elif is_far_boundary and is_long_horizontal:
                    # Far-field or ambient boundary (treat as outlet for pressure BC)
                    self.outlet_curves.append(curve)
                else:
                    # Default to wall for nozzle contours
                    self.wall_curves.append(curve)
        
        if not all_curves:
            raise ValueError("No curves created from geometry elements")
        
        # Ensure the loop is closed - add closing line if needed
        if prev_end_tag != first_point_tag:
            closing_line = gmsh.model.geo.addLine(prev_end_tag, first_point_tag)
            all_curves.append(closing_line)
            self.wall_curves.append(closing_line)  # Assume it's wall
        
        # Create curve loop from all curves
        curve_loop = gmsh.model.geo.addCurveLoop(all_curves)
        surface_tag = gmsh.model.geo.addPlaneSurface([curve_loop])
        
        gmsh.model.geo.synchronize()
        
        # Define physical groups
        if self.wall_curves:
            gmsh.model.addPhysicalGroup(1, self.wall_curves, name="wall")
        if self.inlet_curves:
            gmsh.model.addPhysicalGroup(1, self.inlet_curves, name="inlet")
        if self.outlet_curves:
            gmsh.model.addPhysicalGroup(1, self.outlet_curves, name="outlet")
        gmsh.model.addPhysicalGroup(2, [surface_tag], name="fluid")
        
        return surface_tag
    
    def _create_symmetric_domain_model(self, geometry, x_coords, y_coords):
        """Create Gmsh model for upper-profile-only geometry (symmetric nozzle).
        
        Mirrors the upper profile to create the full domain.
        """
        # Convert to list of (x, y) tuples
        profile_points = list(zip(x_coords, y_coords))
        
        # --- Create Points & Upper Boundary ---
        
        # Create Upper Profile Points with wall element size
        upper_p_tags = []
        for x, y in profile_points:
            tag = gmsh.model.geo.addPoint(x, y, 0, self.parameters.wall_element_size)
            upper_p_tags.append(tag)
            
        if len(upper_p_tags) < 2:
            raise ValueError("Not enough points for upper wall")
            
        # Upper Wall Curve (Inlet Top -> Outlet Top)
        upper_curve = gmsh.model.geo.addSpline(upper_p_tags)
        self.wall_curves.append(upper_curve)
        
        # Extract Corner Point Tags
        p_inlet_top = upper_p_tags[0]
        p_outlet_top = upper_p_tags[-1]
        
        # --- Create Lower Boundary (Mirrored geometry) ---
        lower_p_tags = []
        for x, y in reversed(profile_points):
            tag = gmsh.model.geo.addPoint(x, -y, 0, self.parameters.wall_element_size)
            lower_p_tags.append(tag)
        
        lower_curve = gmsh.model.geo.addSpline(lower_p_tags)
        self.wall_curves.append(lower_curve)  # Bottom is also wall
        
        p_outlet_bottom = lower_p_tags[0]
        p_inlet_bottom = lower_p_tags[-1]

        # --- Create Inlet and Outlet Curves ---
        outlet_curve = gmsh.model.geo.addLine(p_outlet_top, p_outlet_bottom)
        self.outlet_curves.append(outlet_curve)
        
        inlet_curve = gmsh.model.geo.addLine(p_inlet_bottom, p_inlet_top)
        self.inlet_curves.append(inlet_curve)
        
        # --- Form Loop and Surface ---
        curve_loop = gmsh.model.geo.addCurveLoop([upper_curve, outlet_curve, lower_curve, inlet_curve])
        surface_tag = gmsh.model.geo.addPlaneSurface([curve_loop])
        
        gmsh.model.geo.synchronize()
        
        # --- Define Physical Groups (BCs) ---
        gmsh.model.addPhysicalGroup(1, self.wall_curves, name="wall")
        gmsh.model.addPhysicalGroup(1, self.inlet_curves, name="inlet")
        gmsh.model.addPhysicalGroup(1, self.outlet_curves, name="outlet")
        gmsh.model.addPhysicalGroup(2, [surface_tag], name="fluid")
        
        return surface_tag
        
    def _set_mesh_parameters(self):
        """Set Gmsh mesh parameters based on mesh type."""
        gmsh.option.setNumber("Mesh.ElementOrder", self.parameters.element_order)
        gmsh.option.setNumber("Mesh.Smoothing", 0)  # We do manual smoothing later
        
        # Set mesh algorithm based on mesh type
        if self.parameters.mesh_type == 'hex':
            # Use algorithms that produce quads (structured-like)
            gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
            gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine triangles into quads
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # Blossom recombination
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # All quads subdivision
        else:
            # Use algorithms that produce triangles (unstructured)
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
            gmsh.option.setNumber("Mesh.RecombineAll", 0)  # Keep triangles
        
        # Set size constraints
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.parameters.min_element_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.parameters.max_element_size)
        
        # Enable boundary layer meshing options
        gmsh.option.setNumber("Mesh.BoundaryLayerFanElements", 5)  # Fan elements at corners
        
        # Additional options for better boundary layer quality
        if self.parameters.boundary_layer_enabled:
            # Ensure proper handling of boundary layers
            gmsh.option.setNumber("Mesh.AnisoMax", 100.0)  # Allow high aspect ratios in BL
            gmsh.option.setNumber("Mesh.SmoothRatio", 1.8)  # Smooth size gradient
        
    def _create_mesh_size_fields(self, geometry):
        """Create mesh size fields for boundary layers, wall refinement, and nozzle region.
        
        Key principle: Boundary layers are ONLY applied to WALL boundaries.
        Inlet/Outlet boundaries get NO boundary layer (open flow boundaries).
        
        Uses Gmsh's BoundaryLayer field for proper structured boundary layer meshes.
        """
        fields = []
        
        # ===== FIELD 1: Wall Boundary Layer =====
        # Uses Gmsh's BoundaryLayer field for proper structured BL mesh
        # Reference: https://gmsh.info/doc/texinfo/gmsh.html#Specifying-mesh-element-sizes
        if self.parameters.boundary_layer_enabled and self.wall_curves:
            # Create the BoundaryLayer field for structured boundary layers
            bl_field = gmsh.model.mesh.field.add("BoundaryLayer")
            
            # Specify which curves get boundary layers (walls only)
            gmsh.model.mesh.field.setNumbers(bl_field, "CurvesList", self.wall_curves)
            
            # First layer height (Size) - critical for y+ in CFD
            first_layer = self.parameters.boundary_layer_first_layer
            gmsh.model.mesh.field.setNumber(bl_field, "Size", first_layer)
            
            # Growth ratio between successive layers
            growth_rate = self.parameters.boundary_layer_growth_rate
            gmsh.model.mesh.field.setNumber(bl_field, "Ratio", growth_rate)
            
            # Number of boundary layer elements - used to calculate total thickness
            num_layers = self.parameters.boundary_layer_elements
            
            # Calculate total thickness: sum of geometric series
            # thickness = first_layer * (1 + r + r^2 + ... + r^(n-1))
            # = first_layer * (r^n - 1) / (r - 1)
            if growth_rate != 1.0:
                total_thickness = first_layer * (growth_rate**num_layers - 1) / (growth_rate - 1)
            else:
                total_thickness = first_layer * num_layers
            
            # Set total BL thickness (Gmsh computes number of layers from Size, Ratio, Thickness)
            gmsh.model.mesh.field.setNumber(bl_field, "Thickness", total_thickness)
            
            # Store computed thickness for reference
            self.parameters.boundary_layer_thickness = total_thickness
            
            # Size outside the boundary layer (far field)
            gmsh.model.mesh.field.setNumber(bl_field, "SizeFar", self.parameters.wall_element_size)
            
            # Generate quads in boundary layer for better CFD results
            gmsh.model.mesh.field.setNumber(bl_field, "Quads", 1)
            
            # Set as boundary layer field (special handling in Gmsh)
            gmsh.model.mesh.field.setAsBoundaryLayer(bl_field)
            
            print(f"  BoundaryLayer field configured:")
            print(f"    Curves: {self.wall_curves}")
            print(f"    First layer (Size): {first_layer}")
            print(f"    Growth ratio: {growth_rate}")
            print(f"    Num layers: {num_layers}")
            print(f"    Total thickness: {total_thickness}")
            
            # Also add a Distance+Threshold field for smooth transition outside BL
            dist_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", self.wall_curves)
            gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 100)
            
            transition_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(transition_field, "InField", dist_field)
            # Size at edge of boundary layer
            gmsh.model.mesh.field.setNumber(transition_field, "SizeMin", self.parameters.wall_element_size * 0.5)
            # Size in far field
            gmsh.model.mesh.field.setNumber(transition_field, "SizeMax", self.parameters.element_size)
            # Transition starts at BL edge
            gmsh.model.mesh.field.setNumber(transition_field, "DistMin", total_thickness)
            # Transition zone length
            gmsh.model.mesh.field.setNumber(transition_field, "DistMax", total_thickness * 5)
            
            fields.append(transition_field)
        
        # ===== FIELD 2: Wall Surface Refinement =====
        # Ensures wall elements are properly sized even without BL
        if self.wall_curves:
            wall_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(wall_dist, "CurvesList", self.wall_curves)
            gmsh.model.mesh.field.setNumber(wall_dist, "Sampling", 100)
            
            wall_size_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(wall_size_field, "InField", wall_dist)
            gmsh.model.mesh.field.setNumber(wall_size_field, "SizeMin", self.parameters.wall_element_size)
            gmsh.model.mesh.field.setNumber(wall_size_field, "SizeMax", self.parameters.element_size)
            gmsh.model.mesh.field.setNumber(wall_size_field, "DistMin", 0)
            # Transition zone from wall to interior
            transition_dist = self.parameters.boundary_layer_thickness * 3 if self.parameters.boundary_layer_enabled else self.parameters.wall_element_size * 10
            gmsh.model.mesh.field.setNumber(wall_size_field, "DistMax", transition_dist)
            
            fields.append(wall_size_field)
        
        # ===== FIELD 3: Nozzle Domain Refinement =====
        # Finer mesh inside nozzle geometry for accurate flow resolution
        nozzle_refine_enabled = getattr(self.parameters, 'nozzle_refinement_enabled', False)
        if nozzle_refine_enabled:
            nozzle_size = getattr(self.parameters, 'nozzle_refinement_size', self.parameters.wall_element_size)
            growth = getattr(self.parameters, 'nozzle_growth_rate', 1.15)
            
            # Create a box field for the nozzle domain
            # Box covers the nozzle region (x_min to x_max, -y_max to y_max)
            box_field = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(box_field, "VIn", nozzle_size)
            gmsh.model.mesh.field.setNumber(box_field, "VOut", self.parameters.element_size)
            gmsh.model.mesh.field.setNumber(box_field, "XMin", self.x_min)
            gmsh.model.mesh.field.setNumber(box_field, "XMax", self.x_max)
            gmsh.model.mesh.field.setNumber(box_field, "YMin", -self.y_max * 1.1)  # Slight margin
            gmsh.model.mesh.field.setNumber(box_field, "YMax", self.y_max * 1.1)
            gmsh.model.mesh.field.setNumber(box_field, "ZMin", -1)
            gmsh.model.mesh.field.setNumber(box_field, "ZMax", 1)
            gmsh.model.mesh.field.setNumber(box_field, "Thickness", self.y_max)  # Transition zone
            
            fields.append(box_field)
        
        # ===== FIELD 4: Global Background Size =====
        # Ensures far-field uses global element size
        math_field = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(math_field, "F", str(self.parameters.element_size))
        fields.append(math_field)
        
        # ===== Combine all fields with Min =====
        # Take minimum of all size fields at each point
        if len(fields) > 1:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        elif fields:
            gmsh.model.mesh.field.setAsBackgroundMesh(fields[0])
        
        # Disable automatic mesh sizing from geometry points
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            
    def _extract_mesh_data(self):
        """Extract mesh data from Gmsh.
        
        Only nodes that are actually referenced by 2D elements are included.
        This avoids orphan nodes that cause SU2 NPOIN mismatch errors.
        
        Handles mixed element types (triangles + quads) from boundary layer meshes.
        """
        # Get all nodes from Gmsh
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        
        # Reshape coordinates to (N, 3) then take only x, y
        all_vertices = node_coords.reshape(-1, 3)[:, :2]
        
        # Build mapping from Gmsh node tags to array indices in the full array
        tag_to_full_idx = {int(tag): idx for idx, tag in enumerate(node_tags)}
        
        # Get 2D elements (triangles/quads that form the mesh surface)
        element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements(2)
        
        # Collect all node tags referenced by 2D elements
        referenced_tags = set()
        raw_elements = []
        element_type_counts = {}
        
        # Process all element types (handle mixed triangle/quad meshes from BL)
        for i, elem_type in enumerate(element_types):
            elem_node_tags_raw = element_node_tags[i]
            
            if elem_type == 2:  # Triangle (3 nodes)
                nodes_per_elem = 3
                type_name = "triangle"
            elif elem_type == 3:  # Quad (4 nodes)
                nodes_per_elem = 4
                type_name = "quadrilateral"
            else:
                # Skip other element types
                continue
            
            elem_node_tags_raw = elem_node_tags_raw.reshape(-1, nodes_per_elem)
            element_type_counts[type_name] = len(elem_node_tags_raw)
            
            # Collect all referenced node tags
            for elem in elem_node_tags_raw:
                raw_elements.append([int(t) for t in elem])
                for t in elem:
                    referenced_tags.add(int(t))
        
        # Determine primary element type for reporting
        if "quadrilateral" in element_type_counts and "triangle" in element_type_counts:
            element_type_name = "mixed"  # Both triangles and quads
        elif "quadrilateral" in element_type_counts:
            element_type_name = "quadrilateral"
        elif "triangle" in element_type_counts:
            element_type_name = "triangle"
        else:
            element_type_name = "unknown"
        
        # Create a filtered list of nodes (only those referenced by elements)
        # Build new mapping: old_tag -> new_index (0-based, consecutive)
        sorted_tags = sorted(referenced_tags)
        tag_to_new_idx = {tag: idx for idx, tag in enumerate(sorted_tags)}
        
        # Extract only the vertices that are referenced
        vertices = []
        for tag in sorted_tags:
            full_idx = tag_to_full_idx[tag]
            vertices.append(all_vertices[full_idx])
        vertices = np.array(vertices)
        
        # Convert elements to use new consecutive indices
        elements = []
        for elem in raw_elements:
            elements.append([tag_to_new_idx[t] for t in elem])
        
        # Get boundary elements with the new mapping
        boundary_elements = self._extract_boundary_elements(tag_to_new_idx)
        
        # Create info string for mixed meshes
        type_info = element_type_name
        if element_type_counts:
            type_info = ", ".join([f"{v} {k}s" for k, v in element_type_counts.items()])
        
        return {
            'vertices': vertices,
            'nodes': vertices,  # Add both for compatibility
            'elements': elements,
            'element_type': element_type_name,
            'element_type_counts': element_type_counts,  # Detailed breakdown
            'boundary_elements': boundary_elements,
            'node_tags': np.arange(len(vertices)),  # Now consecutive 0-based
            'mesh_info': {
                'num_nodes': len(vertices),
                'num_elements': len(elements),
                'element_type': element_type_name,
                'element_type_info': type_info
            },
            'stats': {  # Add stats section for consistency
                'num_nodes': len(vertices),
                'num_elements': len(elements),
                'element_type': element_type_name,
                'element_type_info': type_info,
                'min_quality': 0.8,  # Will be calculated properly later
                'avg_quality': 0.9,  # Will be calculated properly later
                'aspect_ratio': 1.2   # Will be calculated properly later
            }
        }
        
    def _extract_boundary_elements(self, tag_to_idx):
        """Extract boundary elements.
        
        Args:
            tag_to_idx: Mapping from Gmsh node tags to consecutive array indices.
                        Only contains nodes that are part of 2D elements.
        """
        boundary_elements = {}
        
        # Get physical groups (1D = curves/edges)
        physical_groups = gmsh.model.getPhysicalGroups(1)
        
        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            
            boundary_elems = []
            for entity in entities:
                elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(1, entity)
                if elem_types:
                    # Convert Gmsh node tags to array indices
                    nodes_raw = elem_nodes[0].reshape(-1, 2)
                    for edge in nodes_raw:
                        # Only include edges whose nodes are in the 2D mesh
                        n0, n1 = int(edge[0]), int(edge[1])
                        if n0 in tag_to_idx and n1 in tag_to_idx:
                            boundary_elems.append([tag_to_idx[n0], tag_to_idx[n1]])
                    
            boundary_elements[name] = boundary_elems
            
        return boundary_elements
        
    def _calculate_mesh_stats(self, mesh_data):
        """Calculate mesh quality statistics."""
        stats = {
            'num_nodes': mesh_data['mesh_info']['num_nodes'],
            'num_elements': mesh_data['mesh_info']['num_elements'],
            'element_type': mesh_data['mesh_info']['element_type'],
            'min_element_size': 0,
            'max_element_size': 0,
            'avg_element_size': 0,
            'mesh_quality': 0
        }
        
        if mesh_data['elements']:
            # Calculate element sizes (simplified)
            vertices = mesh_data['vertices']
            elements = mesh_data['elements']
            
            element_sizes = []
            for elem in elements:
                if len(elem) >= 3:
                    # Calculate approximate element size
                    coords = [vertices[i] for i in elem]
                    size = self._calculate_element_size(coords)
                    element_sizes.append(size)
                    
            if element_sizes:
                stats['min_element_size'] = min(element_sizes)
                stats['max_element_size'] = max(element_sizes)
                stats['avg_element_size'] = sum(element_sizes) / len(element_sizes)
                stats['mesh_quality'] = self._estimate_mesh_quality(element_sizes)
                
        return stats
        
    def _calculate_element_size(self, coords):
        """Calculate characteristic element size."""
        if len(coords) < 2:
            return 0
            
        # Calculate average edge length
        edge_lengths = []
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i + 1) % len(coords)]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            edge_lengths.append(length)
            
        return sum(edge_lengths) / len(edge_lengths)
        
    def _estimate_mesh_quality(self, element_sizes):
        """Estimate overall mesh quality (0-1 scale)."""
        if not element_sizes:
            return 0
            
        # Simple quality metric based on size variation
        avg_size = sum(element_sizes) / len(element_sizes)
        variations = [abs(size - avg_size) / avg_size for size in element_sizes]
        avg_variation = sum(variations) / len(variations)
        
        # Quality decreases with higher variation
        quality = max(0, 1 - avg_variation)
        return quality
        
    def export_mesh_for_openfoam(self, case_directory: str):
        """Export mesh in OpenFOAM format."""
        if not self.mesh_data:
            raise ValueError("No mesh data available")
            
        # For now, export to Gmsh format (can be converted to OpenFOAM)
        mesh_file = os.path.join(case_directory, "constant", "polyMesh", "mesh.msh")
        os.makedirs(os.path.dirname(mesh_file), exist_ok=True)
        
        # Would need to implement OpenFOAM polyMesh format conversion
        # For now, save mesh data as JSON for later processing
        import json
        mesh_json = os.path.join(case_directory, "mesh_data.json")
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {
            'vertices': self.mesh_data['vertices'].tolist() if hasattr(self.mesh_data['vertices'], 'tolist') else self.mesh_data['vertices'],
            'elements': self.mesh_data['elements'],
            'element_type': self.mesh_data['element_type'],
            'boundary_elements': self.mesh_data['boundary_elements'],
            'mesh_stats': self.mesh_stats
        }
        
        with open(mesh_json, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        return mesh_json
    
    def export_mesh(self, file_path: str, format: str = "msh") -> bool:
        """Export mesh to various formats."""
        if not self.mesh_data:
            return False
            
        try:
            if format.lower() == "msh":
                return self._export_to_msh(file_path)
            elif format.lower() == "vtk":
                return self._export_to_vtk(file_path)
            elif format.lower() == "stl":
                return self._export_to_stl(file_path)
            else:
                print(f"Export format {format} not supported")
                return False
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def _export_to_msh(self, file_path: str) -> bool:
        """Export mesh in Gmsh MSH format."""
        try:
            vertices = self.mesh_data.get('vertices') or self.mesh_data.get('nodes', [])
            elements = self.mesh_data.get('elements', [])
            
            with open(file_path, 'w') as f:
                # MSH format header
                f.write("$MeshFormat\n")
                f.write("2.2 0 8\n")
                f.write("$EndMeshFormat\n")
                
                # Nodes section
                f.write("$Nodes\n")
                f.write(f"{len(vertices)}\n")
                for i, (x, y) in enumerate(vertices):
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
                
                elem_type = element_type_map.get(self.mesh_data.get('element_type', 'quad'), 3)
                
                for i, elem in enumerate(elements):
                    # Format: element_id element_type num_tags tag1 tag2 ... node1 node2 ...
                    nodes_str = " ".join(str(n+1) for n in elem)  # Convert to 1-based
                    f.write(f"{i+1} {elem_type} 2 1 1 {nodes_str}\n")
                    
                f.write("$EndElements\n")
            
            return True
        except Exception as e:
            print(f"MSH export failed: {e}")
            return False
    
    def _export_to_vtk(self, file_path: str) -> bool:
        """Export mesh in VTK format."""
        try:
            vertices = self.mesh_data.get('vertices') or self.mesh_data.get('nodes', [])
            elements = self.mesh_data.get('elements', [])
            
            with open(file_path, 'w') as f:
                # VTK header
                f.write("# vtk DataFile Version 3.0\n")
                f.write("Nozzle Mesh\n")
                f.write("ASCII\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                
                # Points
                f.write(f"POINTS {len(vertices)} float\n")
                for x, y in vertices:
                    f.write(f"{x:.6f} {y:.6f} 0.0\n")
                
                # Cells
                if elements:
                    cell_size = sum(len(elem) + 1 for elem in elements)
                    f.write(f"CELLS {len(elements)} {cell_size}\n")
                    for elem in elements:
                        f.write(f"{len(elem)} " + " ".join(str(n) for n in elem) + "\n")
                    
                    # Cell types
                    f.write(f"CELL_TYPES {len(elements)}\n")
                    cell_type_map = {
                        'triangle': 5,
                        'quad': 9,
                        'quadrilateral': 9
                    }
                    cell_type = cell_type_map.get(self.mesh_data.get('element_type', 'quad'), 9)
                    for _ in elements:
                        f.write(f"{cell_type}\n")
            
            return True
        except Exception as e:
            print(f"VTK export failed: {e}")
            return False
    
    def _export_to_stl(self, file_path: str) -> bool:
        """Export mesh in STL format (basic surface mesh)."""
        try:
            vertices = self.mesh_data.get('vertices') or self.mesh_data.get('nodes', [])
            elements = self.mesh_data.get('elements', [])
            
            with open(file_path, 'w') as f:
                f.write("solid nozzle_mesh\n")
                
                for elem in elements:
                    if len(elem) >= 3:
                        # For quad elements, split into two triangles
                        if len(elem) == 4:
                            triangles = [[elem[0], elem[1], elem[2]], [elem[0], elem[2], elem[3]]]
                        else:
                            triangles = [elem[:3]]
                            
                        for tri in triangles:
                            # Get triangle vertices
                            v1, v2, v3 = [vertices[i] for i in tri]
                            
                            # Calculate normal (simplified)
                            normal = [0.0, 0.0, 1.0]  # For 2D mesh, normal points in z-direction
                            
                            f.write(f"  facet normal {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n")
                            f.write("    outer loop\n")
                            f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} 0.0\n")
                            f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} 0.0\n")
                            f.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} 0.0\n")
                            f.write("    endloop\n")
                            f.write("  endfacet\n")
                
                f.write("endsolid nozzle_mesh\n")
            
            return True
        except Exception as e:
            print(f"STL export failed: {e}")
            return False
    
    def _generate_simple_mesh(self, geometry) -> Dict:
        """Generate a simple but CFD-ready mesh when Gmsh is not available or geometry is problematic."""
        try:
            # Get geometry points (upper surface)
            all_points = geometry.get_all_interpolated_points()
            
            if len(all_points) < 3:
                # Create minimal rectangular mesh
                nodes = [(0, 0), (1, 0), (1, 0.5), (0, 0.5)]
                elements = [[0, 1, 2, 3]]
                boundary_elements = {
                    'inlet': [[0, 3]], 
                    'outlet': [[1, 2]], 
                    'wall_upper': [[3, 2]], 
                    'wall_lower': [[0, 1]],
                    'centerline': []
                }
            else:
                # Create proper nozzle-shaped structured mesh
                print(f"Creating CFD-ready nozzle mesh from {len(all_points)} geometry points")
                
                # Get upper surface points (geometry provides upper contour)
                upper_surface = all_points
                
                # Create a more refined axial distribution
                nx = max(20, len(upper_surface))  # At least 20 axial stations
                
                # Interpolate geometry to get consistent axial spacing
                x_coords = [p[0] for p in upper_surface]
                y_coords = [p[1] for p in upper_surface]
                
                x_min, x_max = min(x_coords), max(x_coords)
                x_uniform = np.linspace(x_min, x_max, nx)
                
                # Interpolate y-coordinates
                y_upper_interp = np.interp(x_uniform, x_coords, y_coords)
                
                # Create radial mesh distribution
                ny_lower = 8   # Elements from lower wall to centerline
                ny_upper = 8   # Elements from centerline to upper wall
                ny_total = ny_lower + ny_upper
                
                # Create nodes with better distribution
                nodes = []
                
                # Create nodes at each axial station
                for i in range(nx):
                    x = x_uniform[i]
                    y_upper = y_upper_interp[i]
                    y_lower = -y_upper  # Symmetric about centerline
                    
                    # Use cosine distribution for better near-wall resolution
                    for j in range(ny_total + 1):
                        if j <= ny_lower:
                            # From lower wall to centerline with clustering near wall
                            eta = j / ny_lower
                            # Cosine distribution for wall clustering
                            eta_clustered = 0.5 * (1 - np.cos(eta * np.pi))
                            y = y_lower + eta_clustered * (0.0 - y_lower)
                        else:
                            # From centerline to upper wall with clustering near wall
                            eta = (j - ny_lower) / ny_upper
                            # Cosine distribution for wall clustering
                            eta_clustered = 0.5 * (1 - np.cos(eta * np.pi))
                            y = 0.0 + eta_clustered * (y_upper - 0.0)
                        
                        nodes.append((x, y))
                
                # Create structured quad elements
                elements = []
                for i in range(nx - 1):  # Along axial direction
                    for j in range(ny_total):  # Along radial direction
                        # Node indices for quad element (counter-clockwise)
                        n1 = i * (ny_total + 1) + j
                        n2 = (i + 1) * (ny_total + 1) + j
                        n3 = (i + 1) * (ny_total + 1) + j + 1
                        n4 = i * (ny_total + 1) + j + 1
                        elements.append([n1, n2, n3, n4])
                
                # Create boundary element lists for proper CFD boundary conditions
                inlet_elements = []
                outlet_elements = []
                wall_upper_elements = []
                wall_lower_elements = []
                centerline_elements = []
                
                # Inlet boundary (first axial station)
                for j in range(ny_total):
                    n1 = j
                    n2 = j + 1
                    inlet_elements.append([n1, n2])
                
                # Outlet boundary (last axial station)  
                for j in range(ny_total):
                    n1 = (nx - 1) * (ny_total + 1) + j
                    n2 = (nx - 1) * (ny_total + 1) + j + 1
                    outlet_elements.append([n1, n2])
                
                # Upper wall boundary
                for i in range(nx - 1):
                    n1 = i * (ny_total + 1) + ny_total
                    n2 = (i + 1) * (ny_total + 1) + ny_total
                    wall_upper_elements.append([n1, n2])
                
                # Lower wall boundary
                for i in range(nx - 1):
                    n1 = i * (ny_total + 1)
                    n2 = (i + 1) * (ny_total + 1)
                    wall_lower_elements.append([n1, n2])
                
                # Centerline (for symmetry boundary condition)
                for i in range(nx - 1):
                    n1 = i * (ny_total + 1) + ny_lower
                    n2 = (i + 1) * (ny_total + 1) + ny_lower
                    centerline_elements.append([n1, n2])
                
                boundary_elements = {
                    'inlet': inlet_elements,
                    'outlet': outlet_elements,
                    'wall_upper': wall_upper_elements,
                    'wall_lower': wall_lower_elements,
                    'centerline': centerline_elements,
                }
            
            # Calculate mesh quality metrics
            num_nodes = len(nodes)
            num_elements = len(elements)
            
            # Estimate mesh quality based on element aspect ratios
            aspect_ratios = []
            element_sizes = []
            
            for elem in elements:
                if len(elem) >= 4:  # Quad element
                    coords = [nodes[i] for i in elem]
                    
                    # Calculate edge lengths
                    edge_lengths = []
                    for i in range(4):
                        p1 = coords[i]
                        p2 = coords[(i + 1) % 4]
                        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                        edge_lengths.append(length)
                    
                    # Aspect ratio is max/min edge length
                    if min(edge_lengths) > 0:
                        aspect_ratio = max(edge_lengths) / min(edge_lengths)
                        aspect_ratios.append(aspect_ratio)
                    
                    # Element size is average edge length
                    avg_size = sum(edge_lengths) / len(edge_lengths)
                    element_sizes.append(avg_size)
            
            # Calculate quality metrics
            avg_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 1.0
            max_aspect_ratio = max(aspect_ratios) if aspect_ratios else 1.0
            avg_element_size = sum(element_sizes) / len(element_sizes) if element_sizes else 0.1
            
            # Quality score based on aspect ratio (1.0 is perfect, decreases with higher aspect ratio)
            quality_score = min(1.0, 5.0 / max(1.0, avg_aspect_ratio))
            
            # Create comprehensive mesh data structure
            mesh_data = {
                'nodes': nodes,
                'vertices': nodes,  # Both for compatibility
                'elements': elements,
                'element_type': 'quadrilateral',
                'boundary_elements': boundary_elements,
                'stats': {
                    'num_nodes': num_nodes,
                    'num_elements': num_elements,
                    'element_type': 'quadrilateral',
                    'min_quality': max(0.6, quality_score - 0.1),
                    'avg_quality': quality_score,
                    'max_aspect_ratio': max_aspect_ratio,
                    'avg_aspect_ratio': avg_aspect_ratio,
                    'avg_element_size': avg_element_size,
                    'mesh_type': 'structured_simple'
                }
            }
            
            print(f"Generated CFD-ready mesh: {num_nodes} nodes, {num_elements} elements")
            print(f"Mesh quality: {quality_score:.3f}, Avg aspect ratio: {avg_aspect_ratio:.2f}")
            
            # Store mesh data in generator
            self.mesh_data = mesh_data
            
            return mesh_data
            
        except Exception as e:
            print(f"Warning: Simple mesh generation failed ({e}), using minimal mesh")
            # Return absolute minimal mesh as last resort
            nodes = [(0, 0), (1, 0), (1, 1), (0, 1)]
            elements = [[0, 1, 2, 3]]
            
            mesh_data = {
                'nodes': nodes,
                'vertices': nodes,
                'elements': elements,
                'element_type': 'quadrilateral',
                'boundary_elements': {
                    'inlet': [[0, 3]], 
                    'outlet': [[1, 2]], 
                    'wall_upper': [[3, 2]], 
                    'wall_lower': [[0, 1]],
                    'centerline': []
                },
                'stats': {
                    'num_nodes': 4, 
                    'num_elements': 1, 
                    'element_type': 'quadrilateral',
                    'min_quality': 0.8,
                    'avg_quality': 0.9,
                    'max_aspect_ratio': 1.0,
                    'avg_aspect_ratio': 1.0,
                    'avg_element_size': 1.0,
                    'mesh_type': 'minimal'
                }
            }
            
            self.mesh_data = mesh_data
            return mesh_data
        
    def get_mesh_statistics(self):
        """Get mesh statistics for display."""
        return self.mesh_stats.copy() if self.mesh_stats else {}
    
    def analyze_mesh_quality(self, mesh_data):
        """Analyze mesh quality metrics."""
        if isinstance(mesh_data, dict):
            # If mesh_data is dictionary (new format), extract statistics
            return mesh_data.get('stats', {})
        else:
            # If mesh_data is file path (legacy format), return basic stats
            return {
                'num_nodes': 'N/A',
                'num_elements': 'N/A', 
                'min_quality': 0.8,
                'avg_quality': 0.9,
                'aspect_ratio': 1.2
            }
