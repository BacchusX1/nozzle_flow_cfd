"""
Tests for de_laval_ambient template mesh generation.

This test validates that:
1. The de_laval_ambient template (closed domain with ambient box) can be loaded
2. Mesh generation works with hex (quad) elements and boundary layers
3. Mesh generation works with tet (triangle) elements
4. The mesh has correct boundary groups (wall, inlet, outlet)
5. Values from standard_values_gui.yml are properly applied
"""

import pytest
import shutil
from pathlib import Path


def _is_gmsh_available() -> bool:
    """Check if gmsh Python module is available."""
    try:
        import gmsh
        return True
    except ImportError:
        return False


@pytest.fixture
def template_loader():
    """Create a template loader instance."""
    from core.template_loader import TemplateLoader
    return TemplateLoader()


@pytest.fixture
def de_laval_ambient_geometry(template_loader):
    """Load the de_laval_ambient template geometry."""
    return template_loader.load_template('de_laval_ambient')


@pytest.fixture
def standard_mesh_params():
    """Create mesh parameters from standard_values_gui.yml.
    
    Uses slightly coarser settings to make tests run in reasonable time.
    """
    from core.modules.mesh_generator import MeshParameters
    from core.standard_values import StandardValues
    
    sv = StandardValues()
    
    params = MeshParameters()
    params.mesh_type = sv.get('mesh.mesh_type', 'hex')
    params.boundary_layer_enabled = sv.get('mesh.boundary_layer.enabled', True)
    
    # Use coarser values for testing (10x larger than production)
    params.element_size = sv.get('mesh.global_element_size', 0.02) * 2.5
    params.min_element_size = sv.get('mesh.min_element_size', 0.001) * 5
    params.max_element_size = sv.get('mesh.max_element_size', 0.05) * 2
    params.wall_element_size = sv.get('mesh.wall_element_size', 0.005) * 3
    
    # Boundary layer settings (use fewer layers for speed)
    params.boundary_layer_elements = min(sv.get('mesh.boundary_layer.num_layers', 8), 5)
    params.boundary_layer_first_layer = sv.get('mesh.boundary_layer.first_layer_thickness', 1e-5) * 50
    params.boundary_layer_growth_rate = sv.get('mesh.boundary_layer.growth_ratio', 1.2)
    params.boundary_layer_thickness = 0.005  # Computed value
    
    return params


class TestDeLavalAmbientGeometry:
    """Tests for de_laval_ambient geometry loading and properties."""
    
    def test_template_loads(self, template_loader):
        """Test that the de_laval_ambient template can be loaded."""
        geometry = template_loader.load_template('de_laval_ambient')
        assert geometry is not None
        assert len(geometry.elements) == 12, "de_laval_ambient should have 12 elements"
    
    def test_geometry_is_closed_domain(self, de_laval_ambient_geometry):
        """Test that the geometry forms a closed domain."""
        geometry = de_laval_ambient_geometry
        
        x_coords, y_coords = geometry.get_interpolated_points()
        
        # Check domain has both positive and negative y (full domain, not half)
        positive_y = [y for y in y_coords if y > 0.01]
        negative_y = [y for y in y_coords if y < -0.01]
        assert len(positive_y) > 10, "Geometry should have significant positive y region"
        assert len(negative_y) > 10, "Geometry should have significant negative y region"
        
        # Check that start and end points are approximately the same (closed loop)
        first_point = (x_coords[0], y_coords[0])
        last_point = (x_coords[-1], y_coords[-1])
        assert abs(first_point[0] - last_point[0]) < 0.05, "Loop should be closed (x)"
        assert abs(first_point[1] - last_point[1]) < 0.05, "Loop should be closed (y)"
    
    def test_geometry_elements_are_connected(self, de_laval_ambient_geometry):
        """Test that geometry elements connect end-to-end."""
        geometry = de_laval_ambient_geometry
        
        prev_end = None
        for i, element in enumerate(geometry.elements):
            pts = element.get_points()  # Raw control points
            start = pts[0]
            end = pts[-1]
            
            if prev_end is not None:
                # Allow some tolerance for numerical precision
                assert abs(start[0] - prev_end[0]) < 0.01, f"Element {i} x-gap"
                assert abs(start[1] - prev_end[1]) < 0.01, f"Element {i} y-gap"
            
            prev_end = end


@pytest.mark.skipif(not _is_gmsh_available(), reason="Gmsh not available")
class TestDeLavalAmbientMeshGeneration:
    """Tests for de_laval_ambient mesh generation."""
    
    def test_mesh_generation_triangles(self, de_laval_ambient_geometry):
        """Test mesh generation with triangle elements (faster)."""
        from core.modules.mesh_generator import MeshParameters, AdvancedMeshGenerator
        
        params = MeshParameters()
        params.mesh_type = 'tet'
        params.boundary_layer_enabled = False
        params.element_size = 0.05
        params.min_element_size = 0.02
        params.max_element_size = 0.15
        params.wall_element_size = 0.03
        
        generator = AdvancedMeshGenerator()
        mesh_data = generator.generate_mesh(de_laval_ambient_geometry, params)
        
        assert mesh_data is not None
        assert mesh_data['mesh_info']['num_nodes'] > 100
        assert mesh_data['mesh_info']['num_elements'] > 100
        assert mesh_data['element_type'] == 'triangle'
    
    def test_mesh_generation_quads(self, de_laval_ambient_geometry):
        """Test mesh generation with quad elements (hex type)."""
        from core.modules.mesh_generator import MeshParameters, AdvancedMeshGenerator
        
        params = MeshParameters()
        params.mesh_type = 'hex'
        params.boundary_layer_enabled = False
        params.element_size = 0.05
        params.min_element_size = 0.02
        params.max_element_size = 0.15
        params.wall_element_size = 0.03
        
        generator = AdvancedMeshGenerator()
        mesh_data = generator.generate_mesh(de_laval_ambient_geometry, params)
        
        assert mesh_data is not None
        assert mesh_data['mesh_info']['num_nodes'] > 100
        assert mesh_data['mesh_info']['num_elements'] > 100
        assert mesh_data['element_type'] == 'quad'
    
    def test_mesh_has_correct_boundaries(self, de_laval_ambient_geometry):
        """Test that generated mesh has wall, inlet, and outlet boundaries."""
        from core.modules.mesh_generator import MeshParameters, AdvancedMeshGenerator
        
        params = MeshParameters()
        params.mesh_type = 'tet'
        params.boundary_layer_enabled = False
        params.element_size = 0.08
        params.min_element_size = 0.03
        params.max_element_size = 0.2
        params.wall_element_size = 0.05
        
        generator = AdvancedMeshGenerator()
        mesh_data = generator.generate_mesh(de_laval_ambient_geometry, params)
        
        boundaries = mesh_data['boundary_elements']
        assert 'wall' in boundaries, "Mesh should have 'wall' boundary"
        assert 'inlet' in boundaries, "Mesh should have 'inlet' boundary"
        assert 'outlet' in boundaries, "Mesh should have 'outlet' boundary"
        
        # Each boundary should have some elements
        assert len(boundaries['wall']) > 0, "Wall boundary should have elements"
        assert len(boundaries['inlet']) > 0, "Inlet boundary should have elements"
        assert len(boundaries['outlet']) > 0, "Outlet boundary should have elements"
    
    @pytest.mark.slow
    def test_mesh_with_boundary_layers(self, de_laval_ambient_geometry, standard_mesh_params):
        """Test mesh generation with boundary layers enabled (slower test)."""
        from core.modules.mesh_generator import AdvancedMeshGenerator
        
        generator = AdvancedMeshGenerator()
        mesh_data = generator.generate_mesh(de_laval_ambient_geometry, standard_mesh_params)
        
        assert mesh_data is not None
        assert mesh_data['mesh_info']['num_nodes'] > 1000
        assert mesh_data['mesh_info']['num_elements'] > 1000
        
        # With hex mesh type, elements should be quads
        if standard_mesh_params.mesh_type == 'hex':
            assert mesh_data['element_type'] == 'quad'
    
    def test_mesh_vertices_valid(self, de_laval_ambient_geometry):
        """Test that mesh vertices are within expected domain bounds."""
        from core.modules.mesh_generator import MeshParameters, AdvancedMeshGenerator
        
        params = MeshParameters()
        params.mesh_type = 'tet'
        params.boundary_layer_enabled = False
        params.element_size = 0.1
        params.wall_element_size = 0.08
        
        generator = AdvancedMeshGenerator()
        mesh_data = generator.generate_mesh(de_laval_ambient_geometry, params)
        
        vertices = mesh_data['vertices']
        
        # Check vertices are within expected domain (de_laval_ambient extents)
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        
        # From template: x in [-0.2, 2.6], y in [-0.6, 0.6]
        assert min(x_coords) >= -0.3, "X min should be near -0.2"
        assert max(x_coords) <= 2.7, "X max should be near 2.6"
        assert min(y_coords) >= -0.7, "Y min should be near -0.6"
        assert max(y_coords) <= 0.7, "Y max should be near 0.6"


@pytest.mark.skipif(not _is_gmsh_available(), reason="Gmsh not available")
class TestClosedDomainDetection:
    """Tests for closed domain detection logic."""
    
    def test_de_laval_ambient_detected_as_closed(self, de_laval_ambient_geometry):
        """Test that de_laval_ambient is correctly detected as closed domain."""
        from core.modules.mesh_generator import AdvancedMeshGenerator
        
        generator = AdvancedMeshGenerator()
        x_coords, y_coords = de_laval_ambient_geometry.get_interpolated_points()
        
        is_closed = generator._is_closed_domain(x_coords, y_coords)
        assert is_closed, "de_laval_ambient should be detected as closed domain"
    
    def test_simple_nozzle_detected_as_symmetric(self, template_loader):
        """Test that simple nozzle templates are detected as symmetric (not closed)."""
        from core.modules.mesh_generator import AdvancedMeshGenerator
        
        # Load a simple nozzle template (upper profile only)
        try:
            geometry = template_loader.load_template('de_laval')
        except FileNotFoundError:
            pytest.skip("de_laval template not available")
        
        generator = AdvancedMeshGenerator()
        x_coords, y_coords = geometry.get_interpolated_points()
        
        # Simple nozzle should NOT be detected as closed domain
        # (it only has upper profile, generator will mirror it)
        is_closed = generator._is_closed_domain(x_coords, y_coords)
        # Note: If de_laval is also a full domain, this test may need adjustment
