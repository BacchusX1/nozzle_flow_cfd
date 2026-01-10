"""
Tests for validating generated Laval nozzle meshes for SU2 compatibility.

This test suite validates that meshes generated from the de_laval template
are valid for SU2 simulations - specifically checking for:
1. No orphan nodes (NPOIN matches elements)
2. Valid boundary markers
3. Correct mesh file format

The NPOIN mismatch error in SU2 occurs when:
- NPOIN declares N nodes but elements only reference M < N nodes
- This happens when the mesh has orphan/unreferenced nodes
"""
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest


def _is_su2_available() -> bool:
    """Check if SU2_CFD is available on PATH."""
    return shutil.which("SU2_CFD") is not None


def _is_gmsh_available() -> bool:
    """Check if gmsh Python module is available."""
    try:
        import gmsh
        return True
    except ImportError:
        return False


def _parse_su2_mesh(mesh_path: Path) -> Dict:
    """Parse SU2 mesh file and extract statistics."""
    content = mesh_path.read_text()
    lines = content.strip().split('\n')
    
    result = {
        'ndim': 2,
        'npoin_declared': 0,
        'npoin_actual': 0,
        'nelem_declared': 0,
        'nelem_actual': 0,
        'node_coords': {},
        'elements': [],
        'nodes_in_elements': set(),
        'markers': {},
    }
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('NDIME='):
            result['ndim'] = int(line.split('=')[1].strip())
            i += 1
            
        elif line.startswith('NPOIN='):
            result['npoin_declared'] = int(line.split('=')[1].strip().split()[0])
            i += 1
            # Read all points
            for _ in range(result['npoin_declared']):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                if len(parts) >= 3:
                    idx = int(parts[-1])
                    x, y = float(parts[0]), float(parts[1])
                    result['node_coords'][idx] = (x, y)
                    result['npoin_actual'] += 1
                i += 1
                
        elif line.startswith('NELEM='):
            result['nelem_declared'] = int(line.split('=')[1].strip())
            i += 1
            # Read all elements
            for _ in range(result['nelem_declared']):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    elem_type = int(parts[0])
                    # Parse node indices based on element type
                    if elem_type == 5:  # Triangle
                        nodes = [int(parts[j]) for j in range(1, 4)]
                    elif elem_type == 9:  # Quad
                        nodes = [int(parts[j]) for j in range(1, 5)]
                    else:
                        nodes = [int(parts[j]) for j in range(1, len(parts) - 1)]
                    result['elements'].append((elem_type, nodes))
                    result['nodes_in_elements'].update(nodes)
                    result['nelem_actual'] += 1
                i += 1
                
        elif line.startswith('MARKER_TAG='):
            marker_name = line.split('=')[1].strip()
            i += 1
            if i < len(lines) and lines[i].strip().startswith('MARKER_ELEMS='):
                n_elems = int(lines[i].split('=')[1].strip())
                i += 1
                marker_nodes = set()
                for _ in range(n_elems):
                    if i >= len(lines):
                        break
                    parts = lines[i].strip().split()
                    # Line elements have format: 3 n1 n2
                    if len(parts) >= 3:
                        marker_nodes.add(int(parts[1]))
                        marker_nodes.add(int(parts[2]))
                    i += 1
                result['markers'][marker_name] = marker_nodes
        else:
            i += 1
            
    return result


def _validate_mesh_integrity(mesh_data: Dict) -> Tuple[bool, List[str]]:
    """Validate mesh integrity for SU2 compatibility.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check 1: NPOIN declared matches actual points read
    if mesh_data['npoin_declared'] != mesh_data['npoin_actual']:
        errors.append(
            f"NPOIN declared ({mesh_data['npoin_declared']}) != "
            f"actual points read ({mesh_data['npoin_actual']})"
        )
    
    # Check 2: NELEM declared matches actual elements read
    if mesh_data['nelem_declared'] != mesh_data['nelem_actual']:
        errors.append(
            f"NELEM declared ({mesh_data['nelem_declared']}) != "
            f"actual elements read ({mesh_data['nelem_actual']})"
        )
    
    # Check 3: All declared nodes are referenced by elements
    declared_nodes = set(mesh_data['node_coords'].keys())
    referenced_nodes = mesh_data['nodes_in_elements']
    
    orphan_nodes = declared_nodes - referenced_nodes
    if orphan_nodes:
        errors.append(
            f"Found {len(orphan_nodes)} orphan nodes not referenced by any element: "
            f"{sorted(orphan_nodes)[:10]}..."
        )
    
    # Check 4: All element node references are valid
    invalid_refs = referenced_nodes - declared_nodes
    if invalid_refs:
        errors.append(
            f"Elements reference {len(invalid_refs)} undefined nodes: "
            f"{sorted(invalid_refs)[:10]}..."
        )
    
    # Check 5: Node indices should be consecutive from 0 to NPOIN-1
    expected_indices = set(range(mesh_data['npoin_declared']))
    actual_indices = declared_nodes
    missing_indices = expected_indices - actual_indices
    extra_indices = actual_indices - expected_indices
    
    if missing_indices:
        errors.append(
            f"Missing node indices: {sorted(missing_indices)[:10]}..."
        )
    if extra_indices:
        errors.append(
            f"Unexpected node indices: {sorted(extra_indices)[:10]}..."
        )
    
    # Check 6: Boundary markers reference valid nodes
    for marker_name, marker_nodes in mesh_data['markers'].items():
        invalid_marker_refs = marker_nodes - declared_nodes
        if invalid_marker_refs:
            errors.append(
                f"Marker '{marker_name}' references undefined nodes: "
                f"{sorted(invalid_marker_refs)[:10]}..."
            )
    
    return len(errors) == 0, errors


@pytest.mark.skipif(not _is_gmsh_available(), reason="Gmsh not available")
class TestLavalNozzleMeshGeneration:
    """Test mesh generation from de Laval nozzle template."""
    
    def test_generate_mesh_from_de_laval_template(self, tmp_path: Path):
        """Generate a mesh from de Laval template and validate integrity."""
        from src.core.template_loader import TemplateLoader
        from src.core.modules.mesh_generator import AdvancedMeshGenerator, MeshParameters
        from src.core.su2_mesh_converter import SU2MeshConverter
        
        # Load de Laval template - returns a NozzleGeometry object
        loader = TemplateLoader()
        geometry = loader.load_template("de_laval")
        
        # Generate mesh with moderate refinement
        mesh_gen = AdvancedMeshGenerator()
        params = MeshParameters()
        params.element_size = 0.02  # Moderate refinement
        params.boundary_layer_enabled = False  # Disable for simpler test
        
        mesh_data = mesh_gen.generate_mesh(geometry, params)
        
        # Export to SU2 format
        converter = SU2MeshConverter()
        converter.load_from_mesh_data(mesh_data)
        converter.detect_boundaries_from_geometry(geometry)
        
        mesh_path = tmp_path / "mesh.su2"
        converter.write_su2_mesh(str(mesh_path))
        
        # Validate the generated mesh
        parsed = _parse_su2_mesh(mesh_path)
        is_valid, errors = _validate_mesh_integrity(parsed)
        
        # Print diagnostics
        print(f"\nMesh statistics:")
        print(f"  NPOIN declared: {parsed['npoin_declared']}")
        print(f"  NPOIN actual: {parsed['npoin_actual']}")
        print(f"  NELEM declared: {parsed['nelem_declared']}")
        print(f"  NELEM actual: {parsed['nelem_actual']}")
        print(f"  Nodes in elements: {len(parsed['nodes_in_elements'])}")
        print(f"  Markers: {list(parsed['markers'].keys())}")
        
        if errors:
            print("\nErrors found:")
            for err in errors:
                print(f"  - {err}")
        
        assert is_valid, f"Mesh validation failed:\n" + "\n".join(errors)
    
    def test_mesh_node_count_matches_elements(self, tmp_path: Path):
        """Ensure all declared nodes are referenced by elements (no orphans)."""
        from src.core.template_loader import TemplateLoader
        from src.core.modules.mesh_generator import AdvancedMeshGenerator, MeshParameters
        from src.core.su2_mesh_converter import SU2MeshConverter
        
        # Load de Laval template - returns a NozzleGeometry object
        loader = TemplateLoader()
        geometry = loader.load_template("de_laval")
        
        # Generate mesh
        mesh_gen = AdvancedMeshGenerator()
        params = MeshParameters()
        params.element_size = 0.05  # Coarser for faster test
        params.boundary_layer_enabled = False
        
        mesh_data = mesh_gen.generate_mesh(geometry, params)
        
        # Export to SU2 format
        converter = SU2MeshConverter()
        converter.load_from_mesh_data(mesh_data)
        converter.detect_boundaries_from_geometry(geometry)
        
        mesh_path = tmp_path / "mesh.su2"
        converter.write_su2_mesh(str(mesh_path))
        
        # Parse and validate
        parsed = _parse_su2_mesh(mesh_path)
        
        orphan_count = len(parsed['node_coords'].keys() - parsed['nodes_in_elements'])
        
        assert orphan_count == 0, \
            f"Mesh has {orphan_count} orphan nodes. " \
            f"NPOIN={parsed['npoin_declared']}, nodes_in_elements={len(parsed['nodes_in_elements'])}"


@pytest.mark.skipif(not _is_su2_available(), reason="SU2_CFD not available")
@pytest.mark.skipif(not _is_gmsh_available(), reason="Gmsh not available")
class TestSU2MeshCompatibility:
    """Test that generated meshes work with SU2 solver."""
    
    def test_su2_accepts_generated_mesh_serial(self, tmp_path: Path):
        """Test that SU2 can read and validate the generated mesh (serial mode)."""
        from src.core.template_loader import TemplateLoader
        from src.core.modules.mesh_generator import AdvancedMeshGenerator, MeshParameters
        from src.core.su2_mesh_converter import SU2MeshConverter
        from src.core.su2_runner import SU2Runner
        from src.core.modules.simulation_setup import (
            SimulationSetup, SolverType, BoundaryType, BoundaryCondition
        )
        
        # Load de Laval template - returns NozzleGeometry object
        loader = TemplateLoader()
        geometry = loader.load_template("de_laval")
        
        # Generate mesh
        mesh_gen = AdvancedMeshGenerator()
        params = MeshParameters()
        params.element_size = 0.03
        params.boundary_layer_enabled = False
        
        mesh_data = mesh_gen.generate_mesh(geometry, params)
        
        # Export to SU2 format
        converter = SU2MeshConverter()
        converter.load_from_mesh_data(mesh_data)
        converter.detect_boundaries_from_geometry(geometry)
        
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        mesh_path = case_dir / "mesh.su2"
        converter.write_su2_mesh(str(mesh_path))
        
        # First validate mesh integrity ourselves
        parsed = _parse_su2_mesh(mesh_path)
        is_valid, errors = _validate_mesh_integrity(parsed)
        assert is_valid, f"Mesh pre-validation failed:\n" + "\n".join(errors)
        
        # Create minimal SU2 config
        sim = SimulationSetup()
        sim.solver_settings.solver_type = SolverType.EULER
        sim.solver_settings.max_iterations = 1
        
        # Add boundary conditions matching detected boundaries
        for marker_name in parsed['markers'].keys():
            if marker_name == 'inlet':
                sim.add_boundary_condition(BoundaryCondition(
                    name="inlet",
                    boundary_type=BoundaryType.INLET,
                    total_pressure=300000,
                    total_temperature=300
                ))
            elif marker_name == 'outlet':
                sim.add_boundary_condition(BoundaryCondition(
                    name="outlet",
                    boundary_type=BoundaryType.OUTLET,
                    static_pressure=101325
                ))
            elif marker_name in ('wall', 'symmetry'):
                sim.add_boundary_condition(BoundaryCondition(
                    name=marker_name,
                    boundary_type=BoundaryType.WALL if marker_name == 'wall' else BoundaryType.SYMMETRY
                ))
        
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        # Run SU2 in serial mode (1 processor)
        runner = SU2Runner(str(case_dir))
        success = runner.run_solver(n_processors=1)
        
        log_path = case_dir / "log.SU2_CFD"
        log_content = log_path.read_text() if log_path.exists() else "No log file"
        
        # Check for the specific NPOIN mismatch error
        assert "Mismatch between NPOIN" not in log_content, \
            f"SU2 reported NPOIN mismatch error:\n{log_content}"
        
        # Check for successful mesh reading
        assert "grid points" in log_content.lower() or success, \
            f"SU2 failed to read mesh properly. Log:\n{log_content[:2000]}"
    
    def test_su2_accepts_generated_mesh_parallel(self, tmp_path: Path):
        """Test that SU2 can partition and run the mesh in parallel mode."""
        if not shutil.which("mpirun"):
            pytest.skip("mpirun not available")
        
        from src.core.template_loader import TemplateLoader
        from src.core.modules.mesh_generator import AdvancedMeshGenerator, MeshParameters
        from src.core.su2_mesh_converter import SU2MeshConverter
        from src.core.su2_runner import SU2Runner
        from src.core.modules.simulation_setup import (
            SimulationSetup, SolverType, BoundaryType, BoundaryCondition
        )
        
        # Load de Laval template - returns NozzleGeometry object
        loader = TemplateLoader()
        geometry = loader.load_template("de_laval")
        
        # Generate mesh (need enough elements for partitioning)
        mesh_gen = AdvancedMeshGenerator()
        params = MeshParameters()
        params.element_size = 0.02  # Finer mesh for partitioning
        params.boundary_layer_enabled = False
        
        mesh_data = mesh_gen.generate_mesh(geometry, params)
        
        # Export to SU2 format
        converter = SU2MeshConverter()
        converter.load_from_mesh_data(mesh_data)
        converter.detect_boundaries_from_geometry(geometry)
        
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        mesh_path = case_dir / "mesh.su2"
        converter.write_su2_mesh(str(mesh_path))
        
        # Validate mesh integrity
        parsed = _parse_su2_mesh(mesh_path)
        is_valid, errors = _validate_mesh_integrity(parsed)
        assert is_valid, f"Mesh validation failed:\n" + "\n".join(errors)
        
        print(f"\nMesh for parallel test: {parsed['npoin_declared']} nodes, "
              f"{parsed['nelem_declared']} elements")
        
        # Create minimal SU2 config
        sim = SimulationSetup()
        sim.solver_settings.solver_type = SolverType.EULER
        sim.solver_settings.max_iterations = 1
        
        for marker_name in parsed['markers'].keys():
            if marker_name == 'inlet':
                sim.add_boundary_condition(BoundaryCondition(
                    name="inlet",
                    boundary_type=BoundaryType.INLET,
                    total_pressure=300000,
                    total_temperature=300
                ))
            elif marker_name == 'outlet':
                sim.add_boundary_condition(BoundaryCondition(
                    name="outlet",
                    boundary_type=BoundaryType.OUTLET,
                    static_pressure=101325
                ))
            elif marker_name in ('wall', 'symmetry'):
                sim.add_boundary_condition(BoundaryCondition(
                    name=marker_name,
                    boundary_type=BoundaryType.WALL if marker_name == 'wall' else BoundaryType.SYMMETRY
                ))
        
        sim.generate_su2_case(str(case_dir), mesh_data=None)
        
        # Run SU2 with 2 processors
        runner = SU2Runner(str(case_dir))
        success = runner.run_solver(n_processors=2)
        
        log_path = case_dir / "log.SU2_CFD"
        log_content = log_path.read_text() if log_path.exists() else "No log file"
        
        # Check for the specific NPOIN mismatch error
        assert "Mismatch between NPOIN" not in log_content, \
            f"SU2 reported NPOIN mismatch error:\n{log_content}"
        
        # Check for segfault
        assert "Segmentation fault" not in log_content and "segfault" not in log_content.lower(), \
            f"Segfault occurred:\n{log_content}"
        
        # The test passes if SU2 at least started successfully
        assert "grid points" in log_content.lower() or success, \
            f"SU2 parallel run failed. Log:\n{log_content[:3000]}"
