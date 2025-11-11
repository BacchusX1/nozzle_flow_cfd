#!/usr/bin/env python3
"""
Template loader utility for nozzle geometry templates.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class TemplateLoader:
    """Load and manage nozzle geometry templates."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        if templates_dir is None:
            # Default to geometry/templates relative to project root
            # Go up from src/core/ to project root, then down to geometry/templates
            project_root = Path(__file__).parent.parent.parent
            self.templates_dir = project_root / "geometry" / "templates"
        else:
            self.templates_dir = Path(templates_dir)
    
    def list_templates(self) -> List[str]:
        """List available template names."""
        if not self.templates_dir.exists():
            return []
        
        templates = []
        for file_path in self.templates_dir.glob("*.json"):
            templates.append(file_path.stem)
        return sorted(templates)
    
    def get_template_path(self, template_name: str) -> Path:
        """Get the full path to a template file."""
        return self.templates_dir / f"{template_name}.json"
    
    def load_template(self, template_name: str):
        """Load and create geometry objects from template."""
        geometry, template_data = self.create_geometry_from_template(template_name)
        return geometry
    
    def load_template_data(self, template_name: str) -> Dict:
        """Load raw template data as dictionary."""
        template_path = self.get_template_path(template_name)
        if not template_path.exists():
            return {}
        
        with open(template_path, 'r') as f:
            return json.load(f)
    
    def get_template_info(self, template_name: str) -> Dict:
        """Get basic info about a template without loading full data."""
        template_data = self.load_template_data(template_name)
        return {
            "name": template_data.get("name", template_name),
            "description": template_data.get("description", ""),
            "type": template_data.get("type", "unknown"),
            "elements_count": len(template_data.get("elements", [])),
            "has_flow_conditions": "flow_conditions" in template_data,
            "has_mesh_settings": "mesh_settings" in template_data
        }
    
    def create_geometry_from_template(self, template_name: str):
        """Create geometry objects from template data."""
        from core.nozzle import NozzleGeometry
        from core.geometry import PolynomialElement, LineElement, ArcElement
        
        template_data = self.load_template_data(template_name)
        geometry = NozzleGeometry()
        
        for element_data in template_data.get("elements", []):
            element_type = element_data["type"]
            control_points = element_data["control_points"]
            
            if element_type == "PolynomialElement":
                element = PolynomialElement(control_points)
            elif element_type == "LineElement":
                element = LineElement(control_points[0], control_points[1])
            elif element_type == "ArcElement":
                element = ArcElement(points=control_points[:3])
            else:
                continue  # Skip unknown element types
            
            geometry.add_element(element)
        
        # Set geometry properties if available
        if "inlet_diameter" in template_data:
            geometry.inlet_diameter = template_data["inlet_diameter"]
        if "outlet_diameter" in template_data:
            geometry.outlet_diameter = template_data["outlet_diameter"]
        if "throat_diameter" in template_data:
            geometry.throat_diameter = template_data["throat_diameter"]
        
        return geometry, template_data


def main():
    """Test the template loader."""
    import sys
    from pathlib import Path
    
    # Add src to path for standalone testing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    loader = TemplateLoader()
    
    print("📁 Available Templates:")
    templates = loader.list_templates()
    for template in templates:
        info = loader.get_template_info(template)
        print(f"  • {info['name']} ({template})")
        print(f"    {info['description']}")
        print(f"    Elements: {info['elements_count']}, Flow: {info['has_flow_conditions']}, Mesh: {info['has_mesh_settings']}")
        print()


if __name__ == "__main__":
    main()
