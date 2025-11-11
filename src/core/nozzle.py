import numpy as np
import json
import os
from typing import List, Tuple

from core.geometry import GeometryElement, PolynomialElement, LineElement, ArcElement

class NozzleGeometry:
    """Class to handle nozzle geometry creation and manipulation."""
    
    def __init__(self):
        self.elements: List[GeometryElement] = []
        self.is_symmetric = True
        
    def add_element(self, element: GeometryElement):
        """Add a geometry element."""
        self.elements.append(element)
            
    def remove_last_element(self):
        """Remove the last added element."""
        if self.elements:
            self.elements.pop()
            
    def clear(self):
        """Clear all geometry elements."""
        self.elements = []
    
    def set_symmetric(self, is_symmetric: bool):
        """Set symmetry flag."""
        self.is_symmetric = is_symmetric
        
    def get_interpolated_points(self, num_points_per_element: int = 50) -> Tuple[List[float], List[float]]:
        """Get interpolated points as separate x,y coordinate lists."""
        if not self.elements:
            return [], []
        
        all_points = []
        for element in self.elements:
            element_points = element.get_interpolated_points(num_points_per_element)
            if all_points and element_points:
                # Avoid duplicating connection points
                if np.allclose(all_points[-1], element_points[0]):
                    element_points = element_points[1:]
            all_points.extend(element_points)
        
        if not all_points:
            return [], []
            
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        return x_coords, y_coords
    
    def get_all_interpolated_points(self, num_points_per_element: int = 50) -> List[Tuple[float, float]]:
        """Get all interpolated points as a list of (x,y) tuples."""
        if not self.elements:
            return []
        
        all_points = []
        for element in self.elements:
            element_points = element.get_interpolated_points(num_points_per_element)
            if all_points and element_points:
                # Avoid duplicating connection points
                if np.allclose(all_points[-1], element_points[0]):
                    element_points = element_points[1:]
            all_points.extend(element_points)
        
        return all_points
        
    def save_to_file(self, filename: str):
        """Save geometry elements to a JSON file."""
        data = {
            'is_symmetric': self.is_symmetric,
            'elements': []
        }
        
        for element in self.elements:
            element_data = {
                'type': element.element_type,
                'points': element.get_points()
            }
            if isinstance(element, PolynomialElement):
                element_data['degree'] = element.degree
            data['elements'].append(element_data)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str):
        """Load geometry elements from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.is_symmetric = data.get('is_symmetric', True)
        self.elements = []
        
        for element_data in data.get('elements', []):
            element_type = element_data.get('type')
            points = [tuple(p) for p in element_data.get('points', [])]

            if not points:
                continue

            if element_type == 'polynomial':
                degree = element_data.get('degree')
                self.elements.append(PolynomialElement(points, degree))
            elif element_type == 'line':
                self.elements.append(LineElement(points[0], points[1]))
            elif element_type == 'arc':
                self.elements.append(ArcElement(points))
    
    def to_dict(self):
        """Convert geometry to dictionary format."""
        return {
            'is_symmetric': self.is_symmetric,
            'elements': [
                {
                    'type': element.element_type,
                    'points': element.get_points(),
                    'degree': getattr(element, 'degree', None)
                }
                for element in self.elements
            ]
        }
    
    def load_from_dict(self, data):
        """Load geometry from dictionary format."""
        self.is_symmetric = data.get('is_symmetric', True)
        self.elements = []
        
        for element_data in data.get('elements', []):
            element_type = element_data.get('type')
            points = [tuple(p) for p in element_data.get('points', [])]

            if not points:
                continue

            if element_type == 'polynomial':
                degree = element_data.get('degree')
                self.elements.append(PolynomialElement(points, degree))
            elif element_type == 'line':
                self.elements.append(LineElement(points[0], points[1]))
            elif element_type == 'arc':
                self.elements.append(ArcElement(points))
