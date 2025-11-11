import numpy as np
import math
from scipy.interpolate import UnivariateSpline
from typing import List, Tuple

class GeometryElement:
    """Base class for geometry elements."""
    
    def __init__(self, element_type: str):
        self.element_type = element_type
        
    def get_points(self) -> List[Tuple[float, float]]:
        """Get points that define this element."""
        raise NotImplementedError
        
    def get_interpolated_points(self, num_points: int = 50) -> List[Tuple[float, float]]:
        """Get interpolated points for smooth representation."""
        raise NotImplementedError

    def get_length(self) -> float:
        """Get the length of the element."""
        points = self.get_interpolated_points()
        if len(points) < 2:
            return 0.0
        return np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))


class PolynomialElement(GeometryElement):
    """Polynomial curve element defined by control points."""
    
    def __init__(self, points: List[Tuple[float, float]], degree: int = None):
        super().__init__("polynomial")
        self.points = points
        # Auto-determine degree if not specified (max 3, min 1)
        if degree is None:
            self.degree = min(len(points) - 1, 3) if len(points) > 1 else 1
        else:
            self.degree = max(1, min(degree, len(points) - 1)) if len(points) > 1 else 1
        
    def get_points(self) -> List[Tuple[float, float]]:
        return self.points
        
    def get_interpolated_points(self, num_points: int = 50) -> List[Tuple[float, float]]:
        if len(self.points) < 2:
            return self.points
        elif len(self.points) == 2:
            # Linear interpolation for 2 points
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            t = np.linspace(0, 1, num_points)
            x_interp = x1 + t * (x2 - x1)
            y_interp = y1 + t * (y2 - y1)
            return list(zip(x_interp, y_interp))
        else:
            # Polynomial fitting for multiple points
            try:
                # Sort points by x-coordinate
                sorted_points = sorted(self.points, key=lambda p: p[0])
                x_coords = np.array([p[0] for p in sorted_points])
                y_coords = np.array([p[1] for p in sorted_points])
                
                # Fit polynomial
                poly_coeffs = np.polyfit(x_coords, y_coords, self.degree)
                poly_func = np.poly1d(poly_coeffs)
                
                # Generate smooth curve
                x_min, x_max = x_coords[0], x_coords[-1]
                x_smooth = np.linspace(x_min, x_max, num_points)
                y_smooth = poly_func(x_smooth)
                
                return list(zip(x_smooth, y_smooth))
            except (np.RankWarning, np.linalg.LinAlgError):
                # Fallback to spline interpolation
                try:
                    sorted_points = sorted(self.points, key=lambda p: p[0])
                    x_coords = np.array([p[0] for p in sorted_points])
                    y_coords = np.array([p[1] for p in sorted_points])
                    
                    spline = UnivariateSpline(x_coords, y_coords, s=0)
                    x_smooth = np.linspace(x_coords[0], x_coords[-1], num_points)
                    y_smooth = spline(x_smooth)
                    
                    return list(zip(x_smooth, y_smooth))
                except Exception:
                    # Final fallback to linear interpolation
                    return self._linear_interpolation(num_points)
            except Exception:
                # Final fallback to linear interpolation
                return self._linear_interpolation(num_points)
    
    def _linear_interpolation(self, num_points: int) -> List[Tuple[float, float]]:
        """Fallback linear interpolation between points."""
        if len(self.points) < 2:
            return self.points
        
        # Simple linear interpolation between first and last point
        x1, y1 = self.points[0]
        x2, y2 = self.points[-1]
        t = np.linspace(0, 1, num_points)
        x_interp = x1 + t * (x2 - x1)
        y_interp = y1 + t * (y2 - y1)
        return list(zip(x_interp, y_interp))


class LineElement(GeometryElement):
    """Simple line element between two points."""
    
    def __init__(self, start: Tuple[float, float], end: Tuple[float, float]):
        super().__init__("line")
        self.start = start
        self.end = end
        
    def get_points(self) -> List[Tuple[float, float]]:
        return [self.start, self.end]
        
    def get_interpolated_points(self, num_points: int = 2) -> List[Tuple[float, float]]:
        x1, y1 = self.start
        x2, y2 = self.end
        t = np.linspace(0, 1, num_points)
        x_interp = x1 + t * (x2 - x1)
        y_interp = y1 + t * (y2 - y1)
        return list(zip(x_interp, y_interp))


class ArcElement(GeometryElement):
    """Arc element defined by center, radius, start/end angles or by three points."""
    
    def __init__(self, points=None, center=None, radius=None, start_angle=None, end_angle=None):
        super().__init__("arc")
        
        if points is not None:
            # Create from three points
            if len(points) != 3:
                raise ValueError("ArcElement requires exactly three points.")
            self.points = points
            self.center, self.radius, self.start_angle, self.end_angle = self._arc_from_three_points(points)
        elif all(param is not None for param in [center, radius, start_angle, end_angle]):
            # Create from parameters
            self.center = center
            self.radius = radius
            self.start_angle = start_angle
            self.end_angle = end_angle
            # Generate three points from parameters for consistency
            self.points = self._generate_points_from_parameters()
        else:
            raise ValueError("ArcElement requires either 'points' or all of 'center', 'radius', 'start_angle', 'end_angle'")
    
    def _generate_points_from_parameters(self):
        """Generate three points from center, radius, and angles."""
        cx, cy = self.center
        
        # Start point
        x1 = cx + self.radius * math.cos(self.start_angle)
        y1 = cy + self.radius * math.sin(self.start_angle)
        
        # Middle point (at average angle)
        mid_angle = (self.start_angle + self.end_angle) / 2
        x2 = cx + self.radius * math.cos(mid_angle)
        y2 = cy + self.radius * math.sin(mid_angle)
        
        # End point
        x3 = cx + self.radius * math.cos(self.end_angle)
        y3 = cy + self.radius * math.sin(self.end_angle)
        
        return [(x1, y1), (x2, y2), (x3, y3)]

    @staticmethod
    def _arc_from_three_points(points):
        """Calculate arc parameters from three points."""
        p1, p2, p3 = points
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Calculate center using perpendicular bisectors
        # If points are collinear, create a simple approximation
        try:
            # Calculate the center of the circle passing through three points
            d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            if abs(d) < 1e-10:  # Points are nearly collinear
                # Create a simple arc approximation
                center = ((x1 + x3) / 2, (y1 + y3) / 2)
                radius = ((x3 - x1)**2 + (y3 - y1)**2)**0.5 / 2
                start_angle = math.atan2(y1 - center[1], x1 - center[0])
                end_angle = math.atan2(y3 - center[1], x3 - center[0])
            else:
                ux = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / d
                uy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / d
                center = (ux, uy)
                
                radius = ((x1 - ux)**2 + (y1 - uy)**2)**0.5
                start_angle = math.atan2(y1 - uy, x1 - ux)
                end_angle = math.atan2(y3 - uy, x3 - uy)
            
            return center, radius, start_angle, end_angle
            
        except Exception:
            # Fallback: create simple arc
            center = (x2, y2)  # Use middle point as center
            radius = max(0.01, min(((x2-x1)**2 + (y2-y1)**2)**0.5, ((x3-x2)**2 + (y3-y2)**2)**0.5))
            start_angle = math.atan2(y1 - y2, x1 - x2)
            end_angle = math.atan2(y3 - y2, x3 - x2)
            return center, radius, start_angle, end_angle

    def get_points(self) -> List[Tuple[float, float]]:
        return self.points

    def get_interpolated_points(self, num_points: int = 50) -> List[Tuple[float, float]]:
        # Ensure the angles are ordered correctly
        start_angle, end_angle = self.start_angle, self.end_angle
        if start_angle > end_angle:
            # This can happen depending on point order, simple swap is not always right
            # A better way is to check the middle point
            mid_angle = math.atan2(self.points[1][1] - self.center[1], self.points[1][0] - self.center[0])
            if not (start_angle > mid_angle > end_angle):
                 if start_angle < mid_angle < (end_angle + 2*np.pi) or (start_angle-2*np.pi) < mid_angle < end_angle:
                     pass # it is ok
                 else:
                    start_angle, end_angle = end_angle, start_angle

        angles = np.linspace(start_angle, end_angle, num_points)
        x_coords = self.center[0] + self.radius * np.cos(angles)
        y_coords = self.center[1] + self.radius * np.sin(angles)
        return list(zip(x_coords, y_coords))
