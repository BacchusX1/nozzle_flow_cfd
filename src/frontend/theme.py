"""
World-Class Modern UI Theme for Nozzle CFD Design Tool

An absolutely stunning dark theme with premium styling, advanced animations,
and professional-grade visual design inspired by the best modern applications.

Configuration is loaded from configuration.yaml at the project root.
"""

import os
import yaml
from pathlib import Path


def _load_configuration():
    """Load configuration from configuration.yaml file."""
    config = {}
    
    # Look for configuration.yaml in common locations
    search_paths = [
        Path(__file__).parent.parent.parent / "configuration.yaml",  # project root
        Path.cwd() / "configuration.yaml",  # current working directory
        Path.home() / ".nozzle_cfd" / "configuration.yaml",  # user home
    ]
    
    for config_path in search_paths:
        if config_path.exists():
            # Strict loading: propagate exceptions if file exists but fails to parse
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            break
    
    return config


# Load configuration at module import time
_CONFIG = _load_configuration()


class Theme:
    """World-class dark theme with premium styling and advanced visual effects.
    
    Values can be customized via configuration.yaml at the project root.
    """
    
    # Configuration accessors for sub-sections
    _theme_cfg = _CONFIG.get('theme', {})
    _typography_cfg = _CONFIG.get('typography', {})
    _buttons_cfg = _CONFIG.get('buttons', {})
    _layout_cfg = _CONFIG.get('layout', {})
    _display_cfg = _CONFIG.get('display', {})
    
    # Core Background Colors - Modern sophisticated palette
    BACKGROUND = _theme_cfg.get('background', '#0d1117')
    SURFACE = _theme_cfg.get('surface', '#161b22')
    SURFACE_VARIANT = _theme_cfg.get('surface_variant', '#21262d')
    SURFACE_ELEVATED = _theme_cfg.get('surface_elevated', '#30363d')

    SECONDARY = '#484f58'            # Secondary surface elements
    
    # Modern Accent Colors - Configurable via configuration.yaml
    PRIMARY = _theme_cfg.get('primary', '#58a6ff')
    PRIMARY_VARIANT = _theme_cfg.get('primary_variant', '#388bfd')
    PRIMARY_LIGHT = _theme_cfg.get('primary_light', '#79c0ff')
    ACCENT = _theme_cfg.get('accent', '#f78166')
    ACCENT_VARIANT = _theme_cfg.get('accent_variant', '#e36749')
    ACCENT_LIGHT = _theme_cfg.get('accent_light', '#ffa28b')
    
    # Professional Status Colors - Configurable
    SUCCESS = _theme_cfg.get('success', '#3fb950')
    SUCCESS_VARIANT = '#2ea043'      # Deeper green
    WARNING = _theme_cfg.get('warning', '#d29922')
    WARNING_VARIANT = '#bf8700'      # Deeper warning amber
    ERROR = _theme_cfg.get('error', '#f85149')
    ERROR_VARIANT = '#da3633'        # Deeper error red
    INFO = _theme_cfg.get('info', '#58a6ff')
    INFO_VARIANT = '#388bfd'         # Deeper info blue
    
    # Modern High Contrast Text - Configurable
    TEXT_PRIMARY = _theme_cfg.get('text_primary', '#f0f6fc')
    TEXT_SECONDARY = _theme_cfg.get('text_secondary', '#8b949e')
    TEXT_TERTIARY = _theme_cfg.get('text_tertiary', '#6e7681')
    TEXT_INVERSE = '#0d1117'         # Dark text for light backgrounds
    TEXT_ACCENT = _theme_cfg.get('primary', '#58a6ff')
    TEXT_SUCCESS = _theme_cfg.get('success', '#3fb950')
    TEXT_WARNING = _theme_cfg.get('warning', '#d29922')
    TEXT_ERROR = _theme_cfg.get('error', '#f85149')
    
    # Backward compatibility
    TEXT = TEXT_PRIMARY              # Alias for old theme references
    
    # Modern Interactive Elements - Derived from primary/accent colors
    BUTTON_PRIMARY = _theme_cfg.get('primary', '#58a6ff')
    BUTTON_PRIMARY_HOVER = _theme_cfg.get('primary_light', '#79c0ff')
    BUTTON_PRIMARY_ACTIVE = _theme_cfg.get('primary_variant', '#388bfd')
    BUTTON_SECONDARY = '#30363d'     # Muted secondary buttons
    BUTTON_SECONDARY_HOVER = '#484f58' # Lighter on hover
    BUTTON_DANGER = _theme_cfg.get('error', '#f85149')
    BUTTON_DANGER_HOVER = '#ff6b6b'  # Lighter danger on hover
    BUTTON_SUCCESS = _theme_cfg.get('success', '#3fb950')
    BUTTON_SUCCESS_HOVER = '#56d364' # Lighter success on hover
    
    # Modern Borders and Dividers - Configurable
    BORDER = _theme_cfg.get('border', '#30363d')
    BORDER_SUBTLE = _theme_cfg.get('border_subtle', '#21262d')
    BORDER_ACCENT = _theme_cfg.get('border_accent', '#58a6ff')
    BORDER_SUCCESS = _theme_cfg.get('success', '#3fb950')
    BORDER_WARNING = _theme_cfg.get('warning', '#d29922')
    BORDER_ERROR = _theme_cfg.get('error', '#f85149')
    
    # Modern Input Elements - Configurable
    INPUT_BACKGROUND = _theme_cfg.get('input_background', '#0d1117')
    INPUT_BORDER = _theme_cfg.get('input_border', '#30363d')
    INPUT_BORDER_FOCUS = _theme_cfg.get('input_border_focus', '#58a6ff')
    INPUT_BORDER_ERROR = _theme_cfg.get('error', '#f85149')
    INPUT_PLACEHOLDER = '#6e7681'    # Placeholder text
    INPUT_SELECTION = 'rgba(88, 166, 255, 0.3)' # Selection highlight
    
    # Modern Visual Effects - Subtle and sophisticated
    SHADOW_SOFT = 'rgba(0, 0, 0, 0.1)'       # Soft shadows
    SHADOW_MEDIUM = 'rgba(0, 0, 0, 0.2)'     # Medium shadows
    SHADOW_STRONG = 'rgba(0, 0, 0, 0.3)'     # Strong shadows for modals
    SHADOW_GLOW = 'rgba(88, 166, 255, 0.15)' # Subtle blue glow effect
    OVERLAY = 'rgba(13, 17, 23, 0.8)'        # Modal overlay
    HIGHLIGHT = 'rgba(88, 166, 255, 0.1)'    # Selection highlight
    HIGHLIGHT_STRONG = 'rgba(88, 166, 255, 0.2)' # Strong highlight
    
    # Modern Canvas and Graphics - Configurable
    CANVAS_BACKGROUND = _theme_cfg.get('canvas_background', '#0d1117')
    GRID_MAJOR = _theme_cfg.get('grid_major', '#30363d')
    GRID_MINOR = _theme_cfg.get('grid_minor', '#21262d')
    GEOMETRY_LINE = _theme_cfg.get('geometry_line', '#58a6ff')
    GEOMETRY_POINT = _theme_cfg.get('geometry_point', '#f78166')
    GEOMETRY_SELECTED = _theme_cfg.get('geometry_selected', '#3fb950')
    MESH_LINE = _theme_cfg.get('mesh_line', '#6e7681')
    MESH_NODE = '#f85149'            # Muted red mesh nodes
    MESH_BOUNDARY = _theme_cfg.get('mesh_boundary', '#d29922')
    
    # Typography Scale - Configurable via configuration.yaml (4K OPTIMIZED)
    FONT_SIZE_TINY = _typography_cfg.get('font_size_tiny', 13)
    FONT_SIZE_SMALL = _typography_cfg.get('font_size_small', 14)
    FONT_SIZE_NORMAL = _typography_cfg.get('font_size_normal', 16)
    FONT_SIZE_MEDIUM = _typography_cfg.get('font_size_medium', 18)
    FONT_SIZE_LARGE = _typography_cfg.get('font_size_large', 20)
    FONT_SIZE_XLARGE = _typography_cfg.get('font_size_xlarge', 24)
    FONT_SIZE_TITLE = _typography_cfg.get('font_size_title', 28)
    FONT_SIZE_HERO = _typography_cfg.get('font_size_hero', 34)
    
    # Font weights for premium typography - Configurable
    FONT_WEIGHT_LIGHT = _typography_cfg.get('font_weight_light', 300)
    FONT_WEIGHT_NORMAL = _typography_cfg.get('font_weight_normal', 400)
    FONT_WEIGHT_MEDIUM = _typography_cfg.get('font_weight_medium', 500)
    FONT_WEIGHT_SEMIBOLD = _typography_cfg.get('font_weight_semibold', 600)
    FONT_WEIGHT_BOLD = _typography_cfg.get('font_weight_bold', 700)
    FONT_WEIGHT_BLACK = 900
    
    # Spacing Scale - Configurable (4K OPTIMIZED)
    SPACING_XS = _layout_cfg.get('spacing_xs', 8)
    SPACING_SM = _layout_cfg.get('spacing_sm', 14)
    SPACING_MD = _layout_cfg.get('spacing_md', 24)
    SPACING_LG = _layout_cfg.get('spacing_lg', 36)
    SPACING_XL = _layout_cfg.get('spacing_xl', 48)
    SPACING_XXL = _layout_cfg.get('spacing_xxl', 72)
    
    # Border Radius Scale - Configurable (4K OPTIMIZED)
    RADIUS_SM = _layout_cfg.get('radius_sm', 10)
    RADIUS_MD = _layout_cfg.get('radius_md', 14)
    RADIUS_LG = _layout_cfg.get('radius_lg', 20)
    RADIUS_XL = _layout_cfg.get('radius_xl', 28)
    RADIUS_ROUND = 50
    
    # Button styling from configuration
    BUTTON_MIN_HEIGHT = _buttons_cfg.get('min_height', 40)
    BUTTON_MIN_WIDTH = _buttons_cfg.get('min_width', 100)
    BUTTON_PADDING_H = _buttons_cfg.get('padding_horizontal', 24)
    BUTTON_PADDING_V = _buttons_cfg.get('padding_vertical', 10)
    BUTTON_BORDER_WIDTH = _buttons_cfg.get('border_width', 2)
    BUTTON_BORDER_RADIUS = _buttons_cfg.get('border_radius', 8)
    BUTTON_DEFAULT_STYLE = _buttons_cfg.get('default_style', 'outlined')
    
    # Layout configuration
    INPUT_PANEL_MIN_WIDTH = _layout_cfg.get('input_panel_min_width', 520)
    BOTTOM_PANEL_MIN_HEIGHT = _layout_cfg.get('bottom_panel_min_height', 150)
    SPLITTER_PANEL_RATIO = _layout_cfg.get('splitter_panel_ratio', 35)
    SPLITTER_CANVAS_RATIO = _layout_cfg.get('splitter_canvas_ratio', 65)
    SPLITTER_BOTTOM_RATIO = _layout_cfg.get('splitter_bottom_ratio', 20)
    
    # Display scaling configuration
    DISPLAY_AUTO_SCALE = _display_cfg.get('auto_scale', True)
    DISPLAY_SCALE_FACTOR = _display_cfg.get('scale_factor', None)
    DISPLAY_MIN_SCALE = _display_cfg.get('min_scale', 1.0)
    DISPLAY_MAX_SCALE = _display_cfg.get('max_scale', 2.5)
    
    @classmethod
    def reload_configuration(cls):
        """Reload configuration from file. Call this to apply changes without restart."""
        global _CONFIG
        _CONFIG = _load_configuration()
        # Update class-level config references
        cls._theme_cfg = _CONFIG.get('theme', {})
        cls._typography_cfg = _CONFIG.get('typography', {})
        cls._buttons_cfg = _CONFIG.get('buttons', {})
        cls._layout_cfg = _CONFIG.get('layout', {})
        cls._display_cfg = _CONFIG.get('display', {})
    
    @classmethod
    def get_gradient(cls, color1, color2, direction="vertical", stops=None):
        """Generate premium gradient with advanced options."""
        if stops is None:
            stops = [(0, color1), (1, color2)]
        
        stop_str = ", ".join([f"stop:{stop[0]} {stop[1]}" for stop in stops])
        
        if direction == "vertical":
            return f"qlineargradient(x1:0, y1:0, x2:0, y2:1, {stop_str})"
        elif direction == "horizontal":
            return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, {stop_str})"
        elif direction == "diagonal":
            return f"qlineargradient(x1:0, y1:0, x2:1, y2:1, {stop_str})"
        elif direction == "radial":
            return f"qradialgradient(cx:0.5, cy:0.5, radius:1, {stop_str})"
    
    @classmethod
    def get_premium_gradient(cls, color_name="primary"):
        """Get premium multi-stop gradients for ultra-modern look."""
        gradients = {
            "primary": cls.get_gradient(
                cls.PRIMARY, cls.PRIMARY_VARIANT, "diagonal",
                [(0, cls.PRIMARY_LIGHT), (0.5, cls.PRIMARY), (1, cls.PRIMARY_VARIANT)]
            ),
            "success": cls.get_gradient(
                cls.SUCCESS, cls.SUCCESS_VARIANT, "diagonal",
                [(0, cls.SUCCESS), (1, cls.SUCCESS_VARIANT)]
            ),
            "accent": cls.get_gradient(
                cls.ACCENT, cls.ACCENT_VARIANT, "diagonal", 
                [(0, cls.ACCENT_LIGHT), (0.5, cls.ACCENT), (1, cls.ACCENT_VARIANT)]
            ),
            "surface": cls.get_gradient(
                cls.SURFACE_VARIANT, cls.SURFACE_ELEVATED, "vertical"
            )
        }
        return gradients.get(color_name, gradients["primary"])
    
    @classmethod
    def get_world_class_button_style(cls, variant="primary", size="normal"):
        """Generate world-class button styling with advanced effects."""
        
        # Size configurations
        sizes = {
            "small": {"padding": "6px 12px", "font_size": cls.FONT_SIZE_SMALL, "radius": cls.RADIUS_SM},
            "normal": {"padding": "10px 20px", "font_size": cls.FONT_SIZE_NORMAL, "radius": cls.RADIUS_MD},
            "large": {"padding": "14px 28px", "font_size": cls.FONT_SIZE_MEDIUM, "radius": cls.RADIUS_LG},
            "xl": {"padding": "18px 36px", "font_size": cls.FONT_SIZE_LARGE, "radius": cls.RADIUS_XL}
        }
        
        size_config = sizes.get(size, sizes["normal"])
        
        if variant == "primary":
            return f"""
                QPushButton {{
                    background: {cls.get_premium_gradient("primary")};
                    color: {cls.TEXT_PRIMARY};
                    border: 2px solid {cls.PRIMARY};
                    border-radius: {size_config["radius"]}px;
                    padding: {size_config["padding"]};
                    font-weight: 600;
                    font-size: {size_config["font_size"]}px;
                    min-height: 16px;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    letter-spacing: 0.5px;
                }}
                QPushButton:hover {{
                    background: {cls.get_premium_gradient("primary")};
                    border-color: {cls.PRIMARY_LIGHT};
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px {cls.SHADOW_GLOW};
                }}
                QPushButton:pressed {{
                    background: {cls.PRIMARY_VARIANT};
                    transform: translateY(0px);
                    box-shadow: 0 4px 15px {cls.SHADOW_GLOW};
                }}
                QPushButton:disabled {{
                    background: {cls.SECONDARY};
                    color: {cls.TEXT_TERTIARY};
                    border-color: {cls.BORDER_SUBTLE};
                    transform: none;
                    box-shadow: none;
                }}
            """
        elif variant == "success":
            return f"""
                QPushButton {{
                    background: {cls.get_premium_gradient("success")};
                    color: {cls.TEXT_PRIMARY};
                    border: 2px solid {cls.SUCCESS};
                    border-radius: {size_config["radius"]}px;
                    padding: {size_config["padding"]};
                    font-weight: 600;
                    font-size: {size_config["font_size"]}px;
                    min-height: 16px;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    letter-spacing: 0.5px;
                }}
                QPushButton:hover {{
                    background: {cls.SUCCESS};
                    border-color: {cls.SUCCESS};
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(0, 255, 136, 0.3);
                }}
                QPushButton:pressed {{
                    background: {cls.SUCCESS_VARIANT};
                    transform: translateY(0px);
                }}
            """
        elif variant == "danger":
            return f"""
                QPushButton {{
                    background: {cls.get_gradient(cls.ERROR, cls.ERROR_VARIANT)};
                    color: {cls.TEXT_PRIMARY};
                    border: 2px solid {cls.ERROR};
                    border-radius: {size_config["radius"]}px;
                    padding: {size_config["padding"]};
                    font-weight: 600;
                    font-size: {size_config["font_size"]}px;
                    min-height: 16px;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    letter-spacing: 0.5px;
                }}
                QPushButton:hover {{
                    background: {cls.ERROR_VARIANT};
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(255, 71, 87, 0.3);
                }}
                QPushButton:pressed {{
                    background: {cls.ERROR};
                    transform: translateY(0px);
                }}
            """
        else:  # secondary
            return f"""
                QPushButton {{
                    background: {cls.get_premium_gradient("surface")};
                    color: {cls.TEXT_SECONDARY};
                    border: 2px solid {cls.BORDER};
                    border-radius: {size_config["radius"]}px;
                    padding: {size_config["padding"]};
                    font-weight: 500;
                    font-size: {size_config["font_size"]}px;
                    min-height: 16px;
                    font-family: 'Segoe UI', 'Inter', sans-serif;
                    letter-spacing: 0.3px;
                }}
                QPushButton:hover {{
                    background: {cls.SURFACE_ELEVATED};
                    color: {cls.TEXT_PRIMARY};
                    border-color: {cls.BORDER_ACCENT};
                    transform: translateY(-1px);
                    box-shadow: 0 4px 15px {cls.SHADOW_SOFT};
                }}
                QPushButton:pressed {{
                    background: {cls.SURFACE_VARIANT};
                    transform: translateY(0px);
                }}
            """
    
    @classmethod
    def get_premium_card_style(cls, elevation="medium"):
        """Generate premium card styling with advanced shadows and effects."""
        elevations = {
            "low": f"0 2px 8px {cls.SHADOW_SOFT}",
            "medium": f"0 4px 20px {cls.SHADOW_MEDIUM}",
            "high": f"0 8px 40px {cls.SHADOW_STRONG}"
        }
        
        shadow = elevations.get(elevation, elevations["medium"])
        
        return f"""
            QWidget {{
                background: {cls.get_premium_gradient("surface")};
                border: 1px solid {cls.BORDER};
                border-radius: {cls.RADIUS_LG}px;
                padding: {cls.SPACING_LG}px;
                box-shadow: {shadow};
            }}
        """
    
    @classmethod
    def get_premium_input_style(cls):
        """Generate world-class input field styling."""
        return f"""
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
                background: {cls.INPUT_BACKGROUND};
                color: {cls.TEXT_PRIMARY};
                border: 2px solid {cls.INPUT_BORDER};
                border-radius: {cls.RADIUS_MD}px;
                padding: {cls.SPACING_MD}px {cls.SPACING_LG}px;
                font-size: {cls.FONT_SIZE_NORMAL}px;
                font-family: 'Segoe UI', 'Inter', sans-serif;
                selection-background-color: {cls.INPUT_SELECTION};
                letter-spacing: 0.3px;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
                border-color: {cls.INPUT_BORDER_FOCUS};
                background: {cls.SURFACE};
                box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1);
            }}
            QLineEdit::placeholder {{
                color: {cls.INPUT_PLACEHOLDER};
                font-style: italic;
            }}
            QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
                border-color: {cls.BORDER_ACCENT};
                background: {cls.SURFACE};
            }}
        """
    
    @classmethod
    def get_responsive_font_size(cls, base_size, scale_factor=1.0):
        """Calculate responsive font size based on screen DPI and scale factor."""
        # Ensure minimum readable size
        min_size = 10
        max_size = 72
        
        # Apply scale factor
        calculated_size = int(base_size * scale_factor)
        
        # Clamp to reasonable bounds
        return max(min_size, min(max_size, calculated_size))
    
    @classmethod
    def get_responsive_spacing(cls, base_spacing, scale_factor=1.0):
        """Calculate responsive spacing based on screen size."""
        return max(2, int(base_spacing * scale_factor))
    
    @classmethod
    def get_responsive_button_style(cls, scale_factor=1.0, variant="primary"):
        """Generate responsive button styling with proper scaling."""
        font_size = cls.get_responsive_font_size(cls.FONT_SIZE_NORMAL, scale_factor)
        padding_v = cls.get_responsive_spacing(cls.SPACING_MD, scale_factor)
        padding_h = cls.get_responsive_spacing(cls.SPACING_LG, scale_factor)
        border_radius = cls.get_responsive_spacing(cls.RADIUS_MD, scale_factor)
        
        # Different button variants
        variants = {
            "primary": {
                "bg": cls.BUTTON_PRIMARY,
                "bg_hover": cls.BUTTON_PRIMARY_HOVER,
                "bg_active": cls.BUTTON_PRIMARY_ACTIVE,
                "text": cls.TEXT_INVERSE
            },
            "secondary": {
                "bg": cls.BUTTON_SECONDARY,
                "bg_hover": cls.BUTTON_SECONDARY_HOVER,
                "bg_active": cls.BORDER,
                "text": cls.TEXT_PRIMARY
            },
            "success": {
                "bg": cls.BUTTON_SUCCESS,
                "bg_hover": cls.BUTTON_SUCCESS_HOVER,
                "bg_active": cls.SUCCESS_VARIANT,
                "text": cls.TEXT_INVERSE
            },
            "danger": {
                "bg": cls.BUTTON_DANGER,
                "bg_hover": cls.BUTTON_DANGER_HOVER,
                "bg_active": cls.ERROR_VARIANT,
                "text": cls.TEXT_INVERSE
            }
        }
        
        colors = variants.get(variant, variants["primary"])
        
        return f"""
            QPushButton {{
                background: {colors["bg"]};
                color: {colors["text"]};
                border: none;
                border-radius: {border_radius}px;
                padding: {padding_v}px {padding_h}px;
                font-size: {font_size}px;
                font-weight: 500;
                font-family: 'Segoe UI', 'Inter', sans-serif;
                letter-spacing: 0.5px;
                text-transform: none;
                min-height: {max(32, int(40 * scale_factor))}px;
            }}
            QPushButton:hover {{
                background: {colors["bg_hover"]};
                transform: translateY(-1px);
            }}
            QPushButton:pressed {{
                background: {colors["bg_active"]};
                transform: translateY(0px);
            }}
            QPushButton:disabled {{
                background: {cls.BORDER};
                color: {cls.TEXT_TERTIARY};
                opacity: 0.6;
            }}
        """
    
    @classmethod
    def get_responsive_text_style(cls, scale_factor=1.0, size_category="normal"):
        """Generate responsive text styling."""
        size_map = {
            "tiny": cls.FONT_SIZE_TINY,
            "small": cls.FONT_SIZE_SMALL,
            "normal": cls.FONT_SIZE_NORMAL,
            "medium": cls.FONT_SIZE_MEDIUM,
            "large": cls.FONT_SIZE_LARGE,
            "xlarge": cls.FONT_SIZE_XLARGE,
            "title": cls.FONT_SIZE_TITLE,
            "hero": cls.FONT_SIZE_HERO
        }
        
        base_size = size_map.get(size_category, cls.FONT_SIZE_NORMAL)
        font_size = cls.get_responsive_font_size(base_size, scale_factor)
        
        return f"""
            font-size: {font_size}px;
            font-family: 'Segoe UI', 'Inter', 'SF Pro Display', 'Roboto', sans-serif;
            color: {cls.TEXT_PRIMARY};
            letter-spacing: 0.3px;
            line-height: 1.5;
        """
    
    @classmethod
    def get_ultra_premium_card_style(cls, scale_factor=1.0, variant="primary"):
        """Generate ultra-premium card styling with world-class design."""
        border_radius = cls.get_responsive_spacing(cls.RADIUS_LG, scale_factor)
        padding = cls.get_responsive_spacing(cls.SPACING_XL, scale_factor)
        margin = cls.get_responsive_spacing(cls.SPACING_MD, scale_factor)
        
        variants = {
            "primary": {
                "bg": cls.get_premium_gradient("surface"),
                "border": cls.BORDER_ACCENT,
                "shadow": f"0 {int(8 * scale_factor)}px {int(32 * scale_factor)}px {cls.SHADOW_MEDIUM}"
            },
            "glass": {
                "bg": f"rgba(255, 255, 255, 0.03)",
                "border": cls.BORDER,
                "shadow": f"0 {int(12 * scale_factor)}px {int(40 * scale_factor)}px {cls.SHADOW_STRONG}"
            },
            "elevated": {
                "bg": cls.SURFACE_ELEVATED,
                "border": cls.BORDER_ACCENT,
                "shadow": f"0 {int(16 * scale_factor)}px {int(48 * scale_factor)}px {cls.SHADOW_STRONG}"
            }
        }
        
        style = variants.get(variant, variants["primary"])
        
        return f"""
            background: {style["bg"]};
            border: 1px solid {style["border"]};
            border-radius: {border_radius}px;
            padding: {padding}px;
            margin: {margin}px;
            box-shadow: {style["shadow"]};
            backdrop-filter: blur(12px);
        """
    
    @classmethod
    def get_world_class_animation_style(cls):
        """Generate world-class animation and transition styles."""
        return """
            /* Smooth transitions for everything */
            * {
                transition: all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1);
            }
            
            /* Hover animations */
            QPushButton:hover {
                transform: translateY(-2px) scale(1.02);
                box-shadow: 0 8px 25px rgba(0, 217, 255, 0.3);
            }
            
            QPushButton:pressed {
                transform: translateY(0px) scale(0.98);
                transition-duration: 0.1s;
            }
            
            /* Focus animations */
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                transform: scale(1.02);
                box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.2);
            }
            
            /* Tab animations */
            QTabBar::tab:hover:!selected {
                transform: translateY(-2px);
            }
        """
    
    @classmethod
    def get_fullscreen_scaling_style(cls, is_fullscreen=False, screen_width=1920):
        """Generate proper fullscreen scaling that maintains readability."""
        # Calculate fullscreen scale factor based on screen size
        if is_fullscreen:
            if screen_width >= 3840:  # 4K
                scale_factor = 1.8
                base_font = 18
            elif screen_width >= 2560:  # QHD
                scale_factor = 1.5
                base_font = 16
            elif screen_width >= 1920:  # FHD
                scale_factor = 1.3
                base_font = 15
            else:  # Smaller screens
                scale_factor = 1.2
                base_font = 14
        else:
            scale_factor = 1.0
            base_font = 14
        
        return f"""
            /* Fullscreen responsive scaling */
            * {{
                font-size: {base_font}px;
            }}
            
            QMainWindow {{
                font-size: {base_font}px;
            }}
            
            QGroupBox {{
                font-size: {int(base_font * 1.1)}px;
                padding: {int(20 * scale_factor)}px {int(12 * scale_factor)}px;
            }}
            
            QPushButton {{
                font-size: {int(base_font * 1.0)}px;
                padding: {int(12 * scale_factor)}px {int(24 * scale_factor)}px;
                min-height: {int(40 * scale_factor)}px;
                border-radius: {int(8 * scale_factor)}px;
            }}
            
            QLabel {{
                font-size: {int(base_font * 0.95)}px;
            }}
            
            QTabBar::tab {{
                font-size: {int(base_font * 1.0)}px;
                padding: {int(16 * scale_factor)}px {int(24 * scale_factor)}px;
                min-width: {int(120 * scale_factor)}px;
            }}
            
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                font-size: {int(base_font * 0.95)}px;
                padding: {int(12 * scale_factor)}px {int(16 * scale_factor)}px;
                min-height: {int(36 * scale_factor)}px;
            }}
            
            /* Ensure text stays readable */
            QWidget {{
                font-family: 'Segoe UI', 'Inter', 'SF Pro Display', sans-serif;
                font-weight: 400;
            }}
        """
    
    @classmethod
    def get_glass_morphism_style(cls, scale_factor=1.0):
        """Generate stunning glass morphism effects for modern UI."""
        blur_radius = int(16 * scale_factor)
        border_radius = cls.get_responsive_spacing(cls.RADIUS_LG, scale_factor)
        
        return f"""
            /* Glass morphism panels */
            .glass-panel {{
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur({blur_radius}px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: {border_radius}px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            
            /* Glass morphism buttons */
            .glass-button {{
                background: rgba(0, 217, 255, 0.1);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(0, 217, 255, 0.3);
                color: {cls.TEXT_PRIMARY};
            }}
            
            .glass-button:hover {{
                background: rgba(0, 217, 255, 0.2);
                box-shadow: 0 0 20px rgba(0, 217, 255, 0.4);
            }}
        """
