"""
World-Class Modern UI Theme for Nozzle CFD Design Tool

An absolutely stunning dark theme with premium styling, advanced animations,
and professional-grade visual design inspired by the best modern applications.
"""

class Theme:
    """World-class dark theme with premium styling and advanced visual effects."""
    
    # Core Background Colors - Modern sophisticated palette
    BACKGROUND = '#0d1117'           # GitHub-like dark background
    SURFACE = '#161b22'              # Elevated surface with subtle contrast
    SURFACE_VARIANT = '#21262d'      # Cards and containers
    SURFACE_ELEVATED = '#30363d'     # Highest elevation surfaces
    SECONDARY = '#484f58'            # Secondary surface elements
    
    # Modern Accent Colors - Muted and professional
    PRIMARY = '#58a6ff'              # Soft blue for modern feel
    PRIMARY_VARIANT = '#388bfd'      # Darker blue for depth
    PRIMARY_LIGHT = '#79c0ff'        # Lighter blue for highlights
    ACCENT = '#f78166'               # Muted coral/orange for warmth
    ACCENT_VARIANT = '#e36749'       # Darker coral for interactions
    ACCENT_LIGHT = '#ffa28b'         # Light coral for subtle highlights
    
    # Professional Status Colors - Muted and elegant
    SUCCESS = '#3fb950'              # Muted green for success
    SUCCESS_VARIANT = '#2ea043'      # Deeper green
    WARNING = '#d29922'              # Warm amber for warnings
    WARNING_VARIANT = '#bf8700'      # Deeper warning amber
    ERROR = '#f85149'                # Muted red for errors
    ERROR_VARIANT = '#da3633'        # Deeper error red
    INFO = '#58a6ff'                 # Same as primary for consistency
    INFO_VARIANT = '#388bfd'         # Deeper info blue
    
    # Modern High Contrast Text - Clean and readable
    TEXT_PRIMARY = '#f0f6fc'         # Soft white for maximum readability
    TEXT_SECONDARY = '#8b949e'       # Muted gray for secondary text
    TEXT_TERTIARY = '#6e7681'        # Subtle gray for less important info
    TEXT_INVERSE = '#0d1117'         # Dark text for light backgrounds
    TEXT_ACCENT = '#58a6ff'          # Accent text color (muted blue)
    TEXT_SUCCESS = '#3fb950'         # Success text color (muted green)
    TEXT_WARNING = '#d29922'         # Warning text color (muted amber)
    TEXT_ERROR = '#f85149'           # Error text color (muted red)
    
    # Backward compatibility
    TEXT = TEXT_PRIMARY              # Alias for old theme references
    
    # Modern Interactive Elements - Muted and professional
    BUTTON_PRIMARY = '#58a6ff'       # Soft blue primary buttons
    BUTTON_PRIMARY_HOVER = '#79c0ff' # Lighter blue on hover
    BUTTON_PRIMARY_ACTIVE = '#388bfd' # Darker when pressed
    BUTTON_SECONDARY = '#30363d'     # Muted secondary buttons
    BUTTON_SECONDARY_HOVER = '#484f58' # Lighter on hover
    BUTTON_DANGER = '#f85149'        # Muted danger buttons
    BUTTON_DANGER_HOVER = '#ff6b6b'  # Lighter danger on hover
    BUTTON_SUCCESS = '#3fb950'       # Muted success action buttons
    BUTTON_SUCCESS_HOVER = '#56d364' # Lighter success on hover
    
    # Modern Borders and Dividers - Subtle and clean
    BORDER = '#30363d'               # Subtle borders
    BORDER_SUBTLE = '#21262d'        # Very subtle borders
    BORDER_ACCENT = '#58a6ff'        # Accent borders for focus
    BORDER_SUCCESS = '#3fb950'       # Success state borders
    BORDER_WARNING = '#d29922'       # Warning state borders
    BORDER_ERROR = '#f85149'         # Error state borders
    
    # Modern Input Elements - Clean and minimal
    INPUT_BACKGROUND = '#0d1117'     # Clean input backgrounds
    INPUT_BORDER = '#30363d'         # Input borders
    INPUT_BORDER_FOCUS = '#58a6ff'   # Blue focus borders
    INPUT_BORDER_ERROR = '#f85149'   # Error state borders
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
    
    # Modern Canvas and Graphics - Clean and professional
    CANVAS_BACKGROUND = '#0d1117'    # Clean canvas background
    GRID_MAJOR = '#30363d'           # Major grid lines
    GRID_MINOR = '#21262d'           # Minor grid lines
    GEOMETRY_LINE = '#58a6ff'        # Soft blue geometry lines
    GEOMETRY_POINT = '#f78166'       # Coral geometry points
    GEOMETRY_SELECTED = '#3fb950'    # Green for selected elements
    MESH_LINE = '#6e7681'            # Gray mesh lines
    MESH_NODE = '#f85149'            # Muted red mesh nodes
    MESH_BOUNDARY = '#d29922'        # Amber boundary elements
    
    # Typography Scale - Professional hierarchy (SMALLER SIZES)
    FONT_SIZE_TINY = 9
    FONT_SIZE_SMALL = 10
    FONT_SIZE_NORMAL = 11
    FONT_SIZE_MEDIUM = 12
    FONT_SIZE_LARGE = 14
    FONT_SIZE_XLARGE = 16
    FONT_SIZE_TITLE = 18
    FONT_SIZE_HERO = 22
    
    # Font weights for premium typography
    FONT_WEIGHT_LIGHT = 300
    FONT_WEIGHT_NORMAL = 400
    FONT_WEIGHT_MEDIUM = 500
    FONT_WEIGHT_SEMIBOLD = 600
    FONT_WEIGHT_BOLD = 700
    FONT_WEIGHT_BLACK = 900
    
    # Spacing Scale - Perfect proportions
    SPACING_XS = 4
    SPACING_SM = 8
    SPACING_MD = 16
    SPACING_LG = 24
    SPACING_XL = 32
    SPACING_XXL = 48
    
    # Border Radius Scale - Modern rounded corners
    RADIUS_SM = 6
    RADIUS_MD = 10
    RADIUS_LG = 14
    RADIUS_XL = 20
    RADIUS_ROUND = 50
    
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
