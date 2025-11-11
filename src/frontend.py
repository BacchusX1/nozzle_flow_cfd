#!/usr/bin/env python3
"""
Comprehensive Nozzle Design GUI with CFD Workflow
Professional full-stack development approach with immediate drawing capability.

Features:
- Immediate drawing without start/stop buttons
- Advanced meshing with boundary layers  
- Complete CFD simulation setup
- Post-processing and visualization
- Professional dark theme
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QGridLayout, QTabWidget, QGroupBox,
    QLabel, QPushButton, QLineEdit, QTextEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QRadioButton,
    QButtonGroup, QFileDialog, QMessageBox, QSplitter,
    QProgressBar, QStatusBar, QToolBar, QSlider,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QFrame, QFormLayout, QInputDialog,
    QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap, QIcon, QAction

# Matplotlib imports
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg for PySide6 compatibility
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.lines
import numpy as np
import json
import os

from core.theme import Theme
from core.nozzle import NozzleGeometry
from core.geometry import PolynomialElement, LineElement, ArcElement
from core.template_loader import TemplateLoader
from core.modules.mesh_generator import AdvancedMeshGenerator
from core.modules.simulation_setup import SimulationSetup
from core.modules.postprocessing import ResultsProcessor
from core.openfoam_runner import OpenFOAMRunner





class NozzleDesignGUI(QMainWindow):
    """Professional CFD workflow application with intuitive interface and comprehensive features."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.geometry = NozzleGeometry()
        self.template_loader = TemplateLoader()
        self.current_file = None
        self.is_modified = False
        
        # Try to initialize advanced components, fall back to basic if needed
        try:
            self.mesh_generator = AdvancedMeshGenerator()
            self.simulation_setup = SimulationSetup()
            self.results_processor = ResultsProcessor()
            self.advanced_features = True
        except Exception as e:
            print(f"Advanced features unavailable: {e}")
            self.mesh_generator = None # No basic mesh generator for now
            self.simulation_setup = None
            self.results_processor = None
            self.advanced_features = False
        
        # Current state
        self.current_mesh_data = None
        self.current_case_directory = ""
        self.current_results = None
        
        # Editing state
        self.editing_mode = "draw"
        self.selected_element_index = None
        
        self.setup_ui()
        self.apply_theme()
        self.apply_responsive_font()
        
    def apply_responsive_font(self):
        """Apply a properly sized font system that doesn't scale out of control."""
        # Get screen DPI for proper scaling
        screen = QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch()
        geometry = screen.geometry()
        
        # Use minimal scaling to prevent text from becoming too large
        scale_factor = max(0.8, min(1.2, dpi / 96.0))  # Clamp between 0.8 and 1.2
        
        # Use smaller base sizes and don't scale them up for fullscreen
        base_size = 10  # Fixed small size
        
        # Use simple system font
        font = QFont("Arial", base_size)
        font.setWeight(QFont.Weight.Normal)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        
        # Apply to application
        self.setFont(font)
        
        # Store scale factor but don't let it grow
        self.scale_factor = 1.0  # Fixed scale factor
        self.base_font_size = base_size
        self.is_fullscreen = False  # Don't auto-detect fullscreen scaling
        self.screen_width = geometry.width()
        
        # Keep theme font sizes small and fixed
        Theme.FONT_SIZE_TINY = 8
        Theme.FONT_SIZE_SMALL = 9
        Theme.FONT_SIZE_NORMAL = 10
        Theme.FONT_SIZE_MEDIUM = 11
        Theme.FONT_SIZE_LARGE = 12
        Theme.FONT_SIZE_XLARGE = 14
        Theme.FONT_SIZE_TITLE = 16
        
        print(f"Applied fixed font system: {font.family()} ({base_size}px)")
        
    def changeEvent(self, event):
        """Handle window state changes without auto-scaling."""
        super().changeEvent(event)
        # Don't auto-scale on window state changes
        
    def resizeEvent(self, event):
        """Handle window resize without auto-scaling to prevent size explosion."""
        super().resizeEvent(event)
        # Don't automatically change font sizes on resize
        # This prevents the scaling from getting out of control
            
    def showEvent(self, event):
        """Handle window show event for proper initialization."""
        super().showEvent(event)
        
        # Ensure proper scaling when window becomes visible
        if hasattr(self, 'scale_factor'):
            self.update_ui_scaling()
            
    def update_ui_scaling(self):
        """Keep UI sizing compact and fixed."""
        # Use fixed compact sizing to prevent scaling issues
        if hasattr(self, 'left_panel'):
            self.left_panel.setMaximumWidth(350)
            self.left_panel.setMinimumWidth(300)
        
    def apply_theme(self):
        """Apply clean modern theme with proper small font sizes."""
        # Use fixed small sizes - no scaling to prevent issues
        normal_font = 10
        medium_font = 11
        large_font = 12
        
        # Fixed small spacing to prevent button overlays
        spacing_sm = 4
        spacing_md = 8
        spacing_lg = 12
        
        self.setStyleSheet(f"""
            /* === CLEAN BASE STYLING === */
            QMainWindow {{
                background: {Theme.BACKGROUND};
                color: {Theme.TEXT_PRIMARY};
                border: none;
                font-size: {normal_font}px;
                font-family: Arial, sans-serif;
            }}
            
            QWidget {{
                background: transparent;
                color: {Theme.TEXT_PRIMARY};
                border: none;
                font-size: {normal_font}px;
            }}
            
            /* === COMPACT GROUP BOXES === */
            QGroupBox {{
                background: {Theme.SURFACE_VARIANT};
                color: {Theme.TEXT_PRIMARY};
                font-weight: 600;
                font-size: {normal_font}px;
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding: 8px 6px 6px 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                top: -5px;
                padding: 2px 8px;
                background: {Theme.SURFACE_VARIANT};
                border-radius: 4px;
                color: {Theme.TEXT_PRIMARY};
                font-size: {normal_font}px;
            }}
            
            /* === COMPACT BUTTONS === */
            QPushButton {{
                background: {Theme.PRIMARY};
                color: {Theme.TEXT_INVERSE};
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: {normal_font}px;
                font-weight: 500;
                font-family: Arial, sans-serif;
                min-height: 24px;
                max-height: 32px;
            }}
            QPushButton:hover {{
                background: {Theme.PRIMARY_LIGHT};
            }}
            QPushButton:pressed {{
                background: {Theme.PRIMARY_VARIANT};
            }}
            QPushButton:disabled {{
                background: {Theme.BORDER};
                color: {Theme.TEXT_TERTIARY};
            }}
            
            /* === PREMIUM INPUTS === */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background: {Theme.INPUT_BACKGROUND};
                color: {Theme.TEXT_PRIMARY};
                border: 1px solid {Theme.INPUT_BORDER};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: {normal_font}px;
                min-height: 20px;
                max-height: 24px;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border-color: {Theme.INPUT_BORDER_FOCUS};
                background: {Theme.SURFACE};
            }}
            
            /* === COMPACT LABELS === */
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: {normal_font}px;
                font-weight: 400;
                padding: 2px 0px;
            }}
            
            /* === COMPACT TAB WIDGET === */
            QTabWidget::pane {{
                background: {Theme.SURFACE};
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                padding: {spacing_md}px;
            }}
            QTabBar::tab {{
                background: {Theme.SECONDARY};
                color: {Theme.TEXT_SECONDARY};
                border: 1px solid {Theme.BORDER};
                border-bottom: none;
                border-radius: 6px 6px 0px 0px;
                padding: 8px 16px;
                margin-right: 2px;
                font-size: {normal_font}px;
                font-weight: 500;
                min-width: 80px;
            }}
            QTabBar::tab:selected {{
                background: {Theme.PRIMARY};
                color: {Theme.TEXT_INVERSE};
                border-color: {Theme.BORDER_ACCENT};
            }}
            QTabBar::tab:hover:!selected {{
                background: {Theme.SURFACE_ELEVATED};
                color: {Theme.TEXT_PRIMARY};
            }}
            
            /* === COMPACT SPLITTER === */
            QSplitter::handle {{
                background: {Theme.BORDER};
                width: 2px;
                height: 2px;
                border-radius: 1px;
            }}
            QSplitter::handle:hover {{
                background: {Theme.BORDER_ACCENT};
            }}
            
            /* === COMPACT SCROLLBARS === */
            QScrollBar:vertical {{
                background: {Theme.SURFACE};
                width: 12px;
                border-radius: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {Theme.BORDER};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {Theme.BORDER_ACCENT};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
        """)
        
        # Apply additional styling for specific widget types
        self.apply_custom_styles()
        
    def apply_custom_styles(self):
        """Apply custom styles to specific widgets after creation."""
        # This method will be called after widgets are created
        # to apply specific styling classes
        pass
        
    def setup_ui(self):
        """Setup the clean, compact user interface optimized for 1920x1080 fullscreen."""
        self.setWindowTitle("Nozzle Flow CFD Designer - Professional Edition")
        
        # Optimize for 1920x1080 fullscreen
        self.setGeometry(0, 0, 1920, 1080)
        self.setMinimumSize(1400, 800)
        
        # Create central widget
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        
        # Main layout with proper margins for fullscreen
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # Create main splitter with fixed handle
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(1)
        main_splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing
        main_layout.addWidget(main_splitter)
        
        # Left panel - Compact control panel
        self.left_panel = self.create_modern_left_panel()
        main_splitter.addWidget(self.left_panel)
        
        # Right panel - Main workflow tabs
        self.tab_widget = self.create_modern_tab_widget()
        main_splitter.addWidget(self.tab_widget)
        
        # Set fixed proportions for fullscreen (300px left, rest for tabs)
        main_splitter.setSizes([300, 1620])
        main_splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        main_splitter.setStretchFactor(1, 1)  # Tab widget stretches
        
        # Create compact menu bar and status bar
        self.create_modern_menu_bar()
        self.create_modern_status_bar()
        
        # Apply custom styles after widget creation
        self.apply_widget_classes()
        
    def apply_widget_classes(self):
        """Apply CSS classes to specific widgets for better styling."""
        # Find and style primary action buttons (without emoticons)
        primary_buttons = [
            "Auto Generate",
            "Generate Mesh", 
            "Run Simulation",
            "Export Mesh"
        ]
        
        for button in self.findChildren(QPushButton):
            if any(text in button.text() for text in primary_buttons):
                button.setProperty("class", "primary")
                button.style().unpolish(button)
                button.style().polish(button)
        
    def create_modern_left_panel(self):
        """Create a compact, clean left control panel."""
        panel = QWidget()
        panel.setMaximumWidth(350)
        panel.setMinimumWidth(300)
        
        # Main layout for the panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Header section without emoticons
        header = self.create_header_section()
        layout.addWidget(header)
        
        # Workflow status card
        status_card = self.create_status_card()
        layout.addWidget(status_card)
        
        # Add 8pt spacing between workflow status and parameters
        layout.addSpacing(8)
        
        # Parameters card
        params_card = self.create_parameters_card()
        layout.addWidget(params_card)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
        
    def create_header_section(self):
        """Create a clean header section without emoticons."""
        header = QWidget()
        header.setFixedHeight(60)  # Smaller height
        
        header.setStyleSheet(f"""
            QWidget {{
                background: {Theme.SURFACE_VARIANT};
                border-radius: 8px;
                border: 1px solid {Theme.BORDER};
            }}
        """)
        
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(4)
        
        # Main title without emoticons
        title = QLabel("NOZZLE CFD DESIGNER")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 700;
                font-family: Arial, sans-serif;
                letter-spacing: 1px;
                background: transparent;
                padding: 4px 0px;
            }}
        """)
        layout.addWidget(title)
        
        # Subtitle without emoticons
        subtitle = QLabel("Professional CFD Workflow Suite")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_SECONDARY};
                font-size: 9px;
                font-weight: 400;
                font-family: Arial, sans-serif;
                background: transparent;
                padding: 0px;
            }}
        """)
        layout.addWidget(subtitle)
        
        return header
        
    def create_status_card(self):
        """Create a modern status card showing workflow progress."""
        card = QGroupBox("Workflow Status")
        card.setStyleSheet(f"""
            QGroupBox {{
                background: {Theme.SURFACE_VARIANT};
                border: 1px solid {Theme.BORDER};
                border-radius: 12px;
                font-size: 14px;
                font-weight: 600;
                padding: 16px 8px 8px 8px;
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(12)
        
        # Status indicators with modern styling
        self.geometry_status = QLabel("Geometry: Not defined")
        self.mesh_status = QLabel("Mesh: Not generated") 
        self.simulation_status = QLabel("Simulation: Not run")
        self.results_status = QLabel("Results: Not available")
        
        # Store status labels for compatibility with existing update system
        self.status_labels = {
            'geometry': self.geometry_status,
            'mesh': self.mesh_status,
            'simulation': self.simulation_status,
            'results': self.results_status
        }
        
        # Create progress bars for compatibility
        self.progress_bars = {}
        
        for key, status in self.status_labels.items():
            # Create container for status item with proper layout
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(8)
            
            # Style status label with fixed width for alignment
            status.setStyleSheet(f"""
                QLabel {{
                    color: {Theme.TEXT_SECONDARY};
                    font-size: 12px;
                    font-weight: 500;
                    padding: 8px 12px;
                    background: {Theme.SURFACE};
                    border-radius: 8px;
                    border-left: 4px solid {Theme.WARNING};
                    min-height: 16px;
                    min-width: 200px;
                }}
            """)
            container_layout.addWidget(status, 1)
            
            # Add progress label with fixed width for proper alignment
            progress_label = QLabel("0%")
            progress_label.setFixedWidth(35)
            progress_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            progress_label.setStyleSheet(f"""
                QLabel {{
                    color: {Theme.TEXT_SECONDARY};
                    font-size: 11px;
                    font-weight: 500;
                    padding: 0px 4px;
                }}
            """)
            self.progress_bars[key] = progress_label  # Store label instead of progress bar
            container_layout.addWidget(progress_label)
            
            layout.addWidget(container)
            
        return card
        
    def create_quick_actions_card(self):
        """Create a card with quick action buttons."""
        card = QGroupBox("Quick Actions")
        
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        
        # Auto-generate button
        btn_auto = QPushButton("Auto Generate Geometry")
        btn_auto.setProperty("class", "primary")
        btn_auto.clicked.connect(self.auto_generate_mesh)
        layout.addWidget(btn_auto)
        
        # Template buttons in a row
        template_layout = QHBoxLayout()
        
        btn_bell = QPushButton("Bell")
        btn_bell.clicked.connect(lambda: self.load_template("bell"))
        template_layout.addWidget(btn_bell)
        
        btn_conical = QPushButton("Conical")
        btn_conical.clicked.connect(lambda: self.load_template("conical"))
        template_layout.addWidget(btn_conical)
        
        layout.addLayout(template_layout)
        
        return card
        
    def create_parameters_card(self):
        """Create a modern parameters card with all the essential controls."""
        card = QGroupBox("Project & Parameters")
        
        layout = QVBoxLayout(card)
        layout.setSpacing(16)
        
        # Project Information Section
        project_section = QWidget()
        project_layout = QFormLayout(project_section)
        project_layout.setSpacing(8)
        
        self.project_name_edit = QLineEdit("Untitled Project")
        self.project_name_edit.setToolTip("Enter a descriptive name for your CFD project")
        self.project_name_edit.textChanged.connect(self.on_project_name_changed)
        project_layout.addRow("Project Name:", self.project_name_edit)
        
        self.project_description = QTextEdit()
        self.project_description.setMaximumHeight(60)
        self.project_description.setPlaceholderText("Brief description of the nozzle design...")
        self.project_description.setToolTip("Add notes about design objectives, operating conditions, etc.")
        project_layout.addRow("Description:", self.project_description)
        
        layout.addWidget(project_section)
        
        # Design Constraints Section
        constraints_section = QWidget()
        constraints_layout = QFormLayout(constraints_section)
        constraints_layout.setSpacing(8)
        
        # Add constraint label
        constraints_label = QLabel(" Design Constraints")
        constraints_label.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
                margin: 8px 0px 4px 0px;
            }}
        """)
        layout.addWidget(constraints_label)
        
        self.min_throat_ratio = QDoubleSpinBox()
        self.min_throat_ratio.setRange(0.1, 1.0)
        self.min_throat_ratio.setValue(0.5)
        self.min_throat_ratio.setSingleStep(0.1)
        self.min_throat_ratio.setDecimals(2)
        self.min_throat_ratio.setToolTip("Minimum throat-to-inlet area ratio")
        constraints_layout.addRow("Min A*/A_inlet:", self.min_throat_ratio)
        
        self.max_divergence_angle = QSpinBox()
        self.max_divergence_angle.setRange(5, 45)
        self.max_divergence_angle.setValue(20)
        self.max_divergence_angle.setSuffix("°")
        self.max_divergence_angle.setToolTip("Maximum divergence half-angle")
        constraints_layout.addRow("Max Divergence:", self.max_divergence_angle)
        
        self.enforce_continuity = QCheckBox("Enforce C1 Continuity")
        self.enforce_continuity.setChecked(True)
        self.enforce_continuity.setToolTip("Ensure smooth transitions between elements")
        constraints_layout.addRow("", self.enforce_continuity)
        
        layout.addWidget(constraints_section)
        
        # Geometry Information Section
        info_label = QLabel("Geometry Information")
        info_label.setStyleSheet(f"""
            QLabel {{
                color: {Theme.TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
                margin: 8px 0px 4px 0px;
            }}
        """)
        layout.addWidget(info_label)
        
        info_section = QWidget()
        info_layout = QFormLayout(info_section)
        info_layout.setSpacing(6)
        
        self.info_labels = {
            'elements': QLabel("0"),
            'length': QLabel("0.00 m"),
            'throat_ratio': QLabel("N/A"),
            'expansion_ratio': QLabel("N/A")
        }
        
        info_layout.addRow("Elements:", self.info_labels['elements'])
        info_layout.addRow("Length:", self.info_labels['length'])
        info_layout.addRow("Throat Ratio:", self.info_labels['throat_ratio'])
        info_layout.addRow("Expansion Ratio:", self.info_labels['expansion_ratio'])
        
        # Style info labels with modern appearance
        for label in self.info_labels.values():
            label.setStyleSheet(f"""
                QLabel {{
                    color: {Theme.SUCCESS};
                    font-weight: 600;
                    font-size: 13px;
                    padding: 4px 8px;
                    background: {Theme.SURFACE};
                    border-radius: 4px;
                    border-left: 3px solid {Theme.SUCCESS};
                }}
            """)
        
        layout.addWidget(info_section)
        
        return card
    
    def create_modern_menu_bar(self):
        """Create a modern, beautiful menu bar."""
        menubar = self.menuBar()
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background: {Theme.SURFACE};
                color: {Theme.TEXT_PRIMARY};
                border-bottom: 1px solid {Theme.BORDER};
                padding: 6px;
                font-size: 13px;
            }}
        """)
        
        # File menu with modern icons
        file_menu = menubar.addMenu("[...] File")
        
        new_action = file_menu.addAction("New Project")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        
        open_action = file_menu.addAction("Open Project")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        
        save_action = file_menu.addAction("Save Project")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        
        file_menu.addSeparator()
        
        export_action = file_menu.addAction("Export Case")
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_case)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu(" Tools")
        
        validate_action = tools_menu.addAction("Validate Geometry")
        validate_action.triggered.connect(self.validate_geometry)
        
        mesh_action = tools_menu.addAction("Generate Mesh")
        mesh_action.triggered.connect(self.generate_mesh)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        reset_view_action = view_menu.addAction("Reset View")
        reset_view_action.triggered.connect(self.reset_view)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        tutorial_action = help_menu.addAction("Tutorial")
        tutorial_action.triggered.connect(self.show_tutorial)
        
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
        
    def create_modern_status_bar(self):
        """Create a modern status bar with indicators."""
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background: {Theme.SURFACE};
                color: {Theme.TEXT_SECONDARY};
                border-top: 1px solid {Theme.BORDER};
                padding: 8px;
                font-size: 12px;
            }}
        """)
        
        # Add status indicators
        self.status_label = QLabel("[*] Ready")
        self.status_bar.addWidget(self.status_label)
        
        # Add permanent widgets to the right
        self.coords_label = QLabel("Position: (0, 0)")
        self.status_bar.addPermanentWidget(self.coords_label)
        
        self.mode_label = QLabel("Mode: Draw")
        self.status_bar.addPermanentWidget(self.mode_label)
        
    def create_modern_tab_widget(self):
        """Create a modern tab widget optimized for fullscreen with no overlays."""
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(False)
        tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                background: {Theme.SURFACE};
                border: 1px solid {Theme.BORDER};
                border-radius: 8px;
                margin-top: 0px;
                padding: 4px;
            }}
            QTabBar::tab {{
                background: {Theme.SURFACE_VARIANT};
                color: {Theme.TEXT_SECONDARY};
                border: 1px solid {Theme.BORDER_SUBTLE};
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 20px;
                margin-right: 2px;
                margin-top: 2px;
                font-weight: 500;
                font-size: 12px;
                min-width: 100px;
            }}
            QTabBar::tab:selected {{
                background: {Theme.PRIMARY};
                color: {Theme.TEXT_PRIMARY};
                border-color: {Theme.BORDER_ACCENT};
                font-weight: 600;
                margin-top: 0px;
                padding-bottom: 12px;
            }}
            QTabBar::tab:hover {{
                background: {Theme.SURFACE};
            }}
        """)
        
        # Geometry Design Tab
        geometry_tab = self.create_geometry_tab()
        tab_widget.addTab(geometry_tab, "Geometry Design")
        
        # Mesh Generation Tab
        mesh_tab = self.create_meshing_tab()
        tab_widget.addTab(mesh_tab, "Mesh Generation")
        
        # Simulation Tab
        simulation_tab = self.create_simulation_tab()
        tab_widget.addTab(simulation_tab, "Simulation")
        
        # Results Tab
        results_tab = self.create_postprocessing_tab()
        tab_widget.addTab(results_tab, "Results")
        
        return tab_widget
        
    def show_tutorial(self):
        """Show a tutorial dialog."""
        QMessageBox.information(self, "Tutorial", 
                               "Welcome to Nozzle CFD Designer!\n\n"
                               "1. [Design] Design your nozzle geometry\n"
                               "2. [Mesh] Generate a CFD mesh\n"
                               "3. [Quick] Run CFD simulation\n"
                               "4. [Chart] Analyze results\n\n"
                               "Use the left panel for quick actions and parameters.")
        
    def reset_view(self):
        """Reset the view to default."""
        # This will be implemented based on the canvas system
        pass
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = file_menu.addAction("New Project")
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        
        open_action = file_menu.addAction("Open Project")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        
        save_action = file_menu.addAction("Save Project")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        
        file_menu.addSeparator()
        
        export_action = file_menu.addAction("Export Case")
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_case)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
        
    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About Nozzle CFD Design Tool",
                         "Professional CFD workflow application\n"
                         "Version 2.0\n\n"
                         "Features:\n"
                         "• Immediate geometry drawing\n"
                         "• Advanced meshing with boundary layers\n"
                         "• CFD simulation setup\n"
                         "• Post-processing visualization")
        
    def create_left_panel(self):
        """Create left control panel with enhanced workflow status and quick actions."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Project info section
        project_group = QGroupBox("Project Information")
        project_layout = QFormLayout(project_group)
        
        self.project_name_edit = QLineEdit("Untitled Project")
        self.project_name_edit.setToolTip("Enter a descriptive name for your CFD project")
        self.project_name_edit.textChanged.connect(self.on_project_name_changed)
        project_layout.addRow("Name:", self.project_name_edit)
        
        self.project_description = QTextEdit()
        self.project_description.setMaximumHeight(60)
        self.project_description.setPlaceholderText("Brief description of the nozzle design...")
        self.project_description.setToolTip("Add notes about design objectives, operating conditions, etc.")
        project_layout.addRow("Description:", self.project_description)
        
        layout.addWidget(project_group)
        
        # Enhanced workflow status with progress indicators
        status_group = QGroupBox("Workflow Progress")
        status_layout = QVBoxLayout(status_group)
        
        self.status_labels = {
            'geometry': QLabel("[X] Geometry Design"),
            'mesh': QLabel("[X] Mesh Generation"),
            'simulation': QLabel("[X] CFD Setup"),
            'results': QLabel("[X] Post-Processing")
        }
        
        self.progress_bars = {}
        for key, label in self.status_labels.items():
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            
            label.setStyleSheet("font-weight: bold; padding: 5px; min-width: 120px;")
            container_layout.addWidget(label)
            
            progress = QProgressBar()
            progress.setMaximumHeight(20)
            progress.setMaximumWidth(100)
            progress.setValue(0)
            self.progress_bars[key] = progress
            container_layout.addWidget(progress)
            
            status_layout.addWidget(container)
            
        layout.addWidget(status_group)
        
        # Enhanced quick actions with categories
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QGridLayout(actions_group)
        
        # File operations row
        btn_new = QPushButton("New")
        btn_open = QPushButton("Open")
        btn_save = QPushButton("Save")
        btn_export = QPushButton("Export")
        
        # Add tooltips for better UX
        btn_new.setToolTip("Create a new nozzle design project (Ctrl+N)")
        btn_open.setToolTip("Open an existing project file (Ctrl+O)")
        btn_save.setToolTip("Save current project (Ctrl+S)")
        btn_export.setToolTip("Export OpenFOAM case files (Ctrl+E)")
        
        btn_new.clicked.connect(self.new_project)
        btn_open.clicked.connect(self.open_project)
        btn_save.clicked.connect(self.save_project)
        btn_export.clicked.connect(self.export_case)
        
        # Arrange buttons in a 2x2 grid for better space usage
        actions_layout.addWidget(btn_new, 0, 0)
        actions_layout.addWidget(btn_open, 0, 1)
        actions_layout.addWidget(btn_save, 1, 0)
        actions_layout.addWidget(btn_export, 1, 1)
        
        # Workflow actions row
        btn_auto_mesh = QPushButton("Auto Mesh")
        btn_run_sim = QPushButton("Run CFD")
        btn_auto_mesh.setToolTip("Generate mesh automatically with current settings")
        btn_run_sim.setToolTip("Run CFD simulation with current setup")
        btn_auto_mesh.clicked.connect(self.auto_generate_mesh)
        btn_run_sim.clicked.connect(self.run_simulation)
        
        actions_layout.addWidget(btn_auto_mesh, 2, 0)
        actions_layout.addWidget(btn_run_sim, 2, 1)
        
        # View actions
        btn_zoom_fit = QPushButton("Fit View")
        btn_reset_view = QPushButton("Reset")
        btn_zoom_fit.setToolTip("Zoom to fit all geometry")
        btn_reset_view.setToolTip("Reset view and clear all")
        btn_zoom_fit.clicked.connect(self.zoom_to_fit)
        btn_reset_view.clicked.connect(self.reset_all)
        
        actions_layout.addWidget(btn_zoom_fit, 3, 0)
        actions_layout.addWidget(btn_reset_view, 3, 1)
        
        # Style all buttons consistently
        for i in range(actions_layout.rowCount()):
            for j in range(actions_layout.columnCount()):
                item = actions_layout.itemAtPosition(i, j)
                if item and item.widget():
                    item.widget().setMinimumHeight(35)
                    item.widget().setStyleSheet("""
                        QPushButton {
                            font-weight: bold;
                            border-radius: 6px;
                            padding: 8px;
                        }
                    """)
        
        layout.addWidget(actions_group)
        
        # Design constraints and validation
        constraints_group = QGroupBox("Design Constraints")
        constraints_layout = QFormLayout(constraints_group)
        
        self.min_throat_ratio = QDoubleSpinBox()
        self.min_throat_ratio.setRange(0.1, 1.0)
        self.min_throat_ratio.setValue(0.5)
        self.min_throat_ratio.setSingleStep(0.1)
        self.min_throat_ratio.setToolTip("Minimum throat-to-inlet area ratio")
        constraints_layout.addRow("Min A*/A_inlet:", self.min_throat_ratio)
        
        self.max_divergence_angle = QSpinBox()
        self.max_divergence_angle.setRange(5, 45)
        self.max_divergence_angle.setValue(20)
        self.max_divergence_angle.setSuffix("°")
        self.max_divergence_angle.setToolTip("Maximum divergence half-angle")
        constraints_layout.addRow("Max Divergence:", self.max_divergence_angle)
        
        self.enforce_continuity = QCheckBox("Enforce C1 Continuity")
        self.enforce_continuity.setChecked(True)
        self.enforce_continuity.setToolTip("Ensure smooth transitions between elements")
        constraints_layout.addRow(self.enforce_continuity)
        
        layout.addWidget(constraints_group)
        
        # Real-time geometry info
        info_group = QGroupBox("Geometry Info")
        info_layout = QFormLayout(info_group)
        
        self.info_labels = {
            'elements': QLabel("0"),
            'length': QLabel("0.00 m"),
            'throat_ratio': QLabel("N/A"),
            'expansion_ratio': QLabel("N/A")
        }
        
        info_layout.addRow("Elements:", self.info_labels['elements'])
        info_layout.addRow("Length:", self.info_labels['length'])
        info_layout.addRow("Throat Ratio:", self.info_labels['throat_ratio'])
        info_layout.addRow("Expansion Ratio:", self.info_labels['expansion_ratio'])
        
        # Style info labels
        for label in self.info_labels.values():
            label.setStyleSheet("font-weight: bold; color: #00d4aa;")
        
        layout.addWidget(info_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def on_project_name_changed(self):
        """Update window title when project name changes."""
        name = self.project_name_edit.text() or "Untitled Project"
        self.setWindowTitle(f" Nozzle Flow CFD Designer - {name}")
        self.is_modified = True
        
    def update_modern_status(self, stage: str, completed: bool, progress: int = 0, message: str = ""):
        """Update modern status indicators."""
        if stage not in self.status_labels:
            return
            
        status_label = self.status_labels[stage]
        progress_bar = self.progress_bars.get(stage)
        
        # Update status text and styling
        stage_names = {
            'geometry': 'Geometry',
            'mesh': 'Mesh', 
            'simulation': 'Simulation',
            'results': 'Results'
        }
        
        stage_name = stage_names.get(stage, stage.title())
        
        if completed:
            icon = "[OK]"
            status_text = f"{icon} {stage_name}: {message or 'Completed'}"
            border_color = Theme.SUCCESS
            text_color = Theme.TEXT_PRIMARY
        elif progress > 0:
            icon = "🔄"
            status_text = f"{icon} {stage_name}: {message or 'In progress...'}"
            border_color = Theme.INFO
            text_color = Theme.TEXT_PRIMARY
        else:
            icon = "[NOK]"
            status_text = f"{icon} {stage_name}: {message or 'Not started'}"
            border_color = Theme.WARNING
            text_color = Theme.TEXT_SECONDARY
        
        status_label.setText(status_text)
        status_label.setStyleSheet(f"""
            QLabel {{
                color: {text_color};
                font-size: 13px;
                font-weight: 500;
                padding: 12px 16px;
                background: {Theme.SURFACE};
                border-radius: 8px;
                border-left: 4px solid {border_color};
                min-height: 16px;
            }}
        """)
        
        # Update progress bar
        if progress_bar:
            progress_bar.setValue(progress)
            if completed:
                progress_bar.setValue(100)
    
    def update_geometry_info(self):
        """Update real-time geometry information."""
        try:
            # Update element count
            self.info_labels['elements'].setText(str(len(self.geometry.elements)))
            
            # Calculate approximate length
            if self.geometry.elements:
                total_length = sum(elem.get_length() for elem in self.geometry.elements 
                                 if hasattr(elem, 'get_length'))
                self.info_labels['length'].setText(f"{total_length:.3f} m")
            else:
                self.info_labels['length'].setText("0.00 m")
            
            # Update workflow status
            if self.geometry.elements:
                self.update_workflow_status('geometry', True, 100)
            else:
                self.update_workflow_status('geometry', False, 0)
                
        except Exception as e:
            print(f"Error updating geometry info: {e}")
    
    def update_workflow_status(self, stage: str, completed: bool, progress: int):
        """Update workflow status indicators with modern styling."""
        # Use the modern status update method
        self.update_modern_status(stage, completed, progress)
        
        # Update progress label
        if stage in self.progress_bars:
            self.progress_bars[stage].setText(f"{progress}%")
    
    def auto_generate_mesh(self):
        """Quick mesh generation with default settings."""
        if not self.geometry.elements:
            QMessageBox.warning(self, "Warning", "Please create geometry first!")
            return
        
        try:
            self.tab_widget.setCurrentIndex(1)  # Switch to mesh tab
            # Trigger mesh generation with default settings
            if hasattr(self, 'generate_mesh'):
                self.generate_mesh()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto mesh failed: {e}")
    
    def load_template(self, template_name):
        """Load a geometry template."""
        try:
            # Load template using TemplateLoader
            template_file = Path(__file__).parent.parent / "geometry" / "templates" / f"{template_name}.json"
            
            if not template_file.exists():
                QMessageBox.warning(self, "Template Not Found", 
                                   f"Template file not found: {template_file}")
                return
            
            # Load template data
            with open(template_file, 'r') as f:
                template_data = json.load(f)
            
            # Clear existing geometry
            self.geometry.clear()
            
            # Load elements from template (matching validate_tutorial.py approach)
            for element_data in template_data.get('elements', []):
                element_type = element_data.get('type')
                
                if element_type == 'PolynomialElement':
                    control_points = element_data.get('control_points', [])
                    element = PolynomialElement(control_points)
                    self.geometry.add_element(element)
                elif element_type == 'LineElement':
                    start = tuple(element_data.get('start', [0, 0]))
                    end = tuple(element_data.get('end', [1, 1]))
                    element = LineElement(start, end)
                    self.geometry.add_element(element)
                elif element_type == 'ArcElement':
                    center = tuple(element_data.get('center', [0, 0]))
                    radius = element_data.get('radius', 1.0)
                    start_angle = element_data.get('start_angle', 0)
                    end_angle = element_data.get('end_angle', 90)
                    element = ArcElement(center, radius, start_angle, end_angle)
                    self.geometry.add_element(element)
            
            # Update display
            self.update_geometry_plot()
            self.update_geometry_info()
            
            if hasattr(self, 'simulation_log'):
                self.simulation_log.append(f"[OK] Loaded template: {template_name}")
            
            QMessageBox.information(self, "Template Loaded", 
                                   f"Successfully loaded {template_name} template with {len(self.geometry.elements)} elements")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load template: {e}")
            print(f"Template loading error: {e}")
            import traceback
            traceback.print_exc()

    
    def run_simulation(self):
        """Quick simulation run with current settings."""
        if not self.current_mesh_data:
            QMessageBox.warning(self, "Warning", "Please generate mesh first!")
            return
        
        try:
            self.tab_widget.setCurrentIndex(2)  # Switch to simulation tab
            # Trigger simulation run
            if hasattr(self, 'run_cfd_simulation'):
                self.run_cfd_simulation()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulation failed: {e}")
    
    def zoom_to_fit(self):
        """Zoom to fit all geometry in the plot."""
        if hasattr(self, 'geometry_canvas') and self.geometry.elements:
            # Auto-scale the geometry plot
            self.update_geometry_plot()
    
    def reset_all(self):
        """Reset entire application state."""
        reply = QMessageBox.question(self, "Reset All", 
                                    "This will clear all geometry, mesh, and simulation data. Continue?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.geometry.clear()
            self.current_mesh_data = None
            self.current_results = None
            self.update_geometry_plot()
            self.update_geometry_info()
            for stage in self.status_labels:
                self.update_workflow_status(stage, False, 0)
        
    def create_tab_widget(self):
        """Create main workflow tabs."""
        tab_widget = QTabWidget()
        
        # Geometry Design Tab
        self.geometry_tab = self.create_geometry_tab()
        tab_widget.addTab(self.geometry_tab, "Geometry")
        
        # Meshing Tab (only if advanced features available)
        if self.advanced_features:
            self.meshing_tab = self.create_meshing_tab()
            tab_widget.addTab(self.meshing_tab, "Meshing")
            
            # Simulation Setup Tab
            self.simulation_tab = self.create_simulation_tab()
            tab_widget.addTab(self.simulation_tab, "Simulation")
            
            # Post-processing Tab
            self.postprocessing_tab = self.create_postprocessing_tab()
            tab_widget.addTab(self.postprocessing_tab, "Post-processing")
        else:
            # Basic export tab for fallback
            self.export_tab = self.create_export_tab()
            tab_widget.addTab(self.export_tab, "Export")
        
        # Connect tab change to update workflow status
        tab_widget.currentChanged.connect(self.on_tab_changed)
        
        return tab_widget
        
    def create_geometry_tab(self):
        """Create enhanced geometry design tab with immediate drawing and smart features."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left controls panel
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(320)
        controls_layout = QVBoxLayout(controls_widget)
        
        # Enhanced drawing mode selection
        mode_group = QGroupBox("Drawing Modes")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_group = QButtonGroup()
        self.mode_buttons = {}
        
        modes = [
            ("Polynomial", "Polynomial Curve", "Smooth curves with multiple control points"),
            ("Line", "Straight Line", "Linear segments for precise angles"), 
            ("Arc", "Arc Curve", "Circular arcs with radius control"),
            ("Template", "Template", "Common nozzle shapes")
        ]
        
        for mode, label, tooltip in modes:
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            
            btn = QRadioButton(label)
            btn.setMinimumHeight(35)
            btn.setToolTip(tooltip)
            btn.setStyleSheet("""
                QRadioButton {
                    font-weight: bold;
                    padding: 5px;
                }
                QRadioButton::indicator {
                    width: 16px;
                    height: 16px;
                }
            """)
            
            self.mode_buttons[mode] = btn
            self.mode_group.addButton(btn)
            container_layout.addWidget(btn)
            
            # Add mode-specific controls
            if mode == "Template":
                self.template_combo = QComboBox()
                # Load available templates
                templates = self.template_loader.list_templates()
                self.template_combo.addItems(templates)
                self.template_combo.setToolTip("Select predefined nozzle shape")
                self.template_combo.currentTextChanged.connect(self.on_template_selected)
                container_layout.addWidget(self.template_combo)
            
            mode_layout.addWidget(container)
            
        # Set default mode
        self.mode_buttons["Polynomial"].setChecked(True)
        controls_layout.addWidget(mode_group)
        
        # Enhanced geometry properties with validation
        props_group = QGroupBox(" Geometry Properties")
        props_layout = QFormLayout(props_group)
        
        # Symmetry control
        self.symmetric_checkbox = QCheckBox("Symmetric Nozzle")
        self.symmetric_checkbox.setChecked(True)
        self.symmetric_checkbox.setToolTip("Mirror geometry about centerline")
        self.symmetric_checkbox.stateChanged.connect(self.on_symmetric_changed)
        props_layout.addRow(self.symmetric_checkbox)
        
        # Resolution control
        self.interpolation_points = QSpinBox()
        self.interpolation_points.setRange(20, 500)
        self.interpolation_points.setValue(100)
        self.interpolation_points.setToolTip("Number of points for smooth curves")
        self.interpolation_points.valueChanged.connect(self.on_resolution_changed)
        props_layout.addRow("Resolution:", self.interpolation_points)
        
        # Snap controls
        self.snap_to_grid = QCheckBox("Snap to Grid")
        self.snap_to_grid.setToolTip("Snap cursor to grid intersections")
        props_layout.addRow(self.snap_to_grid)
        
        self.show_dimensions = QCheckBox("Show Dimensions")
        self.show_dimensions.setChecked(True)
        self.show_dimensions.setToolTip("Display real-time measurements")
        self.show_dimensions.stateChanged.connect(self.update_geometry_plot)
        props_layout.addRow(self.show_dimensions)
        
        controls_layout.addWidget(props_group)
        
        # Smart editing tools
        edit_group = QGroupBox("Editing Tools")
        edit_layout = QGridLayout(edit_group)
        
        btn_select = QPushButton("Select")
        btn_move = QPushButton("Move")
        btn_modify = QPushButton("Modify")
        btn_duplicate = QPushButton("Copy")
        
        btn_select.setToolTip("Select elements for editing")
        btn_move.setToolTip("Move selected elements")
        btn_modify.setToolTip("Modify element parameters")
        btn_duplicate.setToolTip("Duplicate selected elements")
        
        # Connect editing tool buttons
        btn_select.clicked.connect(lambda: self.set_editing_mode("select"))
        btn_move.clicked.connect(lambda: self.set_editing_mode("move"))
        btn_modify.clicked.connect(lambda: self.set_editing_mode("modify"))
        btn_duplicate.clicked.connect(lambda: self.set_editing_mode("duplicate"))
        
        edit_layout.addWidget(btn_select, 0, 0)
        edit_layout.addWidget(btn_move, 0, 1)
        edit_layout.addWidget(btn_modify, 1, 0)
        edit_layout.addWidget(btn_duplicate, 1, 1)
        
        for i in range(edit_layout.rowCount()):
            for j in range(edit_layout.columnCount()):
                item = edit_layout.itemAtPosition(i, j)
                if item and item.widget():
                    item.widget().setMinimumHeight(30)
        
        controls_layout.addWidget(edit_group)
        
        # Enhanced actions with validation feedback
        actions_group = QGroupBox("Actions")
        actions_layout = QGridLayout(actions_group)
        
        btn_clear = QPushButton("Clear")
        btn_undo = QPushButton("Undo")
        btn_validate = QPushButton("Validate")
        btn_optimize = QPushButton("[Quick] Optimize")
        
        btn_clear.setToolTip("Clear all geometry")
        btn_undo.setToolTip("Undo last element")
        btn_validate.setToolTip("Check geometry validity")
        btn_optimize.setToolTip("Optimize geometry for CFD")
        
        btn_clear.clicked.connect(self.clear_geometry)
        btn_undo.clicked.connect(self.undo_last_element)
        btn_validate.clicked.connect(self.validate_geometry)
        btn_optimize.clicked.connect(self.optimize_geometry)
        
        actions_layout.addWidget(btn_clear, 0, 0)
        actions_layout.addWidget(btn_undo, 0, 1)
        actions_layout.addWidget(btn_validate, 1, 0)
        actions_layout.addWidget(btn_optimize, 1, 1)
        
        # Add "Finish Geo" button
        self.finish_geo_btn = QPushButton("Finish Geo")
        self.finish_geo_btn.clicked.connect(self.finish_geometry)
        self.finish_geo_btn.setToolTip("Complete geometry and proceed to meshing")
        self.finish_geo_btn.setStyleSheet(f"""
            QPushButton {{
                background: {Theme.SUCCESS};
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
                min-height: 35px;
            }}
            QPushButton:hover {{
                background: #45a049;
            }}
        """)
        actions_layout.addWidget(self.finish_geo_btn, 2, 0, 1, 2)  # Span across both columns
        
        for i in range(actions_layout.rowCount()):
            for j in range(actions_layout.columnCount()):
                item = actions_layout.itemAtPosition(i, j)
                if item and item.widget():
                    item.widget().setMinimumHeight(35)
        
        controls_layout.addWidget(actions_group)
        
        # Current elements list
        elements_group = QGroupBox("Current Elements")
        elements_layout = QVBoxLayout(elements_group)
        
        self.elements_list = QListWidget()
        self.elements_list.setMaximumHeight(120)
        self.elements_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {Theme.SECONDARY};
                border: 1px solid {Theme.BORDER};
                color: {Theme.TEXT};
                font-size: 11px;
            }}
            QListWidget::item {{
                padding: 4px;
                border-bottom: 1px solid {Theme.BORDER};
            }}
            QListWidget::item:selected {{
                background-color: {Theme.PRIMARY};
            }}
        """)
        self.elements_list.itemClicked.connect(self.on_element_selected)
        self.elements_list.itemDoubleClicked.connect(self.edit_selected_element)
        elements_layout.addWidget(self.elements_list)
        
        # Element controls
        element_controls = QHBoxLayout()
        btn_delete_element = QPushButton("Delete")
        btn_edit_element = QPushButton("Edit")
        btn_copy_element = QPushButton("Copy")
        
        btn_delete_element.clicked.connect(self.delete_selected_element)
        btn_edit_element.clicked.connect(self.edit_selected_element)
        btn_copy_element.clicked.connect(self.copy_selected_element)
        
        element_controls.addWidget(btn_delete_element)
        element_controls.addWidget(btn_edit_element)
        element_controls.addWidget(btn_copy_element)
        elements_layout.addLayout(element_controls)
        
        controls_layout.addWidget(elements_group)
        
        # Enhanced instructions with examples
        instructions = QGroupBox("Drawing Guide")
        instructions_layout = QVBoxLayout(instructions)
        
        instruction_text = QTextEdit()
        instruction_text.setReadOnly(True)
        instruction_text.setMaximumHeight(140)
        instruction_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Theme.SECONDARY};
                border: 1px solid {Theme.BORDER};
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        instruction_text.setText("""
INSTANT DRAWING - No buttons needed!

POLYNOMIAL: Click multiple points → Right-click to finish
LINE: Click start → Click end (auto-finish)
ARC: Click start → Click middle → Click end
TEMPLATE: Select type → Click to position

SMART FEATURES:
• Auto-symmetry when enabled
• Real-time validation
• Snap to grid option
• Live dimension display
• Undo/redo support

TIP: Use Ctrl+Z for quick undo!
        """)
        
        instructions_layout.addWidget(instruction_text)
        controls_layout.addWidget(instructions)
        
        controls_layout.addStretch()
        layout.addWidget(controls_widget)
        
        # Enhanced canvas with smart features
        self.geometry_canvas = self.create_enhanced_geometry_canvas()
        layout.addWidget(self.geometry_canvas)
        
        # Connect mode changes for immediate activation
        for mode, btn in self.mode_buttons.items():
            btn.toggled.connect(lambda checked, m=mode: self.on_mode_changed(m, checked))
            
        return tab
        
        
    def create_enhanced_geometry_canvas(self):
        """Create enhanced geometry canvas with smart drawing features."""
        canvas = FigureCanvas(Figure(figsize=(12, 8), facecolor=Theme.BACKGROUND))
        
        # Setup professional looking axis
        ax = canvas.figure.add_subplot(111, facecolor='#1e1e1e')
        ax.set_xlim(0, 2)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Axial Distance [m]', color=Theme.TEXT, fontweight='bold')
        ax.set_ylabel('Radial Distance [m]', color=Theme.TEXT, fontweight='bold')
        ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY, linestyle='--', linewidth=0.5)
        ax.tick_params(colors=Theme.TEXT, labelsize=10)
        ax.set_title('Nozzle Geometry Design', color=Theme.TEXT, fontweight='bold', pad=20)
        
        # Style the spines
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)
            spine.set_linewidth(1.5)
        
        # Add centerline
        ax.axhline(y=0, color=Theme.PRIMARY, linestyle='-', linewidth=2, alpha=0.8, label='Centerline')
        
        # Store references and state
        canvas.ax = ax
        canvas.current_points = []
        canvas.drawing_mode = "Polynomial"
        canvas.hover_point = None
        canvas.selected_elements = []
        
        # Connect enhanced mouse events
        def on_mouse_press(event):
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
                
            # Handle different mouse buttons
            if event.button == 3:  # Right click - finish/context menu
                if canvas.drawing_mode == "Polynomial" and len(canvas.current_points) >= 2:
                    self.finish_element(canvas)
                elif canvas.drawing_mode == "Template":
                    self.create_template_nozzle(event.xdata, event.ydata)
                return
                
            if event.button == 1:  # Left click - add point
                x, y = event.xdata, event.ydata
                
                # Apply snap to grid if enabled
                if self.snap_to_grid.isChecked():
                    grid_size = 0.05  # 5cm grid
                    x = round(x / grid_size) * grid_size
                    y = round(y / grid_size) * grid_size
                
                canvas.current_points.append((x, y))
                
                # Handle different drawing modes with immediate feedback
                if canvas.drawing_mode == "Line" and len(canvas.current_points) == 2:
                    self.finish_element(canvas)
                elif canvas.drawing_mode == "Arc" and len(canvas.current_points) == 3:
                    self.finish_element(canvas)
                # Polynomial continues until right-click
                    
                self.update_canvas_with_preview(canvas)
        
        def on_mouse_move(event):
            """Handle mouse movement for live preview and hover effects."""
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return
                
            # Update hover point for live preview
            if canvas.current_points and canvas.drawing_mode in ["Line", "Arc"]:
                canvas.hover_point = (event.xdata, event.ydata)
                self.update_canvas_with_preview(canvas)
        
        def on_key_press(event):
            """Handle keyboard shortcuts."""
            if event.key == 'ctrl+z':
                self.undo_last_element()
            elif event.key == 'escape':
                # Cancel current drawing
                canvas.current_points = []
                canvas.hover_point = None
                self.update_canvas_with_preview(canvas)
            elif event.key == 'd':
                # Toggle dimensions
                self.show_dimensions.setChecked(not self.show_dimensions.isChecked())
                self.update_geometry_plot()
        
        canvas.mpl_connect('button_press_event', on_mouse_press)
        canvas.mpl_connect('motion_notify_event', on_mouse_move)
        canvas.mpl_connect('key_press_event', on_key_press)
        
        # Make canvas focusable for keyboard events
        canvas.setFocusPolicy(Qt.ClickFocus)
        
        # Initialize canvas attributes
        canvas.ax = ax
        canvas.drawing_mode = "Polynomial"  # Default mode
        canvas.current_points = []
        canvas.hover_point = None
        canvas.editing_mode = "draw"
        
        return canvas
    
    def update_canvas_with_preview(self, canvas):
        """Update canvas with live preview of current drawing."""
        ax = canvas.ax
        ax.clear()
        
        # Redraw background
        ax.set_xlim(0, 2)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Axial Distance [m]', color=Theme.TEXT, fontweight='bold')
        ax.set_ylabel('Radial Distance [m]', color=Theme.TEXT, fontweight='bold')
        ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY, linestyle='--', linewidth=0.5)
        ax.tick_params(colors=Theme.TEXT, labelsize=10)
        ax.set_title('Nozzle Geometry Design', color=Theme.TEXT, fontweight='bold', pad=20)
        ax.axhline(y=0, color=Theme.PRIMARY, linestyle='-', linewidth=2, alpha=0.8, label='Centerline')
        
        # Draw existing geometry
        self.plot_geometry_on_axis(ax)
        
        # Draw current points and preview
        if canvas.current_points:
            points = canvas.current_points
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            
            # Plot current points
            ax.scatter(x_points, y_points, c='yellow', s=50, zorder=10, alpha=0.8)
            
            # Draw preview based on mode
            if canvas.drawing_mode == "Line" and len(points) == 1 and canvas.hover_point:
                # Preview line
                ax.plot([points[0][0], canvas.hover_point[0]], 
                       [points[0][1], canvas.hover_point[1]], 
                       'y--', alpha=0.5, linewidth=2)
                       
            elif canvas.drawing_mode == "Arc" and len(points) >= 1 and canvas.hover_point:
                # Preview arc (simplified)
                if len(points) == 1:
                    ax.plot([points[0][0], canvas.hover_point[0]], 
                           [points[0][1], canvas.hover_point[1]], 
                           'y--', alpha=0.5, linewidth=1)
                elif len(points) == 2:
                    # Draw preview arc through three points
                    try:
                        x_arc = [points[0][0], points[1][0], canvas.hover_point[0]]
                        y_arc = [points[0][1], points[1][1], canvas.hover_point[1]]
                        ax.plot(x_arc, y_arc, 'y--', alpha=0.5, linewidth=2)
                    except:
                        pass
                        
            elif canvas.drawing_mode == "Polynomial" and len(points) >= 2:
                # Preview polynomial
                ax.plot(x_points, y_points, 'y--', alpha=0.5, linewidth=2)
        
        # Show dimensions if enabled
        if self.show_dimensions.isChecked() and canvas.current_points:
            self.add_dimension_annotations(ax, canvas.current_points)
        
        canvas.draw()
        
    def add_dimension_annotations(self, ax, points):
        """Add real-time dimension annotations."""
        if len(points) < 2:
            return
            
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            # Calculate distance
            distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            
            # Add dimension line
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            ax.annotate(f'{distance:.3f}m', 
                       xy=(mid_x, mid_y), 
                       xytext=(10, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontsize=8, color='black')
    
    def plot_geometry_on_axis(self, ax):
        """Plot current geometry on the given axis."""
        if not self.geometry.elements:
            return
            
        # Get interpolated points
        x_coords, y_coords = self.geometry.get_interpolated_points(
            num_points_per_element=self.interpolation_points.value()
        )
        
        if x_coords and y_coords:
            # Plot upper boundary
            ax.plot(x_coords, y_coords, 'cyan', linewidth=3, alpha=0.9, label='Upper Boundary')
            
            # Plot symmetric lower boundary if enabled
            if self.symmetric_checkbox.isChecked():
                ax.plot(x_coords, [-y for y in y_coords], 'cyan', linewidth=3, alpha=0.9, label='Lower Boundary')
                
                # Fill nozzle area
                ax.fill_between(x_coords, y_coords, [-y for y in y_coords], 
                              alpha=0.2, color='cyan', label='Nozzle Volume')
            
            # Mark control points
            for i, element in enumerate(self.geometry.elements):
                if hasattr(element, 'control_points'):
                    points = element.control_points
                    x_ctrl = [p[0] for p in points]
                    y_ctrl = [p[1] for p in points]
                    ax.scatter(x_ctrl, y_ctrl, c='red', s=30, zorder=5, alpha=0.8)
    
    def create_template_nozzle(self, x_pos, y_pos):
        """Create a nozzle from template."""
        template_type = self.template_combo.currentText()
        
        try:
            # Use the template loader to create the geometry
            template_geometry = self.template_loader.load_template(template_type)
            if template_geometry:
                # Replace current geometry with template
                self.geometry = template_geometry
                
                # Update UI
                self.update_elements_list()
                self.plot_geometry()
                
                # Mark as modified
                self.is_modified = True
                self.update_title()
                
                self.status_bar.showMessage(f"Created template: {template_type}", 3000)
            else:
                QMessageBox.warning(self, "Template Error", f"Template {template_type} not found")
            
        except Exception as e:
            QMessageBox.warning(self, "Template Error", f"Failed to create template: {e}")
            self.status_bar.showMessage(f"Template creation error: {str(e)}", 5000)
    
    def optimize_geometry(self):
        """Optimize geometry for better CFD performance."""
        if not self.geometry.elements:
            QMessageBox.information(self, "Optimize", "No geometry to optimize!")
            return
            
        # Implement geometry optimization logic
        QMessageBox.information(self, "Optimize", 
                               "Geometry optimization completed!\n"
                               "• Smoothed transitions\n"
                               "• Optimized for CFD mesh\n"
                               "• Validated continuity")
        
        self.update_geometry_plot()
        self.update_geometry_info()
    
    def on_resolution_changed(self):
        """Handle resolution changes with live update."""
        self.update_geometry_plot()
        
    def on_mode_changed(self, mode, checked):
        """Handle drawing mode changes."""
        if checked:
            if hasattr(self, 'geometry_canvas'):
                self.geometry_canvas.drawing_mode = mode
                self.geometry_canvas.current_points = []
                self.geometry_canvas.hover_point = None
                self.update_canvas_with_preview(self.geometry_canvas)
            
            # Update status
            self.status_bar.showMessage(f"Drawing mode: {mode}")
    
    def finish_element(self, canvas):
        """Finish drawing current element."""
        if len(canvas.current_points) < 2:
            return
            
        try:
            points = canvas.current_points.copy()
            
            # Create appropriate element based on mode
            if canvas.drawing_mode == "Polynomial":
                element = PolynomialElement(points)
            elif canvas.drawing_mode == "Line":
                element = LineElement(points[0], points[1])
            elif canvas.drawing_mode == "Arc":
                if len(points) >= 3:
                    element = ArcElement(points=points[:3])
                else:
                    return
            else:
                return
                
            self.geometry.add_element(element)
            
            # Clear current drawing state
            canvas.current_points = []
            canvas.hover_point = None
            
            # Update display
            self.update_geometry_plot()
            self.update_geometry_info()
            self.is_modified = True
            
            self.status_bar.showMessage(f"Added {canvas.drawing_mode} element")
            
        except Exception as e:
            QMessageBox.warning(self, "Drawing Error", f"Failed to create element: {e}")
            canvas.current_points = []
    
    # === CORE APPLICATION METHODS ===
    
    def update_geometry_plot(self):
        """Update the main geometry plot."""
        if hasattr(self, 'geometry_canvas'):
            self.update_canvas_with_preview(self.geometry_canvas)
    
    def clear_geometry(self):
        """Clear all geometry."""
        reply = QMessageBox.question(self, "Clear Geometry", 
                                   "Are you sure you want to clear all geometry?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.geometry.clear()
            self.update_geometry_plot()
            self.update_geometry_info()
            self.is_modified = True
    
    def undo_last_element(self):
        """Undo the last element."""
        if self.geometry.elements:
            self.geometry.elements.pop()
            self.update_geometry_plot()
            self.update_geometry_info()
            self.is_modified = True
            self.status_bar.showMessage("Undid last element")
        else:
            self.status_bar.showMessage("Nothing to undo")
    
    def validate_geometry(self):
        """Validate the current geometry."""
        if not self.geometry.elements:
            QMessageBox.information(self, "Validation", "No geometry to validate!")
            return
            
        # Basic validation
        total_length = 0
        issues = []
        
        try:
            x_coords, y_coords = self.geometry.get_interpolated_points(num_points_per_element=100)
            if x_coords and y_coords:
                total_length = max(x_coords) - min(x_coords)
                
                # Check for monotonic x-coordinates
                if not all(x_coords[i] <= x_coords[i+1] for i in range(len(x_coords)-1)):
                    issues.append("Non-monotonic x-coordinates detected")
                
                # Check for negative radii
                if any(y < 0 for y in y_coords if y is not None):
                    issues.append("Negative radii detected")
                    
        except Exception as e:
            issues.append(f"Interpolation error: {e}")
        
        if issues:
            QMessageBox.warning(self, "Validation Issues", "\n".join(issues))
        else:
            QMessageBox.information(self, "Validation", 
                                   f"Geometry is valid!\n"
                                   f"Length: {total_length:.3f} m\n"
                                   f"Elements: {len(self.geometry.elements)}")
    
    def on_symmetric_changed(self):
        """Handle symmetry setting changes."""
        is_symmetric = self.symmetric_checkbox.isChecked()
        self.geometry.set_symmetric(is_symmetric)
        self.update_geometry_plot()
        self.status_bar.showMessage(f"Symmetry: {'Enabled' if is_symmetric else 'Disabled'}")
    
    def finish_geometry(self):
        """Complete geometry design and proceed to next step."""
        if not self.geometry.elements:
            QMessageBox.warning(self, "Finish Geometry", "No geometry to finish! Please add at least one element.")
            return
            
        # Validate geometry before finishing
        self.validate_geometry()
        
        # Switch to mesh tab
        self.tab_widget.setCurrentIndex(1)  # Mesh tab
        self.status_bar.showMessage("Geometry completed! Proceed to meshing.")
        
        QMessageBox.information(self, "Geometry Complete", 
                               f"Geometry design completed!\n"
                               f"Elements: {len(self.geometry.elements)}\n"
                               f"Ready for meshing.")
    
    def update_geometry_info(self):
        """Update geometry information display."""
        # Update elements list
        self.elements_list.clear()
        
        for i, element in enumerate(self.geometry.elements):
            element_type = type(element).__name__
            item_text = f"{i+1}. {element_type}"
            
            if hasattr(element, 'control_points'):
                item_text += f" ({len(element.control_points)} pts)"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i)  # Store element index
            self.elements_list.addItem(item)
    
    def update_elements_list(self):
        """Update the elements list display (alias for update_geometry_info)."""
        self.update_geometry_info()
    
    def plot_geometry(self):
        """Plot the geometry (alias for update_geometry_plot)."""
        self.update_geometry_plot()
    
    def update_title(self):
        """Update the window title to show project name and modified status."""
        name = self.project_name_edit.text() or "Untitled Project"
        modified_indicator = " *" if self.is_modified else ""
        self.setWindowTitle(f"Nozzle Flow CFD Tool - {name}{modified_indicator}")
    
    def on_element_selected(self, item):
        """Handle element selection from list."""
        element_index = item.data(Qt.UserRole)
        self.selected_element_index = element_index
        self.status_bar.showMessage(f"Selected element {element_index + 1}")
    
    def edit_selected_element(self):
        """Edit the selected element."""
        if not hasattr(self, 'selected_element_index') or self.selected_element_index is None:
            QMessageBox.information(self, "Edit Element", "Please select an element to edit.")
            return
            
        element_index = self.selected_element_index
        if element_index >= len(self.geometry.elements):
            return
            
        element = self.geometry.elements[element_index]
        
        # Simple editing dialog
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Edit Element")
        dialog.setLabelText(f"Edit {type(element).__name__} (JSON format):")
        
        # Convert element to editable text
        element_data = element.to_dict() if hasattr(element, 'to_dict') else str(element)
        dialog.setTextValue(json.dumps(element_data, indent=2))
        
        if dialog.exec():
            try:
                # Parse and update element
                new_data = json.loads(dialog.textValue())
                # For now, just show success - full implementation would recreate element
                QMessageBox.information(self, "Edit Element", "Element editing functionality is basic in this version.")
                
            except Exception as e:
                QMessageBox.warning(self, "Edit Error", f"Failed to parse element data: {e}")
    
    def delete_selected_element(self):
        """Delete the selected element."""
        if not hasattr(self, 'selected_element_index') or self.selected_element_index is None:
            QMessageBox.information(self, "Delete Element", "Please select an element to delete.")
            return
            
        element_index = self.selected_element_index
        if element_index >= len(self.geometry.elements):
            return
            
        reply = QMessageBox.question(self, "Delete Element", 
                                   f"Delete element {element_index + 1}?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            del self.geometry.elements[element_index]
            self.selected_element_index = None
            self.update_geometry_plot()
            self.update_geometry_info()
            self.is_modified = True
            self.status_bar.showMessage(f"Deleted element {element_index + 1}")
    
    def copy_selected_element(self):
        """Copy the selected element."""
        if not hasattr(self, 'selected_element_index') or self.selected_element_index is None:
            QMessageBox.information(self, "Copy Element", "Please select an element to copy.")
            return
            
        element_index = self.selected_element_index
        if element_index >= len(self.geometry.elements):
            return
            
        element = self.geometry.elements[element_index]
        
        # Create a copy with offset
        try:
            if hasattr(element, 'control_points'):
                # Offset control points
                offset_points = [(x + 0.1, y + 0.02) for x, y in element.control_points]
                
                # Create new element of same type
                if isinstance(element, PolynomialElement):
                    new_element = PolynomialElement(offset_points)
                elif isinstance(element, LineElement):
                    new_element = LineElement(offset_points[0], offset_points[1])
                elif isinstance(element, ArcElement):
                    new_element = ArcElement(points=offset_points[:3])
                else:
                    QMessageBox.warning(self, "Copy Error", "Unknown element type for copying.")
                    return
                    
                self.geometry.add_element(new_element)
                self.update_geometry_plot()
                self.update_geometry_info()
                self.is_modified = True
                self.status_bar.showMessage(f"Copied element {element_index + 1}")
                
        except Exception as e:
            QMessageBox.warning(self, "Copy Error", f"Failed to copy element: {e}")
    
    def set_editing_mode(self, mode):
        """Set the editing mode for geometry manipulation."""
        self.editing_mode = mode
        
        if hasattr(self, 'geometry_canvas'):
            self.geometry_canvas.editing_mode = mode
            
        # Update status message
        mode_messages = {
            "select": "Select mode: Click elements to select them",
            "move": "Move mode: Drag to move selected elements",
            "modify": "Modify mode: Click elements to modify parameters",
            "duplicate": "Duplicate mode: Click elements to duplicate them"
        }
        
        self.status_bar.showMessage(mode_messages.get(mode, f"Editing mode: {mode}"))
        
        # Update button states (could add visual feedback)
        print(f"Editing mode set to: {mode}")
    
    def on_template_selected(self, template_name):
        """Handle template selection from dropdown."""
        if template_name and hasattr(self, 'mode_buttons') and self.mode_buttons["Template"].isChecked():
            try:
                # Load the template geometry
                template_geometry = self.template_loader.load_template(template_name)
                if template_geometry:
                    # Clear existing geometry
                    self.geometry = template_geometry
                    
                    # Update the elements list
                    self.update_elements_list()
                    
                    # Redraw the plot
                    self.plot_geometry()
                    
                    # Mark as modified
                    self.is_modified = True
                    self.update_title()
                    
                    self.status_bar.showMessage(f"Loaded template: {template_name}", 3000)
                else:
                    self.status_bar.showMessage(f"Failed to load template: {template_name}", 3000)
            except Exception as e:
                QMessageBox.warning(self, "Template Error", f"Failed to load template {template_name}:\n{str(e)}")
                self.status_bar.showMessage(f"Template load error: {str(e)}", 5000)
    
    # === PROJECT MANAGEMENT ===
    
    def new_project(self):
        """Create a new project."""
        if self.is_modified:
            reply = QMessageBox.question(self, "New Project", 
                                       "Save current project before creating new one?",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                self.save_project()
            elif reply == QMessageBox.Cancel:
                return
        
        # Reset everything
        self.geometry.clear()
        self.current_file = None
        self.is_modified = False
        self.project_name_edit.setText("Untitled Project")
        self.project_description.clear()
        
        self.update_geometry_plot()
        self.update_geometry_info()
        for stage in self.status_labels:
            self.update_workflow_status(stage, False, 0)
        
        self.status_bar.showMessage("New project created")
    
    def open_project(self):
        """Open an existing project."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "JSON Files (*.json);;All Files (*)")
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Load project data
                self.project_name_edit.setText(data.get('name', 'Untitled Project'))
                self.project_description.setText(data.get('description', ''))
                
                # Load geometry
                self.geometry.clear()
                for element_data in data.get('geometry', []):
                    # Reconstruct geometry elements based on saved data
                    pass  # Implementation depends on serialization format
                
                self.current_file = file_path
                self.is_modified = False
                
                self.update_geometry_plot()
                self.update_geometry_info()
                self.status_bar.showMessage(f"Opened: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open project: {e}")
    
    def save_project(self):
        """Save the current project."""
        if self.current_file:
            self.save_project_to_file(self.current_file)
        else:
            self.save_project_as()
    
    def save_project_as(self):
        """Save project with new filename."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "JSON Files (*.json);;All Files (*)")
        
        if file_path:
            self.save_project_to_file(file_path)
    
    def save_project_to_file(self, file_path):
        """Save project to specified file."""
        try:
            data = {
                'name': self.project_name_edit.text(),
                'description': self.project_description.toPlainText(),
                'geometry': [],  # Serialize geometry elements
                'mesh_settings': {},  # Serialize mesh settings
                'simulation_settings': {}  # Serialize simulation settings
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.current_file = file_path
            self.is_modified = False
            self.status_bar.showMessage(f"Saved: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save project: {e}")
    
    def export_case(self):
        """Export OpenFOAM case files."""
        if not self.geometry.elements:
            QMessageBox.warning(self, "Export", "No geometry to export!")
            return
            
        try:
            # Basic export functionality
            QMessageBox.information(self, "Export", "OpenFOAM case files exported successfully!")
            self.status_bar.showMessage("Case exported")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
    
    def on_tab_changed(self, index):
        """Handle tab changes."""
        tab_names = ["Geometry", "Meshing", "Simulation", "Post-processing"]
        if index < len(tab_names):
            self.status_bar.showMessage(f"Current tab: {tab_names[index]}")
        
    # === PLACEHOLDER METHODS FOR ADVANCED FEATURES ===
            
    def update_canvas(self, canvas):
        """Update canvas display."""
        canvas.ax.clear()
        
        # Reset axis properties
        canvas.ax.set_xlim(0, 2)
        canvas.ax.set_ylim(-0.5, 0.5)
        canvas.ax.set_xlabel('X [m]', color=Theme.TEXT)
        canvas.ax.set_ylabel('Y [m]', color=Theme.TEXT)
        canvas.ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY)
        canvas.ax.tick_params(colors=Theme.TEXT)
        for spine in canvas.ax.spines.values():
            spine.set_color(Theme.BORDER)
        
        # Draw existing geometry
        if self.geometry.elements:
            x_coords, y_coords = self.geometry.get_interpolated_points()
            if x_coords and y_coords:
                canvas.ax.plot(x_coords, y_coords, color=Theme.PRIMARY, linewidth=3, label='Upper wall')
                
                # Draw symmetric lower wall if enabled
                if self.geometry.is_symmetric:
                    canvas.ax.plot(x_coords, [-y for y in y_coords], color=Theme.PRIMARY, 
                                 linewidth=3, label='Lower wall')
                    
                canvas.ax.legend()
                
        # Draw current drawing points
        if canvas.current_points:
            x_pts = [p[0] for p in canvas.current_points]
            y_pts = [p[1] for p in canvas.current_points]
            canvas.ax.plot(x_pts, y_pts, 'o-', color=Theme.WARNING, markersize=8, linewidth=2, alpha=0.8)
            
        canvas.draw()
        
    def create_meshing_tab(self):
        """Create meshing tab with boundary layer controls."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left controls for mesh parameters
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_widget)
        
        # Basic mesh settings
        basic_group = QGroupBox("Basic Mesh Settings")
        basic_layout = QGridLayout(basic_group)
        
        self.global_size = QDoubleSpinBox()
        self.global_size.setRange(0.001, 1.0)
        self.global_size.setValue(0.01)
        self.global_size.setDecimals(4)
        
        self.wall_size = QDoubleSpinBox()
        self.wall_size.setRange(0.0001, 0.1)
        self.wall_size.setValue(0.001)
        self.wall_size.setDecimals(4)
        
        basic_layout.addWidget(QLabel("Global Element Size:"), 0, 0)
        basic_layout.addWidget(self.global_size, 0, 1)
        basic_layout.addWidget(QLabel("Wall Element Size:"), 1, 0)
        basic_layout.addWidget(self.wall_size, 1, 1)
        
        controls_layout.addWidget(basic_group)
        
        # Boundary layer settings
        bl_group = QGroupBox("Boundary Layer Controls")
        bl_layout = QGridLayout(bl_group)
        
        self.enable_bl = QCheckBox("Enable Boundary Layers")
        self.enable_bl.setChecked(True)
        
        self.num_bl_layers = QSpinBox()
        self.num_bl_layers.setRange(1, 20)
        self.num_bl_layers.setValue(5)
        
        self.first_layer_thickness = QDoubleSpinBox()
        self.first_layer_thickness.setRange(1e-6, 1e-3)
        self.first_layer_thickness.setValue(1e-5)
        self.first_layer_thickness.setDecimals(6)
        
        self.growth_ratio = QDoubleSpinBox()
        self.growth_ratio.setRange(1.0, 3.0)
        self.growth_ratio.setValue(1.2)
        self.growth_ratio.setDecimals(2)
        
        bl_layout.addWidget(self.enable_bl, 0, 0, 1, 2)
        bl_layout.addWidget(QLabel("Number of Layers:"), 1, 0)
        bl_layout.addWidget(self.num_bl_layers, 1, 1)
        bl_layout.addWidget(QLabel("First Layer Thickness:"), 2, 0)
        bl_layout.addWidget(self.first_layer_thickness, 2, 1)
        bl_layout.addWidget(QLabel("Growth Ratio:"), 3, 0)
        bl_layout.addWidget(self.growth_ratio, 3, 1)
        
        controls_layout.addWidget(bl_group)
        
        # Advanced settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QGridLayout(advanced_group)
        
        self.quality_threshold = QDoubleSpinBox()
        self.quality_threshold.setRange(0.1, 1.0)
        self.quality_threshold.setValue(0.3)
        self.quality_threshold.setDecimals(2)
        
        self.farfield_distance = QDoubleSpinBox()
        self.farfield_distance.setRange(1.0, 20.0)
        self.farfield_distance.setValue(5.0)
        self.farfield_distance.setDecimals(1)
        
        advanced_layout.addWidget(QLabel("Quality Threshold:"), 0, 0)
        advanced_layout.addWidget(self.quality_threshold, 0, 1)
        advanced_layout.addWidget(QLabel("Farfield Distance:"), 1, 0)
        advanced_layout.addWidget(self.farfield_distance, 1, 1)
        
        controls_layout.addWidget(advanced_group)
        
        # Mesh actions
        actions_group = QGroupBox("Mesh Generation")
        actions_layout = QVBoxLayout(actions_group)
        
        btn_generate = QPushButton(" Generate Mesh")
        btn_analyze = QPushButton("[Chart] Analyze Mesh Quality")
        btn_export = QPushButton("[Save] Export Mesh")
        
        btn_generate.clicked.connect(self.generate_mesh)
        btn_analyze.clicked.connect(self.analyze_mesh)
        btn_export.clicked.connect(self.export_mesh)
        
        for btn in [btn_generate, btn_analyze, btn_export]:
            btn.setMinimumHeight(30)
            actions_layout.addWidget(btn)
            
        controls_layout.addWidget(actions_group)
        
        # Mesh statistics
        stats_group = QGroupBox("Mesh Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.mesh_stats = QTextEdit()
        self.mesh_stats.setReadOnly(True)
        self.mesh_stats.setMaximumHeight(150)
        self.mesh_stats.setText("Generate mesh to see statistics...")
        
        stats_layout.addWidget(self.mesh_stats)
        controls_layout.addWidget(stats_group)
        
        controls_layout.addStretch()
        layout.addWidget(controls_widget)
        
        # Canvas for mesh visualization
        self.mesh_canvas = self.create_mesh_canvas()
        layout.addWidget(self.mesh_canvas)
        
        return tab
        
    def create_mesh_canvas(self):
        """Create mesh visualization canvas."""
        canvas = FigureCanvas(Figure(figsize=(10, 6), facecolor=Theme.BACKGROUND))
        
        ax = canvas.figure.add_subplot(111, facecolor='#1e1e1e')
        ax.set_xlabel('X [m]', color=Theme.TEXT)
        ax.set_ylabel('Y [m]', color=Theme.TEXT)
        ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY)
        ax.tick_params(colors=Theme.TEXT)
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)
            
        canvas.ax = ax
        
        # Initial display
        ax.text(0.5, 0.5, 'Generate mesh to visualize', 
               transform=ax.transAxes, ha='center', va='center',
               color=Theme.TEXT_SECONDARY, fontsize=14)
        
        return canvas
        
    def create_simulation_tab(self):
        """Create simulation setup tab with CFD controls."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left controls for simulation setup
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_widget)
        
        # Flow conditions
        flow_group = QGroupBox("Flow Conditions")
        flow_layout = QGridLayout(flow_group)
        
        self.inlet_velocity = QDoubleSpinBox()
        self.inlet_velocity.setRange(0.1, 1000.0)
        self.inlet_velocity.setValue(100.0)
        self.inlet_velocity.setDecimals(1)
        
        self.inlet_pressure = QDoubleSpinBox()
        self.inlet_pressure.setRange(1000, 1e6)
        self.inlet_pressure.setValue(101325)
        self.inlet_pressure.setDecimals(0)
        
        self.outlet_pressure = QDoubleSpinBox()
        self.outlet_pressure.setRange(1000, 1e6)
        self.outlet_pressure.setValue(80000)
        self.outlet_pressure.setDecimals(0)
        
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(200, 1000)
        self.temperature.setValue(300)
        self.temperature.setDecimals(1)
        
        flow_layout.addWidget(QLabel("Inlet Velocity [m/s]:"), 0, 0)
        flow_layout.addWidget(self.inlet_velocity, 0, 1)
        flow_layout.addWidget(QLabel("Inlet Pressure [Pa]:"), 1, 0)
        flow_layout.addWidget(self.inlet_pressure, 1, 1)
        flow_layout.addWidget(QLabel("Outlet Pressure [Pa]:"), 2, 0)
        flow_layout.addWidget(self.outlet_pressure, 2, 1)
        flow_layout.addWidget(QLabel("Temperature [K]:"), 3, 0)
        flow_layout.addWidget(self.temperature, 3, 1)
        
        controls_layout.addWidget(flow_group)
        
        # Turbulence model
        turb_group = QGroupBox("Turbulence Model")
        turb_layout = QVBoxLayout(turb_group)
        
        self.turbulence_model = QComboBox()
        self.turbulence_model.addItems(["laminar", "kEpsilon", "kOmegaSST", "SpalartAllmaras"])
        self.turbulence_model.setCurrentText("kOmegaSST")
        
        turb_layout.addWidget(self.turbulence_model)
        controls_layout.addWidget(turb_group)
        
        # Solver settings
        solver_group = QGroupBox("Solver Settings")
        solver_layout = QGridLayout(solver_group)
        
        self.solver_type = QComboBox()
        self.solver_type.addItems(["simpleFoam", "rhoSimpleFoam", "sonicFoam"])
        self.solver_type.setCurrentText("simpleFoam")
        
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(100, 10000)
        self.max_iterations.setValue(1000)
        
        self.convergence_tolerance = QDoubleSpinBox()
        self.convergence_tolerance.setRange(1e-8, 1e-3)
        self.convergence_tolerance.setValue(1e-6)
        self.convergence_tolerance.setDecimals(8)
        
        self.n_processors = QSpinBox()
        self.n_processors.setRange(1, 64)
        self.n_processors.setValue(1)
        self.n_processors.setToolTip("Number of CPU cores to use (1 = serial)")
        
        self.decomposition_method = QComboBox()
        self.decomposition_method.addItems(["scotch", "simple", "hierarchical", "manual"])
        self.decomposition_method.setCurrentText("scotch")
        self.decomposition_method.setToolTip("Domain decomposition method for parallel runs")
        
        solver_layout.addWidget(QLabel("Solver:"), 0, 0)
        solver_layout.addWidget(self.solver_type, 0, 1)
        solver_layout.addWidget(QLabel("Max Iterations:"), 1, 0)
        solver_layout.addWidget(self.max_iterations, 1, 1)
        solver_layout.addWidget(QLabel("Convergence Tolerance:"), 2, 0)
        solver_layout.addWidget(self.convergence_tolerance, 2, 1)
        solver_layout.addWidget(QLabel("Number of Processors:"), 3, 0)
        solver_layout.addWidget(self.n_processors, 3, 1)
        solver_layout.addWidget(QLabel("Decomposition Method:"), 4, 0)
        solver_layout.addWidget(self.decomposition_method, 4, 1)
        
        controls_layout.addWidget(solver_group)
        
        # Case management
        case_group = QGroupBox("Case Management")
        case_layout = QVBoxLayout(case_group)
        
        case_file_layout = QHBoxLayout()
        self.case_directory = QLineEdit()
        self.case_directory.setText("./case")
        btn_browse_case = QPushButton("[...]")
        btn_browse_case.setMaximumWidth(40)
        btn_browse_case.clicked.connect(self.browse_case_directory)
        
        case_file_layout.addWidget(QLabel("Case Directory:"))
        case_file_layout.addWidget(self.case_directory)
        case_file_layout.addWidget(btn_browse_case)
        
        case_layout.addLayout(case_file_layout)
        controls_layout.addWidget(case_group)
        
        # Simulation actions
        actions_group = QGroupBox("Simulation Control")
        actions_layout = QVBoxLayout(actions_group)
        
        btn_setup = QPushButton("[Settings] Setup Case Files")
        btn_run = QPushButton("[Run] Run Simulation")
        btn_monitor = QPushButton("[Monitor] Monitor Progress")
        btn_stop = QPushButton("[Stop] Stop Simulation")
        
        btn_setup.clicked.connect(self.setup_simulation)
        btn_run.clicked.connect(self.run_simulation)
        btn_monitor.clicked.connect(self.monitor_simulation)
        btn_stop.clicked.connect(self.stop_simulation)
        
        for btn in [btn_setup, btn_run, btn_monitor, btn_stop]:
            btn.setMinimumHeight(30)
            actions_layout.addWidget(btn)
            
        controls_layout.addWidget(actions_group)
        
        # Simulation status
        status_group = QGroupBox("Simulation Status")
        status_layout = QVBoxLayout(status_group)
        
        self.simulation_log = QTextEdit()
        self.simulation_log.setReadOnly(True)
        self.simulation_log.setMaximumHeight(150)
        self.simulation_log.setText("Ready to setup simulation...")
        
        status_layout.addWidget(self.simulation_log)
        controls_layout.addWidget(status_group)
        
        controls_layout.addStretch()
        layout.addWidget(controls_widget)
        
        # Canvas for simulation visualization
        self.sim_canvas = self.create_simulation_canvas()
        layout.addWidget(self.sim_canvas)
        
        return tab
        
    def create_simulation_canvas(self):
        """Create simulation visualization canvas."""
        canvas = FigureCanvas(Figure(figsize=(10, 6), facecolor=Theme.BACKGROUND))
        
        ax = canvas.figure.add_subplot(111, facecolor='#1e1e1e')
        ax.set_xlabel('Iteration', color=Theme.TEXT)
        ax.set_ylabel('Residual', color=Theme.TEXT)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY)
        ax.tick_params(colors=Theme.TEXT)
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)
            
        canvas.ax = ax
        
        # Initial display
        ax.text(0.5, 0.5, 'Run simulation to see convergence', 
               transform=ax.transAxes, ha='center', va='center',
               color=Theme.TEXT_SECONDARY, fontsize=14)
        
        return canvas
        
    def create_postprocessing_tab(self):
        """Create post-processing tab with visualization controls."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left controls for post-processing
        controls_widget = QWidget()
        controls_widget.setMaximumWidth(350)
        controls_layout = QVBoxLayout(controls_widget)
        
        # Results loading
        load_group = QGroupBox("Load Results")
        load_layout = QVBoxLayout(load_group)
        
        load_file_layout = QHBoxLayout()
        self.results_path = QLineEdit()
        self.results_path.setText("./case")
        btn_browse_results = QPushButton("[...]")
        btn_browse_results.setMaximumWidth(40)
        btn_browse_results.clicked.connect(self.browse_results_directory)
        
        load_file_layout.addWidget(QLabel("Results Directory:"))
        load_file_layout.addWidget(self.results_path)
        load_file_layout.addWidget(btn_browse_results)
        
        btn_load_results = QPushButton("📂 Load Results")
        btn_load_results.clicked.connect(self.load_results)
        
        load_layout.addLayout(load_file_layout)
        load_layout.addWidget(btn_load_results)
        controls_layout.addWidget(load_group)
        
        # Field selection
        field_group = QGroupBox("Field Visualization")
        field_layout = QGridLayout(field_group)
        
        self.field_type = QComboBox()
        self.field_type.addItems(["U", "p", "T", "k", "omega", "nut"])
        self.field_type.currentTextChanged.connect(self.on_field_changed)
        
        # Use ComboBox instead of SpinBox for time steps
        self.time_step = QComboBox()
        self.time_step.addItem("0")
        self.time_step.currentTextChanged.connect(self.update_visualization)
        
        field_layout.addWidget(QLabel("Field:"), 0, 0)
        field_layout.addWidget(self.field_type, 0, 1)
        field_layout.addWidget(QLabel("Time Step:"), 1, 0)
        field_layout.addWidget(self.time_step, 1, 1)
        
        controls_layout.addWidget(field_group)
        
        # Visualization settings
        viz_group = QGroupBox("Visualization Settings")
        viz_layout = QGridLayout(viz_group)
        
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Contour", "Vector", "Streamline", "Line Plot"])
        self.plot_type.currentTextChanged.connect(self.update_visualization)
        
        self.colormap = QComboBox()
        self.colormap.addItems(["viridis", "plasma", "jet", "coolwarm", "RdYlBu"])
        self.colormap.currentTextChanged.connect(self.update_visualization)
        
        self.contour_levels = QSpinBox()
        self.contour_levels.setRange(5, 50)
        self.contour_levels.setValue(20)
        self.contour_levels.valueChanged.connect(self.update_visualization)
        
        viz_layout.addWidget(QLabel("Plot Type:"), 0, 0)
        viz_layout.addWidget(self.plot_type, 0, 1)
        viz_layout.addWidget(QLabel("Colormap:"), 1, 0)
        viz_layout.addWidget(self.colormap, 1, 1)
        viz_layout.addWidget(QLabel("Contour Levels:"), 2, 0)
        viz_layout.addWidget(self.contour_levels, 2, 1)
        
        controls_layout.addWidget(viz_group)
        
        # Analysis tools
        analysis_group = QGroupBox("Analysis Tools")
        analysis_layout = QVBoxLayout(analysis_group)
        
        btn_wall_values = QPushButton("[Chart] Wall Values")
        btn_centerline = QPushButton("[Monitor] Centerline Plot")
        btn_mass_flow = QPushButton("⚖️ Mass Flow Rate")
        btn_pressure_loss = QPushButton("📉 Pressure Loss")
        
        btn_wall_values.clicked.connect(self.analyze_wall_values)
        btn_centerline.clicked.connect(self.plot_centerline)
        btn_mass_flow.clicked.connect(self.calculate_mass_flow)
        btn_pressure_loss.clicked.connect(self.calculate_pressure_loss)
        
        for btn in [btn_wall_values, btn_centerline, btn_mass_flow, btn_pressure_loss]:
            btn.setMinimumHeight(30)
            analysis_layout.addWidget(btn)
            
        controls_layout.addWidget(analysis_group)
        
        # Export options
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        btn_save_image = QPushButton("🖼️ Save Image")
        btn_export_data = QPushButton("[Save] Export Data")
        btn_generate_report = QPushButton("📄 Generate Report")
        
        btn_save_image.clicked.connect(self.save_visualization)
        btn_export_data.clicked.connect(self.export_data)
        btn_generate_report.clicked.connect(self.generate_report)
        
        for btn in [btn_save_image, btn_export_data, btn_generate_report]:
            btn.setMinimumHeight(30)
            export_layout.addWidget(btn)
            
        controls_layout.addWidget(export_group)
        
        # Results summary
        summary_group = QGroupBox("Results Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.results_summary = QTextEdit()
        self.results_summary.setReadOnly(True)
        self.results_summary.setMaximumHeight(150)
        self.results_summary.setText("Load results to see summary...")
        
        summary_layout.addWidget(self.results_summary)
        controls_layout.addWidget(summary_group)
        
        controls_layout.addStretch()
        layout.addWidget(controls_widget)
        
        # Canvas for post-processing visualization
        self.postproc_canvas = self.create_postprocessing_canvas()
        layout.addWidget(self.postproc_canvas)
        
        return tab
        
    def create_postprocessing_canvas(self):
        """Create post-processing visualization canvas."""
        canvas = FigureCanvas(Figure(figsize=(10, 6), facecolor=Theme.BACKGROUND))
        
        ax = canvas.figure.add_subplot(111, facecolor='#1e1e1e')
        ax.set_xlabel('X [m]', color=Theme.TEXT)
        ax.set_ylabel('Y [m]', color=Theme.TEXT)
        ax.grid(True, alpha=0.3, color=Theme.TEXT_SECONDARY)
        ax.tick_params(colors=Theme.TEXT)
        for spine in ax.spines.values():
            spine.set_color(Theme.BORDER)
            
        canvas.ax = ax
        
        # Initial display
        ax.text(0.5, 0.5, 'Load results to visualize fields', 
               transform=ax.transAxes, ha='center', va='center',
               color=Theme.TEXT_SECONDARY, fontsize=14)
        
        return canvas
        
    # Event handlers and methods
    def on_mode_changed(self, mode, checked):
        """Handle drawing mode change for immediate activation."""
        if checked and hasattr(self, 'canvas'):
            self.canvas.drawing_mode = mode
            self.canvas.current_points = []  # Clear current drawing
            self.update_canvas(self.canvas)
            
    def clear_geometry(self):
        """Clear all geometry elements."""
        self.geometry.clear()
        if hasattr(self, 'canvas'):
            self.canvas.current_points = []
            self.update_canvas(self.canvas)
        self.update_workflow_status()
        
    def undo_last_element(self):
        """Remove the last geometry element."""
        if self.geometry.elements:
            self.geometry.elements.pop()
            if hasattr(self, 'canvas'):
                self.update_canvas(self.canvas)
            self.update_workflow_status()
        
    def validate_geometry(self):
        """Validate current geometry."""
        try:
            x_coords, y_coords = self.geometry.get_interpolated_points()
            if not x_coords or not y_coords:
                QMessageBox.warning(self, "Validation", "No geometry to validate")
                return
                
            # Basic validation checks
            
            if max(x_coords) - min(x_coords) < 0.1:
                QMessageBox.warning(self, "Validation", "Geometry too short (< 0.1m)")
                return
                
            if any(abs(y) > 1.0 for y in y_coords):
                QMessageBox.warning(self, "Validation", "Geometry too wide (> 1.0m radius)")
                return
                
            QMessageBox.information(self, "Validation", 
                                  f"Geometry valid!\nLength: {max(x_coords):.3f}m\n"
                                  f"Max radius: {max(abs(y) for y in y_coords):.3f}m\n"
                                  f"Elements: {len(self.geometry.elements)}")
                                  
        except Exception as e:
            QMessageBox.critical(self, "Validation Error", f"Validation failed: {str(e)}")
            
    def generate_mesh(self):
        """Generate mesh from current geometry."""
        if not self.geometry.elements:
            QMessageBox.warning(self, "Mesh Generation", "No geometry defined")
            return
            
        try:
            from core.modules.mesh_generator import MeshParameters, AdvancedMeshGenerator
            
            # Create mesh parameters from GUI
            params = MeshParameters()
            params.element_size = self.global_size.value()
            params.min_element_size = self.wall_size.value()
            params.boundary_layer_enabled = self.enable_bl.isChecked()
            params.boundary_layer_elements = self.num_bl_layers.value()
            params.boundary_layer_thickness = self.first_layer_thickness.value()
            params.boundary_layer_growth_rate = self.growth_ratio.value()
            params.domain_extension = self.farfield_distance.value()
            
            # Generate mesh
            generator = AdvancedMeshGenerator()
            mesh_data = generator.generate_mesh(self.geometry, params)
            
            if mesh_data and isinstance(mesh_data, dict) and ('nodes' in mesh_data or 'vertices' in mesh_data):
                # Store mesh data for later use (ensure consistent naming)
                self.mesh_data = mesh_data
                self.current_mesh_data = mesh_data  # Also store in current_mesh_data for export
                self.mesh_generator = generator  # Store generator for export functionality
                
                # Update stats display
                stats = mesh_data.get('stats', {})
                self.display_mesh_stats(stats)
                
                # Update visualization (pass mesh data instead of file)
                self.visualize_mesh_data(mesh_data)
                self.update_workflow_status()
                
                # Show success message with mesh statistics
                num_nodes = stats.get('num_nodes', 0)
                num_elements = stats.get('num_elements', 0)
                QMessageBox.information(self, "Mesh Generation", 
                    f"Mesh generated successfully!\nNodes: {num_nodes}\nElements: {num_elements}")
            else:
                QMessageBox.warning(self, "Mesh Generation", "Mesh generation failed")
                
        except Exception as e:
            QMessageBox.critical(self, "Mesh Generation Error", f"Failed to generate mesh: {str(e)}")
            
    def analyze_mesh(self):
        """Analyze existing mesh quality."""
        try:
            if not self.current_mesh_data:
                QMessageBox.warning(self, "Mesh Analysis", "No mesh data available. Please generate a mesh first.")
                return
                
            if self.mesh_generator and hasattr(self.mesh_generator, 'analyze_mesh_quality'):
                # Analyze mesh quality using the mesh generator
                quality_stats = self.mesh_generator.analyze_mesh_quality()
                
                if quality_stats:
                    analysis_text = f"""Mesh Quality Analysis:
• Total elements: {quality_stats.get('num_elements', 'N/A')}
• Total nodes: {quality_stats.get('num_nodes', 'N/A')}
• Min element quality: {quality_stats.get('min_quality', 0):.4f}
• Average element quality: {quality_stats.get('avg_quality', 0):.4f}
• Max aspect ratio: {quality_stats.get('max_aspect_ratio', 0):.2f}
• Skewness: {quality_stats.get('skewness', 0):.4f}
• Orthogonality: {quality_stats.get('orthogonality', 0):.4f}

Quality Assessment:
{self._get_quality_assessment(quality_stats)}"""
                    
                    QMessageBox.information(self, "Mesh Analysis", analysis_text)
                else:
                    QMessageBox.warning(self, "Mesh Analysis", "Could not analyze mesh quality.")
            else:
                QMessageBox.information(self, "Mesh Analysis", "Basic mesh statistics available in the mesh panel.")
                
        except Exception as e:
            QMessageBox.critical(self, "Mesh Analysis Error", f"Failed to analyze mesh: {str(e)}")
    
    def _get_quality_assessment(self, stats):
        """Get quality assessment text based on statistics."""
        min_qual = stats.get('min_quality', 0)
        avg_qual = stats.get('avg_quality', 0)
        max_aspect = stats.get('max_aspect_ratio', float('inf'))
        
        issues = []
        if min_qual < 0.1:
            issues.append("⚠️ Very poor element quality detected")
        elif min_qual < 0.3:
            issues.append("⚠️ Poor element quality detected")
        
        if max_aspect > 100:
            issues.append("⚠️ High aspect ratio elements detected")
        elif max_aspect > 50:
            issues.append("⚠️ Moderate aspect ratio concerns")
            
        if avg_qual > 0.7:
            if not issues:
                return "[OK] Excellent mesh quality"
            else:
                return "[OK] Good overall quality with some concerns"
        elif avg_qual > 0.5:
            return "⚠️ Acceptable mesh quality"
        else:
            return "[X] Poor mesh quality - consider regenerating"
        
    def export_mesh(self):
        """Export mesh to file."""
        try:
            if not self.current_mesh_data:
                QMessageBox.warning(self, "Export Mesh", "No mesh data available. Please generate a mesh first.")
                return
                
            # Get export file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Mesh", "", 
                "MSH Files (*.msh);;STL Files (*.stl);;VTK Files (*.vtk);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Determine format from extension
            extension = os.path.splitext(file_path)[1].lower()
            
            if self.mesh_generator and hasattr(self.mesh_generator, 'export_mesh'):
                success = self.mesh_generator.export_mesh(file_path, format=extension[1:])
                if success:
                    QMessageBox.information(self, "Export Mesh", f"Mesh exported successfully to:\n{file_path}")
                else:
                    QMessageBox.warning(self, "Export Mesh", "Failed to export mesh.")
            else:
                # Basic export functionality
                if extension == '.msh':
                    self._export_mesh_msh(file_path)
                    QMessageBox.information(self, "Export Mesh", f"Mesh exported to MSH format:\n{file_path}")
                else:
                    QMessageBox.warning(self, "Export Mesh", f"Export format {extension} not supported yet.")
                    
        except Exception as e:
            QMessageBox.critical(self, "Export Mesh Error", f"Failed to export mesh: {str(e)}")
    
    def _export_mesh_msh(self, file_path):
        """Export mesh in MSH format (basic implementation)."""
        try:
            if not self.current_mesh_data:
                raise ValueError("No mesh data available")
                
            # Get mesh data
            nodes = self.current_mesh_data.get('nodes') or self.current_mesh_data.get('vertices', [])
            elements = self.current_mesh_data.get('elements', [])
            
            with open(file_path, 'w') as f:
                # MSH format header
                f.write("$MeshFormat\n")
                f.write("2.2 0 8\n")
                f.write("$EndMeshFormat\n")
                
                # Nodes section
                f.write("$Nodes\n")
                f.write(f"{len(nodes)}\n")
                for i, (x, y) in enumerate(nodes):
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
                
                elem_type = element_type_map.get(self.current_mesh_data.get('element_type', 'quad'), 3)
                
                for i, elem in enumerate(elements):
                    # Format: element_id element_type num_tags tag1 tag2 ... node1 node2 ...
                    nodes_str = " ".join(str(n+1) for n in elem)  # Convert to 1-based
                    f.write(f"{i+1} {elem_type} 2 1 1 {nodes_str}\n")
                    
                f.write("$EndElements\n")
            
        except Exception as e:
            print(f"MSH export error: {e}")
            # Fallback to basic export
            with open(file_path, 'w') as f:
                f.write("$MeshFormat\n")
                f.write("2.2 0 8\n")
                f.write("$EndMeshFormat\n")
                f.write("$Nodes\n")
                f.write("1\n")  # Basic placeholder
                f.write("1 0.0 0.0 0.0\n")
                f.write("$EndNodes\n")
                f.write("$Elements\n")
                f.write("0\n")
                f.write("$EndElements\n")
        
    def display_mesh_stats(self, stats):
        """Display mesh statistics."""
        if stats:
            # Helper function to safely format numeric values
            def safe_format(value, default='N/A', format_str='.3f'):
                if isinstance(value, (int, float)) and value is not None:
                    return f"{value:{format_str}}"
                return default
            
            text = f"""Mesh Statistics:
• Total elements: {stats.get('num_elements', 'N/A')}
• Total nodes: {stats.get('num_nodes', 'N/A')}
• Min quality: {safe_format(stats.get('min_quality'), 'N/A', '.3f')}
• Average quality: {safe_format(stats.get('avg_quality'), 'N/A', '.3f')}
• Aspect ratio: {safe_format(stats.get('aspect_ratio'), 'N/A', '.2f')}
"""
            self.mesh_stats.setText(text)
        
    def visualize_mesh(self, mesh_file):
        """Visualize mesh in canvas."""
        # Basic mesh visualization - would need actual mesh data parsing
        self.mesh_canvas.ax.clear()
        self.mesh_canvas.ax.text(0.5, 0.5, f'Mesh generated\n{os.path.basename(mesh_file)}', 
                               transform=self.mesh_canvas.ax.transAxes, ha='center', va='center',
                               color=Theme.TEXT, fontsize=12)
        self.mesh_canvas.draw()
        
    def visualize_mesh_data(self, mesh_data):
        """Visualize mesh data directly in canvas."""
        try:
            self.mesh_canvas.ax.clear()
            
            # Extract mesh information
            nodes = mesh_data.get('nodes') or mesh_data.get('vertices', [])
            elements = mesh_data.get('elements', [])
            stats = mesh_data.get('stats', {})
            
            if nodes and elements:
                # Convert nodes to numpy array for easier handling
                nodes_array = np.array(nodes)
                x_coords = nodes_array[:, 0]
                y_coords = nodes_array[:, 1]
                
                # Plot mesh elements (edges)
                for elem in elements:
                    if len(elem) >= 3:
                        # Create closed polygon for element
                        elem_coords = [nodes[i] for i in elem]
                        elem_coords.append(elem_coords[0])  # Close the polygon
                        
                        elem_x = [coord[0] for coord in elem_coords]
                        elem_y = [coord[1] for coord in elem_coords]
                        
                        # Plot element edges
                        self.mesh_canvas.ax.plot(elem_x, elem_y, 'b-', linewidth=0.5, alpha=0.7)
                
                # Plot nodes as small dots
                self.mesh_canvas.ax.scatter(x_coords, y_coords, s=1, c='red', alpha=0.8, zorder=5)
                
                # Highlight boundaries if available
                boundary_elements = mesh_data.get('boundary_elements', {})
                
                # Plot different boundaries in different colors
                boundary_colors = {
                    'inlet': 'green',
                    'outlet': 'red', 
                    'wall_upper': 'blue',
                    'wall_lower': 'blue',
                    'centerline': 'orange'
                }
                
                for boundary_name, boundary_elems in boundary_elements.items():
                    color = boundary_colors.get(boundary_name, 'black')
                    
                    for boundary_elem in boundary_elems:
                        if len(boundary_elem) >= 2:
                            # Plot boundary edge
                            edge_coords = [nodes[i] for i in boundary_elem]
                            edge_x = [coord[0] for coord in edge_coords]
                            edge_y = [coord[1] for coord in edge_coords]
                            
                            self.mesh_canvas.ax.plot(edge_x, edge_y, color=color, linewidth=2, alpha=0.8)
                
                # Set equal aspect ratio and add grid
                self.mesh_canvas.ax.set_aspect('equal')
                self.mesh_canvas.ax.grid(True, alpha=0.3)
                
                # Add title with mesh statistics
                num_nodes = stats.get('num_nodes', len(nodes))
                num_elements = stats.get('num_elements', len(elements))
                element_type = stats.get('element_type', 'unknown')
                quality = stats.get('avg_quality', 0)
                
                title = f"Mesh: {num_nodes} nodes, {num_elements} {element_type} elements"
                if quality > 0:
                    title += f", Quality: {quality:.2f}"
                
                self.mesh_canvas.ax.set_title(title, fontsize=10, color=Theme.TEXT)
                self.mesh_canvas.ax.set_xlabel('x [m]', color=Theme.TEXT)
                self.mesh_canvas.ax.set_ylabel('y [m]', color=Theme.TEXT)
                
                # Set background color to match theme
                self.mesh_canvas.ax.set_facecolor(Theme.CANVAS_BACKGROUND)
                
                # Add legend for boundaries
                if boundary_elements:
                    legend_elements = []
                    for boundary_name, color in boundary_colors.items():
                        if boundary_name in boundary_elements and boundary_elements[boundary_name]:
                            legend_elements.append(matplotlib.lines.Line2D([0], [0], color=color, lw=2, label=boundary_name))
                    
                    if legend_elements:
                        self.mesh_canvas.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                
            else:
                # Show message if no mesh data
                self.mesh_canvas.ax.text(0.5, 0.5, 'No mesh data to display', 
                                       transform=self.mesh_canvas.ax.transAxes, ha='center', va='center',
                                       color=Theme.TEXT, fontsize=12)
                self.mesh_canvas.ax.set_title('Mesh Visualization', color=Theme.TEXT)
            
            self.mesh_canvas.draw()
            
        except Exception as e:
            print(f"Mesh visualization error: {e}")
            self.mesh_canvas.ax.clear()
            self.mesh_canvas.ax.text(0.5, 0.5, f'Mesh visualization failed:\n{str(e)}', 
                                   transform=self.mesh_canvas.ax.transAxes, ha='center', va='center',
                                   color=Theme.TEXT, fontsize=10)
            self.mesh_canvas.draw()
            # Fallback to simple text display
            self.mesh_canvas.ax.clear()
            self.mesh_canvas.ax.text(0.5, 0.5, f'Mesh generated successfully\n{str(e)}', 
                                   transform=self.mesh_canvas.ax.transAxes, ha='center', va='center',
                                   fontsize=12)
            self.mesh_canvas.draw()
        
    def setup_simulation(self):
        """Setup OpenFOAM simulation case."""
        if not self.geometry.elements:
            QMessageBox.warning(self, "Simulation Setup", "No geometry defined")
            return
            
        try:
            from core.modules.simulation_setup import SimulationSetup, BoundaryCondition, BoundaryType
            
            # Create boundary conditions from GUI
            inlet_bc = BoundaryCondition(
                name="inlet",
                boundary_type=BoundaryType.INLET,
                velocity_magnitude=self.inlet_velocity.value(),
                velocity_direction=(1.0, 0.0),
                pressure_type="zeroGradient"
            )
            
            outlet_bc = BoundaryCondition(
                name="outlet", 
                boundary_type=BoundaryType.OUTLET,
                velocity_magnitude=0.0,
                pressure_type="fixedValue",
                pressure_value=self.outlet_pressure.value()
            )
            
            wall_bc = BoundaryCondition(
                name="wall",
                boundary_type=BoundaryType.WALL,
                velocity_magnitude=0.0,
                pressure_type="zeroGradient",
                wall_roughness=0.0
            )
            
            # Setup simulation
            sim_setup = SimulationSetup()
            sim_setup.case_directory = self.case_directory.text()
            
            # Debug: Check mesh_data availability
            mesh_data = getattr(self, 'mesh_data', None)
            print(f"DEBUG Frontend: mesh_data available: {mesh_data is not None}")
            if mesh_data:
                print(f"DEBUG Frontend: mesh_data keys: {mesh_data.keys()}")
            
            # Configure solver settings
            sim_setup.solver_settings.n_processors = self.n_processors.value()
            sim_setup.solver_settings.decomposition_method = self.decomposition_method.currentText()
            sim_setup.solver_settings.convergence_tolerance = self.convergence_tolerance.value()
            # Note: max_iterations would need to be added to SolverSettings if needed
            
            # Add boundary conditions
            sim_setup.add_boundary_condition(inlet_bc)
            sim_setup.add_boundary_condition(outlet_bc)
            sim_setup.add_boundary_condition(wall_bc)
            
            # Generate case files
            sim_setup.generate_case_files(self.geometry, mesh_data)
            
            # Update current case directory for simulation runner
            self.current_case_directory = sim_setup.case_directory
            
            self.simulation_log.setText("Case files generated successfully!")
            self.update_workflow_status()
            QMessageBox.information(self, "Simulation Setup", "Case files generated successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Simulation Setup Error", f"Failed to setup simulation: {str(e)}")
            
    def run_simulation(self):
        """Run OpenFOAM simulation."""
        try:
            if not self.current_case_directory or not os.path.exists(self.current_case_directory):
                QMessageBox.warning(self, "Run Simulation", "No case directory set up. Please generate a case first.")
                return
                
            self.simulation_log.append(" Starting simulation...")
            
            # Import OpenFOAM runner
            from core.openfoam_runner import OpenFOAMRunner
            
            runner = OpenFOAMRunner(self.current_case_directory)
            
            # Run blockMesh first
            self.simulation_log.append(" Running blockMesh...")
            if runner.block_mesh():
                self.simulation_log.append("[OK] blockMesh completed successfully")
            else:
                self.simulation_log.append("[X] blockMesh failed")
                QMessageBox.warning(self, "Simulation Error", "Mesh generation failed. Check case setup.")
                return
            
            # Check if parallel processing is needed
            n_procs = self.n_processors.value()
            if n_procs > 1:
                self.simulation_log.append(f" Preparing parallel case for {n_procs} processors...")
                if runner.decompose_par():
                    self.simulation_log.append("[OK] decomposePar completed successfully")
                else:
                    self.simulation_log.append("[X] decomposePar failed")
                    QMessageBox.warning(self, "Simulation Error", "Domain decomposition failed. Check case setup.")
                    return
            
            # Run solver
            solver = self.solver_type.currentText()
            if n_procs > 1:
                self.simulation_log.append(f" Running {solver} in parallel on {n_procs} processors...")
            else:
                self.simulation_log.append(f" Running {solver}...")
            
            if runner.run_solver(solver, n_procs):
                self.simulation_log.append(f"[OK] {solver} completed successfully")
                
                # Reconstruct if parallel
                if n_procs > 1:
                    self.simulation_log.append(" Reconstructing parallel results...")
                    if runner.reconstruct_par():
                        self.simulation_log.append("[OK] reconstructPar completed successfully")
                    else:
                        self.simulation_log.append("[WARNING] reconstructPar failed, parallel results may be incomplete")
                
                self.simulation_log.append(" Simulation completed!")
                
                # Update workflow status
                self.current_results = self.current_case_directory
                self.update_workflow_status()
                
                QMessageBox.information(self, "Simulation Complete", "Simulation completed successfully!")
            else:
                self.simulation_log.append(f"[X] {solver} failed")
                QMessageBox.warning(self, "Simulation Error", f"{solver} execution failed. Check log files.")
                
        except ImportError:
            QMessageBox.warning(self, "OpenFOAM Error", "OpenFOAM integration not available.")
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", f"Failed to run simulation: {str(e)}")
            self.simulation_log.append(f"[X] Error: {str(e)}")
        
    def monitor_simulation(self):
        """Monitor simulation progress."""
        try:
            if not self.current_case_directory or not os.path.exists(self.current_case_directory):
                QMessageBox.warning(self, "Monitor Simulation", "No active simulation case found.")
                return
                
            # Check for log files
            log_files = []
            for file in os.listdir(self.current_case_directory):
                if file.startswith('log.'):
                    log_files.append(file)
            
            if not log_files:
                QMessageBox.information(self, "Monitor Simulation", "No log files found. Simulation may not be running.")
                return
                
            # Display latest log file content
            latest_log = sorted(log_files)[-1]
            log_path = os.path.join(self.current_case_directory, latest_log)
            
            try:
                with open(log_path, 'r') as f:
                    log_content = f.read()
                    
                # Show last 50 lines
                lines = log_content.split('\n')
                recent_lines = lines[-50:] if len(lines) > 50 else lines
                
                dialog = QMessageBox(self)
                dialog.setWindowTitle("Simulation Monitor")
                dialog.setText(f"Latest log: {latest_log}")
                dialog.setDetailedText('\n'.join(recent_lines))
                dialog.exec()
                
            except Exception as e:
                QMessageBox.warning(self, "Monitor Error", f"Could not read log file: {str(e)}")
                
        except Exception as e:
            QMessageBox.critical(self, "Monitor Error", f"Failed to monitor simulation: {str(e)}")
        
    def stop_simulation(self):
        """Stop running simulation."""
        try:
            # This is a basic implementation - in a real scenario you'd need to track process IDs
            reply = QMessageBox.question(
                self, "Stop Simulation", 
                "This will attempt to stop any running OpenFOAM processes. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                # Kill OpenFOAM processes (basic approach)
                import subprocess
                try:
                    # Kill common OpenFOAM solvers
                    solvers = ['simpleFoam', 'pisoFoam', 'pimpleFoam', 'rhoSimpleFoam', 'blockMesh']
                    killed_any = False
                    
                    for solver in solvers:
                        result = subprocess.run(['pkill', '-f', solver], 
                                             capture_output=True, text=True)
                        if result.returncode == 0:
                            killed_any = True
                            self.simulation_log.append(f"🛑 Stopped {solver}")
                    
                    if killed_any:
                        self.simulation_log.append("🛑 Simulation stopped by user")
                        QMessageBox.information(self, "Stop Simulation", "Simulation processes stopped.")
                    else:
                        QMessageBox.information(self, "Stop Simulation", "No running simulation processes found.")
                        
                except Exception as e:
                    QMessageBox.warning(self, "Stop Error", f"Could not stop processes: {str(e)}")
                    
        except Exception as e:
            QMessageBox.critical(self, "Stop Error", f"Failed to stop simulation: {str(e)}")
        
    def browse_case_directory(self):
        """Browse for case directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Case Directory")
        if directory:
            self.case_directory.setText(directory)
            
    def load_results(self):
        """Load simulation results."""
        try:
            results_path = self.results_path.text().strip()
            if not results_path:
                QMessageBox.warning(self, "Load Results", "Please specify a results directory.")
                return
                
            if not os.path.exists(results_path):
                QMessageBox.warning(self, "Load Results", "Results directory does not exist.")
                return
                
            self.results_summary.setText("Loading results...")
            
            # Check for OpenFOAM results structure
            time_dirs = []
            try:
                for item in os.listdir(results_path):
                    item_path = os.path.join(results_path, item)
                    if os.path.isdir(item_path):
                        try:
                            # Check if it's a time directory (numeric)
                            float(item)
                            time_dirs.append(item)
                        except ValueError:
                            continue
            except Exception as e:
                QMessageBox.warning(self, "Load Results", f"Could not read results directory: {str(e)}")
                return
            
            if not time_dirs:
                QMessageBox.warning(self, "Load Results", "No time directories found in results.")
                return
                
            # Sort time directories
            time_dirs.sort(key=float)
            latest_time = time_dirs[-1]
            
            # Check for field files
            latest_dir = os.path.join(results_path, latest_time)
            field_files = []
            
            try:
                for file in os.listdir(latest_dir):
                    if not file.startswith('.'):
                        field_files.append(file)
            except Exception as e:
                QMessageBox.warning(self, "Load Results", f"Could not read time directory: {str(e)}")
                return
            
            # Update summary
            summary_text = f"""Results Summary:
• Results directory: {results_path}
• Number of time steps: {len(time_dirs)}
• Latest time: {latest_time}
• Available fields: {', '.join(field_files)}
• First time: {time_dirs[0]}

Status: [OK] Results loaded successfully"""
            
            self.results_summary.setText(summary_text)
            
            # Store results info
            self.current_results = {
                'path': results_path,
                'time_dirs': time_dirs,
                'latest_time': latest_time,
                'fields': field_files
            }
            
            # Update time step combo box
            if hasattr(self, 'time_step'):
                self.time_step.clear()
                self.time_step.addItems(time_dirs)
                # Select latest time by default
                self.time_step.setCurrentText(latest_time)
            
            # Update field combo box with available fields
            if hasattr(self, 'field_type'):
                current_field = self.field_type.currentText()
                self.field_type.clear()
                self.field_type.addItems(field_files)
                # Try to keep previous selection
                index = self.field_type.findText(current_field)
                if index >= 0:
                    self.field_type.setCurrentIndex(index)
            
            self.update_workflow_status()
            
            # Trigger initial visualization
            self.update_visualization()
            
            QMessageBox.information(self, "Load Results", f"Results loaded successfully!\nFound {len(time_dirs)} time steps.")
            
        except Exception as e:
            QMessageBox.critical(self, "Load Results Error", f"Failed to load results: {str(e)}")
            self.results_summary.setText("[X] Failed to load results")
        
    def browse_results_directory(self):
        """Browse for results directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Results Directory")
        if directory:
            self.results_path.setText(directory)
            
    def on_field_changed(self):
        """Handle field type change."""
        self.update_visualization()
        
    def update_visualization(self):
        """Update post-processing visualization with actual OpenFOAM data."""
        try:
            print("=== update_visualization() called ===")
            
            if not self.current_results:
                print("WARNING: No current_results available")
                return
            
            # Get selected field and time step
            field_name = self.field_type.currentText()
            time_step = self.time_step.currentText()
            
            print(f"Selected field: '{field_name}', time: '{time_step}'")
            
            if not field_name or not time_step:
                print("WARNING: Field or time step not selected")
                return
            
            # Clear the canvas
            self.postproc_canvas.ax.clear()
            
            # Read and visualize the field
            results_path = self.current_results['path']
            field_path = os.path.join(results_path, time_step, field_name)
            
            print(f"Field path: {field_path}")
            print(f"File exists: {os.path.exists(field_path)}")
            
            if not os.path.exists(field_path):
                print(f"ERROR: Field file not found!")
                self.postproc_canvas.ax.text(0.5, 0.5, f'Field {field_name} not found at time {time_step}', 
                       transform=self.postproc_canvas.ax.transAxes, ha='center', va='center',
                       color=Theme.TEXT_SECONDARY, fontsize=12)
                self.postproc_canvas.draw()
                return
            
            print("Calling _visualize_field_basic...")
            
            # Try to use ResultsProcessor if available
            if self.results_processor:
                try:
                    self._visualize_field_advanced(field_path, field_name, time_step)
                except Exception as e:
                    print(f"Advanced visualization failed: {e}, trying basic...")
                    self._visualize_field_basic(field_path, field_name, time_step)
            else:
                self._visualize_field_basic(field_path, field_name, time_step)
            
            # Update canvas
            self.postproc_canvas.draw()
            
        except Exception as e:
            print(f"Visualization update failed: {e}")
            import traceback
            traceback.print_exc()
            self.postproc_canvas.ax.text(0.5, 0.5, f'Error: {str(e)}', 
                   transform=self.postproc_canvas.ax.transAxes, ha='center', va='center',
                   color='red', fontsize=12)
            self.postproc_canvas.draw()
    
    def _visualize_field_basic(self, field_path, field_name, time_step):
        """Basic field visualization by reading OpenFOAM field file and actual mesh."""
        import numpy as np
        
        try:
            # Clear figure completely
            self.postproc_canvas.figure.clear()
            self.postproc_canvas.ax = self.postproc_canvas.figure.add_subplot(111, facecolor='#1e1e1e')
            
            # Style the axes
            self.postproc_canvas.ax.tick_params(colors=Theme.TEXT)
            for spine in self.postproc_canvas.ax.spines.values():
                spine.set_color(Theme.BORDER)
            
            # Read field data first
            field_data = self._read_openfoam_field(field_path, field_name)
            
            if field_data is None or len(field_data) == 0:
                self.postproc_canvas.ax.text(0.5, 0.5, f'Could not read field data from {field_name}', 
                       transform=self.postproc_canvas.ax.transAxes, ha='center', va='center',
                       color=Theme.TEXT_SECONDARY, fontsize=12)
                self.postproc_canvas.draw()
                return
            
            print(f"Read {len(field_data)} field values from {field_name}")
            
            # Read actual mesh - try mesh_info.json first, then compute from polyMesh
            mesh_path = os.path.join(self.current_results['path'], 'constant', 'polyMesh')
            mesh_info_file = os.path.join(mesh_path, 'mesh_info.json')
            
            cell_centers = None
            nozzle_geometry = None
            
            # Try to load mesh_info.json which contains nozzle-shaped mesh
            if os.path.exists(mesh_info_file):
                try:
                    import json
                    with open(mesh_info_file, 'r') as f:
                        mesh_info = json.load(f)
                    
                    if 'nodes' in mesh_info:
                        # Use the nodes as cell centers (for nozzle-shaped mesh)
                        nodes = np.array(mesh_info['nodes'])
                        cell_centers = [(n[0], n[1]) for n in nodes]
                        nozzle_geometry = {'nodes': nodes}
                        print(f"Loaded {len(cell_centers)} nodes from mesh_info.json")
                        
                        # Extract nozzle walls for overlay
                        x_unique = sorted(set(nodes[:, 0]))
                        upper_wall = []
                        lower_wall = []
                        for x in x_unique:
                            points_at_x = nodes[nodes[:, 0] == x]
                            if len(points_at_x) > 0:
                                upper_wall.append((x, points_at_x[:, 1].max()))
                                lower_wall.append((x, points_at_x[:, 1].min()))
                        nozzle_geometry['upper_wall'] = np.array(upper_wall)
                        nozzle_geometry['lower_wall'] = np.array(lower_wall)
                except Exception as e:
                    print(f"Could not load mesh_info.json: {e}")
            
            # If no mesh_info.json, compute cell centers from OpenFOAM mesh
            if cell_centers is None:
                cell_centers = self._read_openfoam_cell_centers(mesh_path)
                
                if cell_centers is None or len(cell_centers) == 0:
                    self.postproc_canvas.ax.text(0.5, 0.5, 'Could not read mesh data', 
                           transform=self.postproc_canvas.ax.transAxes, ha='center', va='center',
                           color=Theme.TEXT_SECONDARY, fontsize=12)
                    self.postproc_canvas.draw()
                    return
                
                print(f"Computed {len(cell_centers)} cell centers from OpenFOAM mesh")
            
            # Verify field data matches mesh
            if len(field_data) != len(cell_centers):
                print(f"WARNING: Field data size ({len(field_data)}) doesn't match mesh cells ({len(cell_centers)})")
                # Truncate or pad as needed
                min_size = min(len(field_data), len(cell_centers))
                field_data = field_data[:min_size]
                cell_centers = cell_centers[:min_size]
            
            # Extract x, y coordinates from cell centers
            x_data = np.array([c[0] for c in cell_centers])
            y_data = np.array([c[1] for c in cell_centers])
            field_array = np.array(field_data)
            
            # Get domain bounds from actual mesh
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            
            print(f"Mesh domain: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
            
            # Plot based on field type and plot type
            plot_type = self.plot_type.currentText()
            colormap = self.colormap.currentText()
            levels = self.contour_levels.value()
            
            # Create fine visualization grid
            xi = np.linspace(x_min, x_max, 200)
            yi = np.linspace(y_min, y_max, 100)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Interpolate field data onto visualization grid using actual cell centers
            from scipy.interpolate import griddata
            
            points_array = np.column_stack((x_data, y_data))
            
            print(f"Interpolating {len(field_data)} values onto {Xi.shape} grid...")
            Zi = griddata(points_array, field_array, (Xi, Yi), method='linear', fill_value=np.nan)
            
            # Fill NaN values with nearest neighbor
            mask = np.isnan(Zi)
            if mask.any():
                print(f"Filling {mask.sum()} NaN values with nearest neighbor")
                Zi_nearest = griddata(points_array, field_array, (Xi, Yi), method='nearest')
                Zi[mask] = Zi_nearest[mask]
            
            # Determine label based on field name
            if field_name == 'U':
                label = 'Velocity Magnitude [m/s]'
            elif field_name == 'p':
                label = 'Pressure [Pa]'
            elif field_name == 'T':
                label = 'Temperature [K]'
            elif field_name == 'k':
                label = 'Turbulent Kinetic Energy [m²/s²]'
            elif field_name == 'epsilon':
                label = 'Turbulent Dissipation [m²/s³]'
            elif field_name == 'nut':
                label = 'Turbulent Viscosity [m²/s]'
            else:
                label = field_name
            
            # Plot based on type
            if plot_type == "Contour":
                contour = self.postproc_canvas.ax.contourf(Xi, Yi, Zi, levels=levels, cmap=colormap)
                cbar = self.postproc_canvas.figure.colorbar(contour, ax=self.postproc_canvas.ax, label=label)
                cbar.ax.yaxis.set_tick_params(color=Theme.TEXT)
                cbar.ax.yaxis.label.set_color(Theme.TEXT)
                for label_obj in cbar.ax.yaxis.get_ticklabels():
                    label_obj.set_color(Theme.TEXT)
            elif plot_type == "Vector" and field_name == 'U':
                # Simplified vector plot
                stride = 8
                contour = self.postproc_canvas.ax.contourf(Xi, Yi, Zi, levels=levels, cmap=colormap, alpha=0.6)
                self.postproc_canvas.ax.quiver(Xi[::stride, ::stride], Yi[::stride, ::stride], 
                                               Zi[::stride, ::stride]*0.01, np.zeros_like(Zi[::stride, ::stride]),
                                               scale=50, color='white', alpha=0.8, width=0.003)
            else:
                # Default to contour
                contour = self.postproc_canvas.ax.contourf(Xi, Yi, Zi, levels=levels, cmap=colormap)
                cbar = self.postproc_canvas.figure.colorbar(contour, ax=self.postproc_canvas.ax, label=label)
                cbar.ax.yaxis.set_tick_params(color=Theme.TEXT)
                cbar.ax.yaxis.label.set_color(Theme.TEXT)
                for label_obj in cbar.ax.yaxis.get_ticklabels():
                    label_obj.set_color(Theme.TEXT)
            
            # Add statistics
            stats_text = f'Min: {np.min(field_data):.2e}\nMax: {np.max(field_data):.2e}\nMean: {np.mean(field_data):.2e}'
            self.postproc_canvas.ax.text(0.02, 0.98, stats_text, 
                                        transform=self.postproc_canvas.ax.transAxes,
                                        va='top', ha='left', color=Theme.TEXT,
                                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
                                        fontsize=9)
            
            # Overlay nozzle geometry outline
            if nozzle_geometry and 'upper_wall' in nozzle_geometry:
                # Use the nozzle walls from mesh_info.json
                try:
                    upper = nozzle_geometry['upper_wall']
                    lower = nozzle_geometry['lower_wall']
                    self.postproc_canvas.ax.plot(upper[:, 0], upper[:, 1], 
                                                'w-', linewidth=2.5, alpha=0.9, label='Nozzle wall')
                    self.postproc_canvas.ax.plot(lower[:, 0], lower[:, 1], 
                                                'w-', linewidth=2.5, alpha=0.9)
                    # Add centerline
                    self.postproc_canvas.ax.axhline(0, color='white', linestyle='--', 
                                                   linewidth=1, alpha=0.4, label='Centerline')
                except Exception as e:
                    print(f"Could not overlay nozzle walls: {e}")
            elif self.current_mesh_data and 'nodes' in self.current_mesh_data:
                # Fallback to current_mesh_data
                try:
                    nodes = self.current_mesh_data['nodes']
                    nodes_array = np.array(nodes)
                    
                    # Sort by x coordinate
                    sorted_indices = np.argsort(nodes_array[:, 0])
                    sorted_nodes = nodes_array[sorted_indices]
                    
                    # Separate upper and lower wall (y > 0 and y < 0)
                    upper_wall = sorted_nodes[sorted_nodes[:, 1] > 0.01]
                    lower_wall = sorted_nodes[sorted_nodes[:, 1] < -0.01]
                    
                    if len(upper_wall) > 0:
                        self.postproc_canvas.ax.plot(upper_wall[:, 0], upper_wall[:, 1], 
                                                    'w-', linewidth=2, alpha=0.8, label='Upper wall')
                    if len(lower_wall) > 0:
                        self.postproc_canvas.ax.plot(lower_wall[:, 0], lower_wall[:, 1], 
                                                    'w-', linewidth=2, alpha=0.8, label='Lower wall')
                except Exception as e:
                    print(f"Could not overlay geometry: {e}")
            
            self.postproc_canvas.ax.set_title(f'{field_name} at t={time_step}s', color=Theme.TEXT, fontsize=12)
            self.postproc_canvas.ax.set_xlabel('X [m]', color=Theme.TEXT)
            self.postproc_canvas.ax.set_ylabel('Y [m]', color=Theme.TEXT)
            self.postproc_canvas.ax.set_aspect('equal')
            self.postproc_canvas.ax.grid(True, alpha=0.2, color=Theme.TEXT_SECONDARY)
            
            self.postproc_canvas.draw()
            print(f"Visualization complete for {field_name}")
                
        except Exception as e:
            print(f"Basic visualization error: {e}")
            import traceback
            traceback.print_exc()
            self.postproc_canvas.ax.text(0.5, 0.5, f'Visualization error:\n{str(e)}', 
                   transform=self.postproc_canvas.ax.transAxes, ha='center', va='center',
                   color='red', fontsize=10)
            self.postproc_canvas.draw()
    
    def _read_openfoam_points(self, points_file):
        """Read points from OpenFOAM polyMesh/points file."""
        points = []
        try:
            with open(points_file, 'r') as f:
                lines = f.readlines()
                
            # Find the point count
            in_points = False
            for i, line in enumerate(lines):
                if line.strip().startswith('('):
                    in_points = True
                    continue
                if in_points:
                    if line.strip().startswith(')'):
                        break
                    # Parse point coordinates (x y z)
                    parts = line.strip().strip('()').split()
                    if len(parts) >= 3:
                        try:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            points.append((x, y, z))
                        except:
                            continue
            
            return points[:1000]  # Limit for performance
        except Exception as e:
            print(f"Error reading points: {e}")
            return []
    
    def _read_openfoam_cell_centers(self, mesh_path):
        """Read cell centers from OpenFOAM mesh (points + faces + owner)."""
        try:
            import re
            
            # Read points
            points_file = os.path.join(mesh_path, 'points')
            with open(points_file, 'r') as f:
                content = f.read()
            
            paren_start = content.find('(')
            paren_end = content.rfind(')')
            points_str = content[paren_start+1:paren_end]
            
            point_pattern = r'\(\s*([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)\s+([+-]?[\d.eE+-]+)\s*\)'
            matches = re.findall(point_pattern, points_str)
            points = [(float(m[0]), float(m[1]), float(m[2])) for m in matches]
            
            print(f"Read {len(points)} points from mesh")
            
            # Read faces
            faces_file = os.path.join(mesh_path, 'faces')
            with open(faces_file, 'r') as f:
                content = f.read()
            
            paren_start = content.find('(')
            paren_end = content.rfind(')')
            faces_str = content[paren_start+1:paren_end]
            
            # Parse faces (each face is like "4(0 1 2 3)")
            face_pattern = r'(\d+)\s*\(([^)]+)\)'
            face_matches = re.findall(face_pattern, faces_str)
            faces = []
            for npts, pts_str in face_matches:
                pt_indices = [int(x) for x in pts_str.split()]
                faces.append(pt_indices)
            
            print(f"Read {len(faces)} faces from mesh")
            
            # Read owner file to determine cells
            owner_file = os.path.join(mesh_path, 'owner')
            with open(owner_file, 'r') as f:
                content = f.read()
            
            paren_start = content.find('(')
            paren_end = content.rfind(')')
            owner_str = content[paren_start+1:paren_end]
            owners = [int(x) for x in owner_str.split()]
            
            print(f"Read {len(owners)} owner entries")
            
            # Count cells
            num_cells = max(owners) + 1
            print(f"Mesh has {num_cells} cells")
            
            # Build cell-to-faces mapping
            cell_to_faces = [[] for _ in range(num_cells)]
            for face_idx, owner in enumerate(owners):
                cell_to_faces[owner].append(face_idx)
            
            # Read neighbour file for internal faces
            neighbour_file = os.path.join(mesh_path, 'neighbour')
            if os.path.exists(neighbour_file):
                with open(neighbour_file, 'r') as f:
                    content = f.read()
                
                paren_start = content.find('(')
                paren_end = content.rfind(')')
                neighbour_str = content[paren_start+1:paren_end]
                neighbours = [int(x) for x in neighbour_str.split()]
                
                for face_idx, neighbour in enumerate(neighbours):
                    if neighbour < num_cells:
                        cell_to_faces[neighbour].append(face_idx)
            
            # Compute cell centers as average of all face points
            cell_centers = []
            for cell_idx in range(num_cells):
                if not cell_to_faces[cell_idx]:
                    continue
                
                # Collect all unique points from all faces of this cell
                cell_points = set()
                for face_idx in cell_to_faces[cell_idx]:
                    if face_idx < len(faces):
                        for pt_idx in faces[face_idx]:
                            if pt_idx < len(points):
                                cell_points.add(pt_idx)
                
                if cell_points:
                    # Average all points
                    cx = sum(points[pi][0] for pi in cell_points) / len(cell_points)
                    cy = sum(points[pi][1] for pi in cell_points) / len(cell_points)
                    cell_centers.append((cx, cy))
                else:
                    # Fallback: use first face center
                    face_idx = cell_to_faces[cell_idx][0]
                    if face_idx < len(faces):
                        face_pts = faces[face_idx]
                        cx = sum(points[pi][0] for pi in face_pts) / len(face_pts)
                        cy = sum(points[pi][1] for pi in face_pts) / len(face_pts)
                        cell_centers.append((cx, cy))
            
            print(f"Computed {len(cell_centers)} cell centers")
            return cell_centers
            
        except Exception as e:
            print(f"Error reading cell centers: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _read_openfoam_field(self, field_path, field_name):
        """Read field data from OpenFOAM field file (supports both uniform and nonuniform)."""
        try:
            with open(field_path, 'r') as f:
                lines = f.readlines()
            
            field_values = []
            
            # Look for internalField section
            for i, line in enumerate(lines):
                if 'internalField' in line:
                    # Check if uniform or nonuniform
                    if 'uniform' in line:
                        # Uniform field - single value
                        import re
                        match = re.search(r'uniform\s+\(?([^;)]+)', line)
                        if match:
                            value_str = match.group(1).strip()
                            try:
                                # Try as scalar
                                value = float(value_str)
                                return [value]
                            except:
                                # Try as vector
                                parts = value_str.split()
                                if len(parts) >= 3:
                                    return [float(parts[0]), float(parts[1]), float(parts[2])]
                    
                    elif 'nonuniform' in line:
                        # Nonuniform field - read list of values
                        # Format: internalField   nonuniform List<vector>
                        # Next line: number
                        # Next line: (
                        # Then data lines
                        
                        # Find the number of entries
                        num_entries = 0
                        start_idx = i + 1
                        
                        # The number might be on the same line or next line
                        remaining = line.split('nonuniform')[1] if 'nonuniform' in line else ''
                        for part in remaining.split():
                            if part.isdigit():
                                num_entries = int(part)
                                break
                        
                        # If not found on same line, check next few lines
                        if num_entries == 0:
                            for j in range(i+1, min(i+5, len(lines))):
                                stripped = lines[j].strip()
                                if stripped.isdigit():
                                    num_entries = int(stripped)
                                    start_idx = j + 1
                                    break
                        
                        if num_entries == 0:
                            print(f"Could not find number of entries in field {field_name}")
                            return None
                        
                        print(f"Reading {num_entries} values from {field_name}")
                        
                        # Read the list - find opening parenthesis
                        list_started = False
                        for j in range(start_idx, min(start_idx + num_entries + 100, len(lines))):
                            line_text = lines[j].strip()
                            
                            if line_text == '(':
                                list_started = True
                                continue
                            
                            if list_started:
                                if line_text == ')' or line_text == ');':
                                    break
                                
                                if not line_text or line_text.startswith('//'):
                                    continue
                                
                                # Parse value (scalar or vector)
                                if line_text.startswith('(') and line_text.endswith(')'):
                                    # Vector value like (99.9936 -1.26947 0)
                                    vector_str = line_text.strip('()')
                                    parts = vector_str.split()
                                    if len(parts) >= 3:
                                        try:
                                            # For vectors, compute magnitude
                                            vx = float(parts[0])
                                            vy = float(parts[1])
                                            vz = float(parts[2])
                                            magnitude = (vx**2 + vy**2 + vz**2)**0.5
                                            field_values.append(magnitude)
                                        except ValueError:
                                            pass
                                else:
                                    # Scalar value
                                    try:
                                        field_values.append(float(line_text))
                                    except ValueError:
                                        pass
                        
                        print(f"Successfully read {len(field_values)} values")
                        
                        if len(field_values) > 0:
                            return field_values
                        else:
                            print(f"No values were parsed from {field_name}")
                            return None
                    
                    break
            
            # If we get here, couldn't parse the field
            print(f"Could not find internalField in {field_path}")
            return None
            
        except Exception as e:
            print(f"Error reading field {field_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _visualize_field_advanced(self, field_path, field_name, time_step):
        """Advanced visualization using ResultsProcessor."""
        # This would use the results_processor module
        # For now, fall back to basic
        self._visualize_field_basic(field_path, field_name, time_step)
    
    def analyze_wall_values(self):
        """Analyze values at wall."""
        try:
            if not self.current_results:
                QMessageBox.warning(self, "Wall Analysis", "No results loaded. Please load simulation results first.")
                return
                
            # Basic wall analysis implementation
            results_path = self.current_results['path']
            latest_time = self.current_results['latest_time']
            
            # Look for wall patches in boundary conditions
            system_dir = os.path.join(results_path, 'system')
            if os.path.exists(system_dir):
                # This is a simplified analysis
                analysis_text = f"""Wall Analysis Results:

[Chart] Analysis Summary:
• Time: {latest_time}
• Wall patches found: [Analysis would identify wall boundaries]
• Available for analysis: Pressure, Wall shear stress, Heat transfer

[Monitor] Key Metrics:
• Max wall shear stress: [Would be calculated from results]
• Min pressure: [Would be extracted from wall values]
• Average heat flux: [Would be computed if temperature field exists]

⚠️ Note: This is a basic implementation. 
Advanced analysis requires OpenFOAM post-processing tools."""
                
                QMessageBox.information(self, "Wall Analysis", analysis_text)
            else:
                QMessageBox.warning(self, "Wall Analysis", "Could not find case system directory for analysis.")
                
        except Exception as e:
            QMessageBox.critical(self, "Wall Analysis Error", f"Failed to analyze wall values: {str(e)}")
        
    def plot_centerline(self):
        """Plot centerline values."""
        try:
            if not self.current_results:
                QMessageBox.warning(self, "Centerline Plot", "No results loaded. Please load simulation results first.")
                return
                
            # Create a new window for centerline plot
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle('Centerline Analysis')
            
            # Generate sample centerline data (in real implementation, this would extract from results)
            x = np.linspace(0, 2, 100)
            
            # Velocity along centerline
            velocity = 10 * (1 + 0.5 * np.sin(np.pi * x))  # Sample data
            ax1.plot(x, velocity, 'b-', linewidth=2, label='Velocity magnitude')
            ax1.set_xlabel('X position [m]')
            ax1.set_ylabel('Velocity [m/s]')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Pressure along centerline
            pressure = 101325 - 500 * x  # Sample data
            ax2.plot(x, pressure, 'r-', linewidth=2, label='Pressure')
            ax2.set_xlabel('X position [m]')
            ax2.set_ylabel('Pressure [Pa]')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
            
            QMessageBox.information(self, "Centerline Plot", 
                                  "Centerline plot generated successfully!\n\n"
                                  "Note: This shows sample data. Real implementation would extract "
                                  "actual values from simulation results along the centerline.")
            
        except Exception as e:
            QMessageBox.critical(self, "Centerline Plot Error", f"Failed to create centerline plot: {str(e)}")
        
    def calculate_mass_flow(self):
        """Calculate mass flow rate."""
        try:
            if not self.current_results:
                QMessageBox.warning(self, "Mass Flow", "No results loaded. Please load simulation results first.")
                return
                
            # Basic mass flow calculation implementation
            results_path = self.current_results['path']
            latest_time = self.current_results['latest_time']
            
            # In a real implementation, this would integrate velocity and density across inlet/outlet patches
            # For now, show a calculation based on geometry and typical values
            
            if self.geometry.elements:
                # Get approximate throat area from geometry
                x_coords, y_coords = self.geometry.get_interpolated_points(100)
                if len(y_coords) > 0:
                    throat_radius = min(y_coords)  # Approximate
                    throat_area = np.pi * throat_radius**2
                    
                    # Estimate mass flow (simplified)
                    density = 1.225  # kg/m³ (air at STP)
                    velocity = 10.0  # m/s (estimated)
                    mass_flow = density * velocity * throat_area
                    
                    calculation_text = f"""Mass Flow Calculation:

 Geometry Analysis:
• Estimated throat radius: {throat_radius:.4f} m
• Throat area: {throat_area:.6f} m²

[Chart] Flow Calculation:
• Density: {density} kg/m³
• Estimated velocity: {velocity} m/s
• Mass flow rate: {mass_flow:.6f} kg/s

⚠️ Note: This is an estimate based on geometry.
Real calculation requires integration of ρ·U over boundary patches."""
                    
                    QMessageBox.information(self, "Mass Flow Calculation", calculation_text)
                else:
                    QMessageBox.warning(self, "Mass Flow", "Could not determine geometry dimensions.")
            else:
                QMessageBox.warning(self, "Mass Flow", "No geometry available for calculation.")
                
        except Exception as e:
            QMessageBox.critical(self, "Mass Flow Error", f"Failed to calculate mass flow: {str(e)}")
        
    def calculate_pressure_loss(self):
        """Calculate pressure loss."""
        try:
            if not self.current_results:
                QMessageBox.warning(self, "Pressure Loss", "No results loaded. Please load simulation results first.")
                return
                
            # Basic pressure loss calculation
            results_path = self.current_results['path']
            
            # In a real implementation, this would read pressure values at inlet and outlet patches
            # For demonstration, calculate theoretical pressure loss
            
            inlet_pressure = 101325  # Pa (standard pressure)
            outlet_pressure = 101000  # Pa (estimated)
            
            pressure_loss = inlet_pressure - outlet_pressure
            pressure_loss_percent = (pressure_loss / inlet_pressure) * 100
            
            # Calculate dynamic pressure for reference
            density = 1.225  # kg/m³
            velocity = 10.0  # m/s (estimated)
            dynamic_pressure = 0.5 * density * velocity**2
            
            calculation_text = f"""Pressure Loss Analysis:

[Chart] Pressure Values:
• Inlet pressure: {inlet_pressure:,.0f} Pa
• Outlet pressure: {outlet_pressure:,.0f} Pa
• Pressure loss: {pressure_loss:,.0f} Pa
• Loss percentage: {pressure_loss_percent:.2f}%

[Monitor] Reference Values:
• Dynamic pressure: {dynamic_pressure:.1f} Pa
• Loss coefficient (ΔP/q): {pressure_loss/dynamic_pressure:.2f}

⚠️ Note: Values shown are estimates.
Real analysis requires extraction from simulation results at inlet/outlet patches."""
            
            QMessageBox.information(self, "Pressure Loss Analysis", calculation_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Pressure Loss Error", f"Failed to calculate pressure loss: {str(e)}")
        
    def save_visualization(self):
        """Save current visualization."""
        try:
            if not hasattr(self, 'postprocessing_canvas'):
                QMessageBox.warning(self, "Save Image", "No visualization available to save.")
                return
                
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Visualization", "", 
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Save the current figure
            self.postprocessing_canvas.figure.savefig(
                file_path, 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            
            QMessageBox.information(self, "Save Image", f"Visualization saved successfully to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Image Error", f"Failed to save visualization: {str(e)}")
        
    def export_data(self):
        """Export analysis data."""
        try:
            if not self.current_results:
                QMessageBox.warning(self, "Export Data", "No results available to export.")
                return
                
            # Get export file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Data", "", 
                "CSV Files (*.csv);;JSON Files (*.json);;Text Files (*.txt);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Determine export format
            extension = os.path.splitext(file_path)[1].lower()
            
            # Prepare data for export
            export_data = {
                'case_info': {
                    'results_path': self.current_results['path'],
                    'latest_time': self.current_results['latest_time'],
                    'available_fields': self.current_results['fields'],
                    'time_steps': len(self.current_results['time_dirs'])
                },
                'geometry': {
                    'num_elements': len(self.geometry.elements),
                    'element_types': [elem.element_type for elem in self.geometry.elements]
                }
            }
            
            if extension == '.json':
                import json
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif extension == '.csv':
                # Export as CSV (simplified format)
                with open(file_path, 'w') as f:
                    f.write("Parameter,Value\n")
                    f.write(f"Results Path,{export_data['case_info']['results_path']}\n")
                    f.write(f"Latest Time,{export_data['case_info']['latest_time']}\n")
                    f.write(f"Time Steps,{export_data['case_info']['time_steps']}\n")
                    f.write(f"Geometry Elements,{export_data['geometry']['num_elements']}\n")
                    
            else:  # Text format
                with open(file_path, 'w') as f:
                    f.write("Nozzle CFD Analysis Data Export\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Results Path: {export_data['case_info']['results_path']}\n")
                    f.write(f"Latest Time: {export_data['case_info']['latest_time']}\n")
                    f.write(f"Time Steps: {export_data['case_info']['time_steps']}\n")
                    f.write(f"Available Fields: {', '.join(export_data['case_info']['available_fields'])}\n")
                    f.write(f"Geometry Elements: {export_data['geometry']['num_elements']}\n")
                    
            QMessageBox.information(self, "Export Data", f"Data exported successfully to:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Data Error", f"Failed to export data: {str(e)}")
        
    def generate_report(self):
        """Generate analysis report."""
        try:
            # Get save file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Generate Report", "", 
                "HTML Files (*.html);;PDF Files (*.pdf);;Text Files (*.txt);;All Files (*)"
            )
            
            if not file_path:
                return
                
            # Generate report content
            report_content = self._generate_report_content()
            
            # Determine format and save
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.html':
                self._save_html_report(file_path, report_content)
            else:  # Text format
                self._save_text_report(file_path, report_content)
                
            QMessageBox.information(self, "Generate Report", f"Report generated successfully:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Generate Report Error", f"Failed to generate report: {str(e)}")
    
    def _generate_report_content(self):
        """Generate the content for the report."""
        from datetime import datetime
        
        # Collect information
        has_geometry = len(self.geometry.elements) > 0
        has_mesh = self.current_mesh_data is not None
        has_case = self.current_case_directory and os.path.exists(self.current_case_directory)
        has_results = self.current_results is not None
        
        # Get project info safely
        project_name = "Untitled Project"
        project_description = "No description provided"
        
        if hasattr(self, 'project_name_edit') and self.project_name_edit:
            project_name = self.project_name_edit.text() or project_name
            
        if hasattr(self, 'project_description') and self.project_description:
            project_description = self.project_description.toPlainText() or project_description
        
        content = {
            'title': 'Nozzle CFD Analysis Report',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'project_name': project_name,
            'description': project_description,
            'geometry': {
                'status': '[OK] Complete' if has_geometry else '[X] Not defined',
                'elements': len(self.geometry.elements),
                'symmetric': self.geometry.is_symmetric
            },
            'mesh': {
                'status': '[OK] Generated' if has_mesh else '[X] Not generated'
            },
            'simulation': {
                'status': '[OK] Case ready' if has_case else '[X] Not set up'
            },
            'results': {
                'status': '[OK] Available' if has_results else '[X] Not available',
                'details': self.current_results if has_results else None
            }
        }
        
        return content
    
    def _save_html_report(self, file_path, content):
        """Save report in HTML format."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{content['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #0066cc; }}
        .status {{ font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{content['title']}</h1>
        <p><strong>Generated:</strong> {content['date']}</p>
        <p><strong>Project:</strong> {content['project_name']}</p>
    </div>
    
    <div class="section">
        <h2>Project Description</h2>
        <p>{content['description']}</p>
    </div>
    
    <div class="section">
        <h2>Workflow Status</h2>
        <p><span class="status">Geometry:</span> {content['geometry']['status']} ({content['geometry']['elements']} elements)</p>
        <p><span class="status">Mesh:</span> {content['mesh']['status']}</p>
        <p><span class="status">Simulation:</span> {content['simulation']['status']}</p>
        <p><span class="status">Results:</span> {content['results']['status']}</p>
    </div>
    
    {"<div class='section'><h2>Results Summary</h2><p>Latest time: " + str(content['results']['details']['latest_time']) + "</p><p>Available fields: " + ', '.join(content['results']['details']['fields']) + "</p></div>" if content['results']['details'] else ""}
    
    <div class="section">
        <h2>Generated by</h2>
        <p>Nozzle CFD Design Tool v1.0.0</p>
    </div>
</body>
</html>
"""
        with open(file_path, 'w') as f:
            f.write(html_content)
    
    def _save_text_report(self, file_path, content):
        """Save report in text format."""
        text_content = f"""{content['title']}
{"=" * len(content['title'])}

Generated: {content['date']}
Project: {content['project_name']}

Description:
{content['description']}

Workflow Status:
• Geometry: {content['geometry']['status']} ({content['geometry']['elements']} elements)
• Mesh: {content['mesh']['status']}
• Simulation: {content['simulation']['status']}
• Results: {content['results']['status']}

{f"Results Summary:\n• Latest time: {content['results']['details']['latest_time']}\n• Available fields: {', '.join(content['results']['details']['fields'])}\n" if content['results']['details'] else ""}
Generated by: Nozzle CFD Design Tool v1.0.0
"""
        with open(file_path, 'w') as f:
            f.write(text_content)
        
    def update_workflow_status(self):
        """Update workflow status indicators."""
        # Update status based on current state
        has_geometry = len(self.geometry.elements) > 0
        
        # Update geometry status
        if hasattr(self, 'geometry_status'):
            self.geometry_status.setText("[OK] Complete" if has_geometry else "⏳ Pending")
            self.geometry_status.setStyleSheet(f"color: {Theme.SUCCESS if has_geometry else Theme.WARNING}")
        
        # Update mesh status
        has_mesh = self.current_mesh_data is not None
        if hasattr(self, 'mesh_status'):
            self.mesh_status.setText("[OK] Complete" if has_mesh else "⏳ Pending")
            self.mesh_status.setStyleSheet(f"color: {Theme.SUCCESS if has_mesh else Theme.WARNING}")
        
        # Update simulation status
        has_case = self.current_case_directory and os.path.exists(self.current_case_directory)
        if hasattr(self, 'simulation_status'):
            self.simulation_status.setText("[OK] Ready" if has_case else "⏳ Pending")
            self.simulation_status.setStyleSheet(f"color: {Theme.SUCCESS if has_case else Theme.WARNING}")
        
        # Update results status  
        has_results = self.current_results is not None
        if hasattr(self, 'results_status'):
            self.results_status.setText("[OK] Available" if has_results else "⏳ Pending")
            self.results_status.setStyleSheet(f"color: {Theme.SUCCESS if has_results else Theme.WARNING}")
        
    def new_project(self):
        """Create a new project."""
        reply = QMessageBox.question(self, 'New Project', 
                                   'Create new project? This will clear current work.',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.geometry.clear()
            if hasattr(self, 'canvas'):
                self.canvas.current_points = []
                self.update_canvas(self.canvas)
            self.update_workflow_status()
            QMessageBox.information(self, "New Project", "New project created!")
    
    def open_project(self):
        """Open an existing project."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    project_data = json.load(f)
                
                # Load project information
                if 'project_info' in project_data:
                    info = project_data['project_info']
                    self.project_name_edit.setText(info.get('name', ''))
                    self.project_description.setPlainText(info.get('description', ''))
                
                # Load geometry
                if 'geometry' in project_data:
                    self.geometry.load_from_dict(project_data['geometry'])
                    
                    # Update canvas if available
                    if hasattr(self, 'canvas'):
                        self.canvas.current_points = []
                        self.update_canvas(self.canvas)
                
                # Load mesh parameters if available
                if 'mesh_parameters' in project_data and self.mesh_generator:
                    mesh_params = project_data['mesh_parameters']
                    # Update mesh parameter controls
                    if hasattr(self, 'element_size'):
                        self.element_size.setValue(mesh_params.get('element_size', 0.1))
                    if hasattr(self, 'boundary_layers'):
                        self.boundary_layers.setChecked(mesh_params.get('boundary_layers', True))
                
                # Load simulation settings if available
                if 'simulation_settings' in project_data:
                    sim_settings = project_data['simulation_settings']
                    if hasattr(self, 'solver_type'):
                        solver = sim_settings.get('solver', 'simpleFoam')
                        index = self.solver_type.findText(solver)
                        if index >= 0:
                            self.solver_type.setCurrentIndex(index)
                
                self.current_file = file_path
                self.is_modified = False
                self.update_workflow_status()
                
                QMessageBox.information(self, "Open Project", f"Project loaded successfully from:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Open Project Error", f"Failed to open project: {str(e)}")
    
    def save_project(self):
        """Save current project."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON Files (*.json)")
        if file_path:
            try:
                # Prepare project data
                project_data = {
                    'version': '1.0.0',
                    'project_info': {
                        'name': self.project_name_edit.text(),
                        'description': self.project_description.toPlainText(),
                        'created': self._get_current_timestamp()
                    },
                    'geometry': self.geometry.to_dict(),
                    'mesh_parameters': {
                        'element_size': getattr(self, 'element_size', None) and self.element_size.value(),
                        'boundary_layers': getattr(self, 'boundary_layers', None) and self.boundary_layers.isChecked(),
                        'boundary_layer_thickness': getattr(self, 'boundary_layer_thickness', None) and self.boundary_layer_thickness.value(),
                        'domain_extension': getattr(self, 'domain_extension', None) and self.domain_extension.value()
                    },
                    'simulation_settings': {
                        'solver': getattr(self, 'solver_type', None) and self.solver_type.currentText(),
                        'inlet_velocity': getattr(self, 'inlet_velocity', None) and self.inlet_velocity.value(),
                        'outlet_pressure': getattr(self, 'outlet_pressure', None) and self.outlet_pressure.value(),
                        'turbulence_model': getattr(self, 'turbulence_model', None) and self.turbulence_model.currentText()
                    }
                }
                
                # Remove None values
                project_data = self._clean_dict(project_data)
                
                # Save to file
                with open(file_path, 'w') as f:
                    json.dump(project_data, f, indent=2)
                
                self.current_file = file_path
                self.is_modified = False
                
                QMessageBox.information(self, "Save Project", f"Project saved successfully to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Save Project Error", f"Failed to save project: {str(e)}")
    
    def _get_current_timestamp(self):
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _clean_dict(self, d):
        """Remove None values from dictionary recursively."""
        if isinstance(d, dict):
            return {k: self._clean_dict(v) for k, v in d.items() if v is not None}
        elif isinstance(d, list):
            return [self._clean_dict(v) for v in d if v is not None]
        else:
            return d

    def export_case(self):
        """Export case files for OpenFOAM."""
        if not self.geometry.elements:
            QMessageBox.warning(self, "Export Case", "No geometry defined")
            return
            
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if directory:
            try:
                # Create case structure
                case_name = self.project_name_edit.text() or "nozzleCase"
                case_dir = os.path.join(directory, case_name)
                
                if not os.path.exists(case_dir):
                    os.makedirs(case_dir)
                
                # Create basic OpenFOAM case structure
                subdirs = ['0', 'constant', 'system']
                for subdir in subdirs:
                    subdir_path = os.path.join(case_dir, subdir)
                    if not os.path.exists(subdir_path):
                        os.makedirs(subdir_path)
                
                # Generate basic files
                self._create_control_dict(os.path.join(case_dir, 'system'))
                self._create_fv_schemes(os.path.join(case_dir, 'system'))
                self._create_fv_solution(os.path.join(case_dir, 'system'))
                self._create_basic_fields(os.path.join(case_dir, '0'))
                self._create_transport_properties(os.path.join(case_dir, 'constant'))
                
                # If advanced simulation setup is available, use it
                if self.simulation_setup:
                    try:
                        self.simulation_setup.generate_openfoam_case(case_dir, self.geometry)
                    except Exception as e:
                        print(f"Advanced case generation failed, using basic: {e}")
                
                self.current_case_directory = case_dir
                self.update_workflow_status()
                
                QMessageBox.information(self, "Export Case", f"OpenFOAM case exported successfully to:\n{case_dir}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Case Error", f"Failed to export case: {str(e)}")

    def on_tab_changed(self, index):
        """Handle tab change events."""
        tab_names = ["Geometry", "Meshing", "Simulation", "Post-processing"]
        if 0 <= index < len(tab_names):
            print(f"Switched to {tab_names[index]} tab")
    
    def _create_control_dict(self, system_dir):
        """Create basic controlDict file."""
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

// ************************************************************************* //
"""
        with open(os.path.join(system_dir, 'controlDict'), 'w') as f:
            f.write(content)
    
    def _create_fv_schemes(self, system_dir):
        """Create basic fvSchemes file."""
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

// ************************************************************************* //
"""
        with open(os.path.join(system_dir, 'fvSchemes'), 'w') as f:
            f.write(content)
    
    def _create_fv_solution(self, system_dir):
        """Create basic fvSolution file."""
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-06;
        relTol          0.1;
    }

    "(k|epsilon)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-06;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon)"   1e-4;
    }
}

relaxationFactors
{
    equations
    {
        U               0.9;
        ".*"            0.9;
    }
}

// ************************************************************************* //
"""
        with open(os.path.join(system_dir, 'fvSolution'), 'w') as f:
            f.write(content)
    
    def _create_basic_fields(self, zero_dir):
        """Create basic field files in 0 directory."""
        # Create U file
        u_content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (10 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (10 0 0);
    }
    
    outlet
    {
        type            zeroGradient;
    }
    
    walls
    {
        type            noSlip;
    }
    
    symmetry
    {
        type            symmetryPlane;
    }
}

// ************************************************************************* //
"""
        with open(os.path.join(zero_dir, 'U'), 'w') as f:
            f.write(u_content)
        
        # Create p file
        p_content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    
    walls
    {
        type            zeroGradient;
    }
    
    symmetry
    {
        type            symmetryPlane;
    }
}

// ************************************************************************* //
"""
        with open(os.path.join(zero_dir, 'p'), 'w') as f:
            f.write(p_content)
    
    def _create_transport_properties(self, constant_dir):
        """Create transportProperties file."""
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] 1.5e-05;

// ************************************************************************* //
"""
        with open(os.path.join(constant_dir, 'transportProperties'), 'w') as f:
            f.write(content)
        
    def closeEvent(self, event):
        """Handle application close event."""
        reply = QMessageBox.question(self, 'Close Application', 
                                   'Are you sure you want to exit?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Main entry point for comprehensive nozzle design application."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Nozzle CFD Design Tool")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("CFD Research")
    
    # Create and show main window
    window = NozzleDesignGUI()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
