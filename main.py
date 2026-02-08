#!/usr/bin/env python3
"""
Nozzle CFD Design Tool - Main Application Launcher

Professional CFD workflow application with immediate drawing capability.
Complete geometry design, meshing, SU2 simulation, and post-processing workflow.

Usage:
    python main.py

Features:
    ✅ Immediate drawing (no start/stop buttons)
    ✅ Advanced meshing with boundary layers
    ✅ SU2 CFD simulation setup
    ✅ Post-processing visualization
    ✅ Professional dark theme interface

Requirements:
    - PySide6, numpy, matplotlib, scipy
    - Optional: gmsh (for advanced meshing)
    - Optional: SU2 (for CFD simulation - https://su2code.github.io/)
"""

import sys
import os

# Add project root to path for proper imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Fix for Wayland/Qt platform issues on Linux
# Force xcb (X11) backend if Wayland fails
if sys.platform == 'linux' and 'QT_QPA_PLATFORM' not in os.environ:
    # Use X11 (xcb) by default to avoid Wayland issues
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    os.environ.setdefault('QT_LOGGING_RULES', 'qt.qpa.plugin=false')

def main():
    """Launch the nozzle CFD design tool."""
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication
        
        # Enable high DPI scaling for multi-resolution support (FHD, QHD, 4K, 8K)
        # These must be set before QApplication is created
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        
        # Create QApplication first
        app = QApplication(sys.argv)
        
        # Set application-wide attributes for consistent cursor and DPI handling
        app.setStyle('Fusion')  # Fusion style works well across all platforms
        
        # Import GUI after QApplication is created
        from frontend.frontend import NozzleDesignGUI
        
        # Create and show main window
        window = NozzleDesignGUI()
        window.showMaximized()
        
        # Run the application
        return app.exec()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure all dependencies are installed:")
        print("  conda install pyside6 matplotlib numpy scipy")
        print("  pip install gmsh")
        return 1
    except Exception as e:
        print(f"❌ Application error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    print("🚀 Starting Nozzle CFD Design Tool...")
    print("✨ Professional CFD Workflow Features:")
    print("  • Immediate geometry drawing")
    print("  • Advanced mesh generation") 
    print("  • CFD simulation setup")
    print("  • Results visualization")
    print()
    
    sys.exit(main())
