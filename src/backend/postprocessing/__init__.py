"""
Postprocessing module for SU2 CFD results analysis.

Contains results processing, visualization, and interactive postprocessor widget.
"""

from backend.postprocessing.postprocessing import (
    ResultsProcessor,
    VisualizationSettings,
    FieldType,
    PlotType,
    ProbeData,
)
from backend.postprocessing.su2_case_analyzer import SU2Case, parse_history_csv
from backend.postprocessing.interactive_postprocessor import InteractivePostprocessorWidget

__all__ = [
    'ResultsProcessor',
    'VisualizationSettings',
    'FieldType',
    'PlotType',
    'ProbeData',
    'SU2Case',
    'parse_history_csv',
    'InteractivePostprocessorWidget',
]
