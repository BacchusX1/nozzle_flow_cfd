"""
Simulation module for SU2 CFD solver integration.

Contains simulation setup, boundary conditions, and SU2 runner.
"""

from backend.simulation.simulation_setup import (
    SimulationSetup,
    BoundaryCondition,
    FluidProperties,
    SolverSettings,
    TurbulenceModelType as TurbulenceModel,
    BoundaryType,
    SolverType,
)
from backend.simulation.su2_runner import SU2Runner

__all__ = [
    'SimulationSetup',
    'BoundaryCondition',
    'FluidProperties',
    'SolverSettings',
    'TurbulenceModel',
    'BoundaryType',
    'SolverType',
    'SU2Runner',
]
