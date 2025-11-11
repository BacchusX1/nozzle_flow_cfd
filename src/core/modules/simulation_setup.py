"""
Simulation Setup Module

Handles CFD simulation configuration, boundary conditions,
solver settings, and OpenFOAM case generation.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class BoundaryType(Enum):
    """Boundary condition types."""
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    EMPTY = "empty"


class SolverType(Enum):
    """Available CFD solvers."""
    SIMPLE_FOAM = "simpleFoam"
    PISO_FOAM = "pisoFoam"
    PIMPLE_FOAM = "pimpleFoam"
    RHOSIMPLE_FOAM = "rhoSimpleFoam"


@dataclass
class FluidProperties:
    """Fluid properties for simulation."""
    density: float = 1.225  # kg/m³ (air at STP)
    viscosity: float = 1.81e-5  # Pa·s (air at STP)
    temperature: float = 300.0  # K
    pressure: float = 101325.0  # Pa
    compressible: bool = False
    turbulent: bool = True


@dataclass
class BoundaryCondition:
    """Boundary condition definition."""
    name: str
    boundary_type: BoundaryType
    
    # Velocity conditions
    velocity_magnitude: float = 0.0
    velocity_direction: Tuple[float, float] = (1.0, 0.0)
    velocity_profile: str = "uniform"  # uniform, parabolic, custom
    
    # Pressure conditions
    pressure_value: float = 0.0
    pressure_type: str = "fixedValue"  # fixedValue, zeroGradient, totalPressure
    
    # Turbulence conditions
    turbulence_intensity: float = 0.05
    turbulence_length_scale: float = 0.1
    
    # Wall conditions
    wall_roughness: float = 0.0
    wall_function: str = "nutWallFunction"


@dataclass
class SolverSettings:
    """Solver configuration."""
    solver_type: SolverType = SolverType.SIMPLE_FOAM
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    time_step: float = 0.001
    end_time: float = 1.0
    write_interval: int = 100
    
    # Parallel processing
    n_processors: int = 1  # Number of CPU cores to use (1 = serial)
    decomposition_method: str = "scotch"  # scotch, simple, hierarchical, manual
    
    # Relaxation factors
    pressure_relaxation: float = 0.3
    velocity_relaxation: float = 0.7
    turbulence_relaxation: float = 0.7
    
    # Solution schemes
    pressure_solver: str = "GAMG"
    velocity_solver: str = "smoothSolver"
    turbulence_solver: str = "smoothSolver"


@dataclass
class TurbulenceModel:
    """Turbulence model settings."""
    enabled: bool = True
    model_type: str = "kEpsilon"  # kEpsilon, kOmegaSST, LES, laminar
    wall_functions: bool = True
    
    # Model coefficients (k-epsilon defaults)
    cmu: float = 0.09
    c1: float = 1.44
    c2: float = 1.92
    sigma_k: float = 1.0
    sigma_epsilon: float = 1.3


class SimulationSetup:
    """Main simulation setup and case generation."""
    
    def __init__(self):
        self.fluid_properties = FluidProperties()
        self.boundary_conditions: Dict[str, BoundaryCondition] = {}
        self.solver_settings = SolverSettings()
        self.turbulence_model = TurbulenceModel()
        self.case_directory = ""
        self.mesh_data = None
        
    def add_boundary_condition(self, bc: BoundaryCondition):
        """Add or update boundary condition."""
        self.boundary_conditions[bc.name] = bc
        
    def remove_boundary_condition(self, name: str):
        """Remove boundary condition."""
        if name in self.boundary_conditions:
            del self.boundary_conditions[name]
            
    def get_boundary_condition(self, name: str) -> Optional[BoundaryCondition]:
        """Get boundary condition by name."""
        return self.boundary_conditions.get(name)
        
    def set_inlet_conditions(self, boundary_name: str, velocity: float, 
                           direction: Tuple[float, float] = (1.0, 0.0),
                           turbulence_intensity: float = 0.05):
        """Set inlet boundary conditions."""
        bc = BoundaryCondition(
            name=boundary_name,
            boundary_type=BoundaryType.INLET,
            velocity_magnitude=velocity,
            velocity_direction=direction,
            turbulence_intensity=turbulence_intensity,
            pressure_type="zeroGradient"
        )
        self.add_boundary_condition(bc)
        
    def set_outlet_conditions(self, boundary_name: str, pressure: float = 0.0):
        """Set outlet boundary conditions."""
        bc = BoundaryCondition(
            name=boundary_name,
            boundary_type=BoundaryType.OUTLET,
            pressure_value=pressure,
            pressure_type="fixedValue",
            velocity_magnitude=0.0  # Will use zeroGradient
        )
        self.add_boundary_condition(bc)
        
    def set_wall_conditions(self, boundary_name: str, roughness: float = 0.0):
        """Set wall boundary conditions."""
        bc = BoundaryCondition(
            name=boundary_name,
            boundary_type=BoundaryType.WALL,
            wall_roughness=roughness,
            velocity_magnitude=0.0  # No-slip
        )
        self.add_boundary_condition(bc)
        
    def generate_case_files(self, geometry, mesh_data: Dict = None):
        """Generate OpenFOAM case files from geometry and mesh data.
        
        This method is called by the GUI and wraps generate_openfoam_case.
        """
        if not self.case_directory:
            raise ValueError("Case directory not set")
            
        # Store geometry and mesh data
        self.geometry = geometry
        self.mesh_data = mesh_data
        
        # Generate the OpenFOAM case
        return self.generate_openfoam_case(self.case_directory, mesh_data)
    
    def generate_openfoam_case(self, case_directory: str, mesh_data: Dict = None):
        """Generate complete OpenFOAM case."""
        self.case_directory = case_directory
        self.mesh_data = mesh_data
        
        # Create directory structure
        self._create_case_structure()
        
        # Generate system files
        self._write_control_dict()
        self._write_fv_schemes()
        self._write_fv_solution()
        
        # Generate constant files
        self._write_transport_properties()
        self._write_turbulence_properties()
        
        # Generate initial conditions
        self._write_initial_conditions()
        
        # Generate boundary conditions
        self._write_boundary_conditions()
        
        # Generate mesh configuration
        self._write_block_mesh_dict()
        
        # Generate parallel decomposition settings if needed
        if self.solver_settings.n_processors > 1:
            self._write_decompose_par_dict()
        
        # Copy mesh if provided
        if mesh_data:
            self._setup_mesh()
            
        return case_directory
        
    def _create_case_structure(self):
        """Create OpenFOAM case directory structure."""
        directories = [
            "0",
            "constant",
            "constant/polyMesh",
            "system"
        ]
        
        for directory in directories:
            os.makedirs(os.path.join(self.case_directory, directory), exist_ok=True)
            
    def _write_control_dict(self):
        """Write system/controlDict file."""
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     {self.solver_settings.solver_type.value};

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         {self.solver_settings.end_time};

deltaT          {self.solver_settings.time_step};

writeControl    runTime;

writeInterval   {self.solver_settings.write_interval * self.solver_settings.time_step};

purgeWrite      0;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "system", "controlDict"), 'w') as f:
            f.write(content)
            
    def _write_fv_schemes(self):
        """Write system/fvSchemes file."""
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
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
    grad(p)         Gauss linear;
    grad(U)         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear orthogonal;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         orthogonal;
}

wallDist
{
    method meshWave;
}

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "system", "fvSchemes"), 'w') as f:
            f.write(content)
            
    def _write_fv_solution(self):
        """Write system/fvSolution file."""
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{{
    p
    {{
        solver          {self.solver_settings.pressure_solver};
        tolerance       {self.solver_settings.convergence_tolerance};
        relTol          0.01;
        smoother        GaussSeidel;
        nPreSweeps      0;
        nPostSweeps     2;
        nFinestSweeps   2;
        cacheAgglomeration true;
        nCellsInCoarsestLevel 10;
        agglomerator    faceAreaPair;
        mergeLevels     1;
    }}

    U
    {{
        solver          {self.solver_settings.velocity_solver};
        smoother        symGaussSeidel;
        tolerance       {self.solver_settings.convergence_tolerance};
        relTol          0.01;
    }}

    k
    {{
        solver          {self.solver_settings.turbulence_solver};
        smoother        symGaussSeidel;
        tolerance       {self.solver_settings.convergence_tolerance};
        relTol          0.01;
    }}

    epsilon
    {{
        solver          {self.solver_settings.turbulence_solver};
        smoother        symGaussSeidel;
        tolerance       {self.solver_settings.convergence_tolerance};
        relTol          0.01;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    residualControl
    {{
        p               {self.solver_settings.convergence_tolerance};
        U               {self.solver_settings.convergence_tolerance};
        k               {self.solver_settings.convergence_tolerance};
        epsilon         {self.solver_settings.convergence_tolerance};
    }}
}}

relaxationFactors
{{
    fields
    {{
        p               {self.solver_settings.pressure_relaxation};
    }}
    equations
    {{
        U               {self.solver_settings.velocity_relaxation};
        k               {self.solver_settings.turbulence_relaxation};
        epsilon         {self.solver_settings.turbulence_relaxation};
    }}
}}

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "system", "fvSolution"), 'w') as f:
            f.write(content)
            
    def _write_transport_properties(self):
        """Write constant/transportProperties file."""
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              nu [0 2 -1 0 0 0 0] {self.fluid_properties.viscosity / self.fluid_properties.density};

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "constant", "transportProperties"), 'w') as f:
            f.write(content)
            
    def _write_turbulence_properties(self):
        """Write constant/turbulenceProperties file."""
        if not self.turbulence_model.enabled:
            model_type = "laminar"
        else:
            model_type = "RAS"
            
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      turbulenceProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType {model_type};
"""

        if self.turbulence_model.enabled:
            content += f"""
RAS
{{
    RASModel        {self.turbulence_model.model_type};

    turbulence      on;

    printCoeffs     on;

    {self.turbulence_model.model_type}Coeffs
    {{
        Cmu             {self.turbulence_model.cmu};
        C1              {self.turbulence_model.c1};
        C2              {self.turbulence_model.c2};
        sigmaK          {self.turbulence_model.sigma_k};
        sigmaEps        {self.turbulence_model.sigma_epsilon};
    }}
}}
"""

        content += "\n// ************************************************************************* //\n"
        
        with open(os.path.join(self.case_directory, "constant", "turbulenceProperties"), 'w') as f:
            f.write(content)
            
    def _write_initial_conditions(self):
        """Write initial condition files in 0/ directory."""
        # Write U (velocity)
        self._write_velocity_field()
        
        # Write p (pressure)
        self._write_pressure_field()
        
        # Write turbulence fields if enabled
        if self.turbulence_model.enabled:
            self._write_turbulence_fields()
            
    def _write_velocity_field(self):
        """Write 0/U file."""
        # Estimate initial velocity from inlet conditions
        inlet_velocity = (0, 0, 0)
        for bc in self.boundary_conditions.values():
            if bc.boundary_type == BoundaryType.INLET:
                inlet_velocity = (
                    bc.velocity_magnitude * bc.velocity_direction[0],
                    bc.velocity_magnitude * bc.velocity_direction[1],
                    0
                )
                break
                
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({inlet_velocity[0]} {inlet_velocity[1]} {inlet_velocity[2]});

boundaryField
{{
"""

        # Add boundary conditions
        for bc_name, bc in self.boundary_conditions.items():
            if bc.boundary_type == BoundaryType.INLET:
                content += f"""    {bc_name}
    {{
        type            fixedValue;
        value           uniform ({bc.velocity_magnitude * bc.velocity_direction[0]} {bc.velocity_magnitude * bc.velocity_direction[1]} 0);
    }}

"""
            elif bc.boundary_type == BoundaryType.OUTLET:
                content += f"""    {bc_name}
    {{
        type            zeroGradient;
    }}

"""
            elif bc.boundary_type == BoundaryType.WALL:
                content += f"""    {bc_name}
    {{
        type            noSlip;
    }}

"""
            elif bc.boundary_type == BoundaryType.SYMMETRY:
                content += f"""    {bc_name}
    {{
        type            symmetryPlane;
    }}

"""

        content += """}

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "0", "U"), 'w') as f:
            f.write(content)
            
    def _write_pressure_field(self):
        """Write 0/p file."""
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
"""

        # Add boundary conditions
        for bc_name, bc in self.boundary_conditions.items():
            if bc.boundary_type == BoundaryType.INLET:
                content += f"""    {bc_name}
    {{
        type            zeroGradient;
    }}

"""
            elif bc.boundary_type == BoundaryType.OUTLET:
                content += f"""    {bc_name}
    {{
        type            fixedValue;
        value           uniform {bc.pressure_value};
    }}

"""
            elif bc.boundary_type == BoundaryType.WALL:
                content += f"""    {bc_name}
    {{
        type            zeroGradient;
    }}

"""
            elif bc.boundary_type == BoundaryType.SYMMETRY:
                content += f"""    {bc_name}
    {{
        type            symmetryPlane;
    }}

"""

        content += """}

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "0", "p"), 'w') as f:
            f.write(content)
            
    def _write_turbulence_fields(self):
        """Write turbulence field files (k, epsilon)."""
        # Estimate turbulence values from inlet conditions
        inlet_k = 0.01
        inlet_epsilon = 0.001
        
        for bc in self.boundary_conditions.values():
            if bc.boundary_type == BoundaryType.INLET:
                # Calculate turbulence kinetic energy
                inlet_k = 1.5 * (bc.velocity_magnitude * bc.turbulence_intensity) ** 2
                
                # Calculate dissipation rate
                inlet_epsilon = (0.09 ** 0.75) * (inlet_k ** 1.5) / bc.turbulence_length_scale
                break
                
        # Write k field
        k_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      k;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {inlet_k};

boundaryField
{{
"""

        for bc_name, bc in self.boundary_conditions.items():
            if bc.boundary_type == BoundaryType.INLET:
                k_val = 1.5 * (bc.velocity_magnitude * bc.turbulence_intensity) ** 2
                k_content += f"""    {bc_name}
    {{
        type            fixedValue;
        value           uniform {k_val};
    }}

"""
            elif bc.boundary_type == BoundaryType.OUTLET:
                k_content += f"""    {bc_name}
    {{
        type            zeroGradient;
    }}

"""
            elif bc.boundary_type == BoundaryType.WALL:
                k_content += f"""    {bc_name}
    {{
        type            kqRWallFunction;
        value           uniform {inlet_k};
    }}

"""

        k_content += """}

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "0", "k"), 'w') as f:
            f.write(k_content)
            
        # Write epsilon field
        epsilon_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      epsilon;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform {inlet_epsilon};

boundaryField
{{
"""

        for bc_name, bc in self.boundary_conditions.items():
            if bc.boundary_type == BoundaryType.INLET:
                eps_val = (0.09 ** 0.75) * ((1.5 * (bc.velocity_magnitude * bc.turbulence_intensity) ** 2) ** 1.5) / bc.turbulence_length_scale
                epsilon_content += f"""    {bc_name}
    {{
        type            fixedValue;
        value           uniform {eps_val};
    }}

"""
            elif bc.boundary_type == BoundaryType.OUTLET:
                epsilon_content += f"""    {bc_name}
    {{
        type            zeroGradient;
    }}

"""
            elif bc.boundary_type == BoundaryType.WALL:
                epsilon_content += f"""    {bc_name}
    {{
        type            epsilonWallFunction;
        value           uniform {inlet_epsilon};
    }}

"""

        epsilon_content += """}

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "0", "epsilon"), 'w') as f:
            f.write(epsilon_content)
            
        # Write nut field (turbulent viscosity)
        inlet_nut = 0.0  # Default for walls and calculated fields
        
        nut_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      nut;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform {inlet_nut};

boundaryField
{{
"""

        for bc_name, bc in self.boundary_conditions.items():
            if bc.boundary_type == BoundaryType.INLET:
                nut_content += f"""    {bc_name}
    {{
        type            calculated;
        value           uniform {inlet_nut};
    }}

"""
            elif bc.boundary_type == BoundaryType.OUTLET:
                nut_content += f"""    {bc_name}
    {{
        type            calculated;
        value           uniform {inlet_nut};
    }}

"""
            elif bc.boundary_type == BoundaryType.WALL:
                nut_content += f"""    {bc_name}
    {{
        type            nutkWallFunction;
        value           uniform {inlet_nut};
    }}

"""

        nut_content += """}

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "0", "nut"), 'w') as f:
            f.write(nut_content)
            
    def _write_boundary_conditions(self):
        """Write constant/polyMesh/boundary file."""
        # This would be generated based on mesh data
        # For now, create a template
        pass
        
    def _write_block_mesh_dict(self):
        """Write system/blockMeshDict file for mesh generation.
        
        If mesh_data is available, use it to create a nozzle-conforming mesh.
        Otherwise, fall back to simple rectangular mesh.
        """
        
        # Debug output
        print(f"DEBUG: mesh_data available: {self.mesh_data is not None}")
        if self.mesh_data:
            print(f"DEBUG: mesh_data keys: {self.mesh_data.keys()}")
            print(f"DEBUG: mesh_data has 'nodes': {'nodes' in self.mesh_data}")
        
        # Try to use mesh_data if available
        if self.mesh_data and 'nodes' in self.mesh_data:
            print("DEBUG: Using mesh_data to generate nozzle-conforming blockMeshDict")
            self._write_block_mesh_dict_from_mesh()
        else:
            print("DEBUG: Using fallback rectangular blockMeshDict (no mesh_data available)")
            self._write_simple_block_mesh_dict()
    
    def _write_block_mesh_dict_from_mesh(self):
        """Generate blockMeshDict from actual mesh geometry."""
        import numpy as np
        
        nodes = np.array(self.mesh_data['nodes'])
        
        # Extract unique X and Y coordinates to determine mesh structure
        x_coords = sorted(set(nodes[:, 0]))
        
        # Find upper and lower walls at each x location
        upper_wall = []
        lower_wall = []
        
        for x in x_coords:
            points_at_x = nodes[nodes[:, 0] == x]
            if len(points_at_x) > 0:
                y_max = points_at_x[:, 1].max()
                y_min = points_at_x[:, 1].min()
                upper_wall.append((x, y_max))
                lower_wall.append((x, y_min))
        
        upper_wall = np.array(upper_wall)
        lower_wall = np.array(lower_wall)
        
        # Determine mesh resolution
        nx = len(x_coords)
        # Count unique y values to estimate ny
        y_coords = sorted(set(nodes[:, 1]))
        ny = len(y_coords)
        
        # Use reasonable resolution (at least 20x10 for nozzle)
        nx = max(nx, 30)
        ny = max(ny // 2, 10)
        
        print(f"Generating blockMeshDict: {nx}x{ny} cells")
        print(f"Nozzle domain: X=[{upper_wall[0, 0]:.3f}, {upper_wall[-1, 0]:.3f}]")
        print(f"              Y=[{lower_wall[:, 1].min():.3f}, {upper_wall[:, 1].max():.3f}]")
        
        # Create vertices following nozzle contour
        # We'll create a structured multi-block mesh
        x_min, x_max = upper_wall[0, 0], upper_wall[-1, 0]
        z_front, z_back = 0.0, 0.1  # Thin 3D mesh for 2D simulation
        
        # Build vertices list
        vertices_list = []
        vertex_index = 0
        
        # Front face vertices (z=0)
        # Lower wall
        vertices_list.append(f"    ({x_min:.6f}   {lower_wall[0, 1]:.6f}   {z_front})   // {vertex_index}")
        vertex_index += 1
        vertices_list.append(f"    ({x_max:.6f}   {lower_wall[-1, 1]:.6f}   {z_front})   // {vertex_index}")
        vertex_index += 1
        # Upper wall  
        vertices_list.append(f"    ({x_max:.6f}   {upper_wall[-1, 1]:.6f}   {z_front})   // {vertex_index}")
        vertex_index += 1
        vertices_list.append(f"    ({x_min:.6f}   {upper_wall[0, 1]:.6f}   {z_front})   // {vertex_index}")
        vertex_index += 1
        
        # Back face vertices (z=0.1)
        vertices_list.append(f"    ({x_min:.6f}   {lower_wall[0, 1]:.6f}   {z_back})   // {vertex_index}")
        vertex_index += 1
        vertices_list.append(f"    ({x_max:.6f}   {lower_wall[-1, 1]:.6f}   {z_back})   // {vertex_index}")
        vertex_index += 1
        vertices_list.append(f"    ({x_max:.6f}   {upper_wall[-1, 1]:.6f}   {z_back})   // {vertex_index}")
        vertex_index += 1
        vertices_list.append(f"    ({x_min:.6f}   {upper_wall[0, 1]:.6f}   {z_back})   // {vertex_index}")
        
        vertices_str = "\n".join(vertices_list)
        
        # Create edges that follow nozzle contours
        edges_list = []
        
        # Lower wall spline (front face: vertices 0-1)
        lower_points = "        (\n"
        for x, y in lower_wall[1:-1]:  # Exclude endpoints
            lower_points += f"            ({x:.6f} {y:.6f} {z_front})\n"
        lower_points += "        )"
        edges_list.append(f"    spline 0 1\n{lower_points}")
        
        # Upper wall spline (front face: vertices 3-2)
        upper_points = "        (\n"
        for x, y in upper_wall[1:-1]:  # Exclude endpoints
            upper_points += f"            ({x:.6f} {y:.6f} {z_front})\n"
        upper_points += "        )"
        edges_list.append(f"    spline 3 2\n{upper_points}")
        
        # Lower wall spline (back face: vertices 4-5)
        lower_points_back = "        (\n"
        for x, y in lower_wall[1:-1]:
            lower_points_back += f"            ({x:.6f} {y:.6f} {z_back})\n"
        lower_points_back += "        )"
        edges_list.append(f"    spline 4 5\n{lower_points_back}")
        
        # Upper wall spline (back face: vertices 7-6)
        upper_points_back = "        (\n"
        for x, y in upper_wall[1:-1]:
            upper_points_back += f"            ({x:.6f} {y:.6f} {z_back})\n"
        upper_points_back += "        )"
        edges_list.append(f"    spline 7 6\n{upper_points_back}")
        
        edges_str = "\n    ".join(edges_list)
        
        # Generate blockMeshDict content
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// Nozzle-conforming mesh generated from geometry

convertToMeters 1;

vertices
(
{vertices_str}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1)
);

edges
(
    {edges_str}
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "system", "blockMeshDict"), 'w') as f:
            f.write(content)
        
        print(f"Generated nozzle-conforming blockMeshDict in {self.case_directory}/system/")
    
    def _write_simple_block_mesh_dict(self):
        """Write simple rectangular blockMeshDict (fallback)."""
        content = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

vertices
(
    // Simple rectangular domain for nozzle
    (-1   -1   0)   // 0
    ( 3   -1   0)   // 1
    ( 3    1   0)   // 2
    (-1    1   0)   // 3
    (-1   -1   0.1) // 4
    ( 3   -1   0.1) // 5
    ( 3    1   0.1) // 6
    (-1    1   0.1) // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (40 20 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }
    outlet
    {
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }
    walls
    {
        type wall;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
        );
    }
    frontAndBack
    {
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);

mergePatchPairs
(
);

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "system", "blockMeshDict"), 'w') as f:
            f.write(content)
        
        print("WARNING: No mesh data available, using simple rectangular mesh")
        
    def _write_decompose_par_dict(self):
        """Write system/decomposeParDict file for parallel processing."""
        n_procs = self.solver_settings.n_processors
        method = self.solver_settings.decomposition_method
        
        # For simple method, calculate nx, ny, nz distribution
        if method == "simple":
            # Distribute processors along x-direction primarily (typical for nozzle flow)
            nx = n_procs
            ny = 1
            nz = 1
            
            # Try to make a more balanced distribution for higher processor counts
            if n_procs >= 4:
                import math
                nx = int(math.sqrt(n_procs))
                ny = n_procs // nx
                nz = 1
            
            coeffs_section = f"""
simpleCoeffs
{{
    n               ({nx} {ny} {nz});
    delta           0.001;
}}"""
        elif method == "hierarchical":
            coeffs_section = f"""
hierarchicalCoeffs
{{
    n               ({n_procs} 1 1);
    delta           0.001;
    order           xyz;
}}"""
        elif method == "scotch":
            # Scotch uses automatic load balancing, no weights needed
            coeffs_section = ""
        elif method == "manual":
            coeffs_section = f"""
manualCoeffs
{{
    dataFile        "cellDecomposition";
}}"""
        else:
            # Default to scotch
            method = "scotch"
            coeffs_section = ""
        
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2012                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains {n_procs};

method          {method};
{coeffs_section}

distributed     no;

roots           ();

// ************************************************************************* //
"""
        
        with open(os.path.join(self.case_directory, "system", "decomposeParDict"), 'w') as f:
            f.write(content)
        
        print(f"Generated decomposeParDict for {n_procs} processors using {method} method")
        
    def _setup_mesh(self):
        """Setup mesh files in constant/polyMesh."""
        # This would convert mesh data to OpenFOAM format
        # For now, just save mesh info
        if self.mesh_data:
            import numpy as np
            mesh_info_file = os.path.join(self.case_directory, "constant", "polyMesh", "mesh_info.json")
            
            # Convert numpy arrays to lists for JSON serialization
            mesh_data_serializable = {}
            for key, value in self.mesh_data.items():
                if isinstance(value, np.ndarray):
                    mesh_data_serializable[key] = value.tolist()
                elif isinstance(value, dict):
                    # Handle nested dicts with numpy arrays
                    mesh_data_serializable[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            mesh_data_serializable[key][k] = v.tolist()
                        else:
                            mesh_data_serializable[key][k] = v
                else:
                    mesh_data_serializable[key] = value
            
            with open(mesh_info_file, 'w') as f:
                json.dump(mesh_data_serializable, f, indent=2)
                
    def save_configuration(self, filename: str):
        """Save simulation configuration to file."""
        config = {
            'fluid_properties': asdict(self.fluid_properties),
            'boundary_conditions': {name: asdict(bc) for name, bc in self.boundary_conditions.items()},
            'solver_settings': asdict(self.solver_settings),
            'turbulence_model': asdict(self.turbulence_model)
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
    def load_configuration(self, filename: str):
        """Load simulation configuration from file."""
        with open(filename, 'r') as f:
            config = json.load(f)
            
        # Restore configuration
        self.fluid_properties = FluidProperties(**config['fluid_properties'])
        self.solver_settings = SolverSettings(**config['solver_settings'])
        self.turbulence_model = TurbulenceModel(**config['turbulence_model'])
        
        # Restore boundary conditions
        self.boundary_conditions.clear()
        for name, bc_data in config['boundary_conditions'].items():
            # Convert boundary_type back to enum
            bc_data['boundary_type'] = BoundaryType(bc_data['boundary_type'])
            bc = BoundaryCondition(**bc_data)
            self.boundary_conditions[name] = bc
