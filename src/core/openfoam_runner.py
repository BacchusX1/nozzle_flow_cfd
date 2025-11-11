import os
import subprocess

class OpenFOAMRunner:
    """Runs OpenFOAM commands in a specified case directory."""

    def __init__(self, case_directory: str):
        if not os.path.isdir(case_directory):
            raise FileNotFoundError(f"Case directory not found: {case_directory}")
        self.case_directory = case_directory

    def run_command(self, command: str, log_file: str = "log.txt"):
        """
        Runs a shell command within the case directory.

        Args:
            command (str): The command to execute (e.g., 'blockMesh').
            log_file (str): The file to which stdout and stderr will be redirected.

        Returns:
            bool: True if the command was successful, False otherwise.
        """
        log_path = os.path.join(self.case_directory, log_file)
        try:
            with open(log_path, "w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.case_directory,
                    shell=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                process.wait()
                return process.returncode == 0
        except Exception as e:
            with open(log_path, "a") as log:
                log.write(f"\n--- Python Exception ---\n{e}\n")
            return False

    def block_mesh(self) -> bool:
        """Runs the blockMesh utility."""
        return self.run_command("blockMesh", "log.blockMesh")

    def decompose_par(self) -> bool:
        """Runs the decomposePar utility for parallel processing."""
        return self.run_command("decomposePar", "log.decomposePar")

    def reconstruct_par(self) -> bool:
        """Runs the reconstructPar utility to reassemble parallel results."""
        return self.run_command("reconstructPar", "log.reconstructPar")

    def run_solver(self, solver: str = "simpleFoam", n_processors: int = 1) -> bool:
        """
        Runs the specified solver (serial or parallel).
        
        Args:
            solver (str): The solver to run (e.g., 'simpleFoam').
            n_processors (int): Number of processors to use. If > 1, runs in parallel.
        
        Returns:
            bool: True if the command was successful, False otherwise.
        """
        if n_processors > 1:
            # Parallel execution with mpirun
            command = f"mpirun -np {n_processors} {solver} -parallel"
            return self.run_command(command, f"log.{solver}")
        else:
            # Serial execution
            return self.run_command(f"{solver}", f"log.{solver}")
