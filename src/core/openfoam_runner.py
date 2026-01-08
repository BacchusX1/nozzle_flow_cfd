import os
import subprocess
import shlex
import shutil
from typing import Optional, Tuple

class OpenFOAMRunner:
    """Runs OpenFOAM commands in a specified case directory."""

    def __init__(self, case_directory: str):
        if not os.path.isdir(case_directory):
            raise FileNotFoundError(f"Case directory not found: {case_directory}")
        self.case_directory = case_directory

    def validate_case(self) -> Tuple[bool, str]:
        """Validate minimal OpenFOAM case structure.

        Returns:
            (ok, message)
        """
        required = [
            os.path.join(self.case_directory, "system", "controlDict"),
            os.path.join(self.case_directory, "system", "fvSchemes"),
            os.path.join(self.case_directory, "system", "fvSolution"),
        ]
        missing = [p for p in required if not os.path.exists(p)]

        transport = os.path.join(self.case_directory, "constant", "transportProperties")
        thermophysical = os.path.join(self.case_directory, "constant", "thermophysicalProperties")

        if not os.path.exists(transport) and not os.path.exists(thermophysical):
            missing.append("constant/transportProperties OR constant/thermophysicalProperties")

        if missing:
            return False, f"Missing required case files: {', '.join(os.path.relpath(p, self.case_directory) if os.path.isabs(p) else p for p in missing)}"
        return True, "OK"

    def get_application_from_control_dict(self) -> Optional[str]:
        """Read `application` from system/controlDict.

        Returns None if the file is missing or cannot be parsed.
        """
        path = os.path.join(self.case_directory, "system", "controlDict")
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("//"):
                        continue
                    if stripped.startswith("application"):
                        # e.g. application     simpleFoam;
                        parts = stripped.split()
                        if len(parts) >= 2:
                            value = parts[1].rstrip(";")
                            return value
        except Exception:
            return None
        return None

    def _is_executable_available(self, exe: str) -> bool:
        return shutil.which(exe) is not None

    def _write_log_error(self, log_file: str, message: str) -> None:
        log_path = os.path.join(self.case_directory, log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n--- Preflight Error ---\n{message}\n")

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
                args = shlex.split(command)
                if not args:
                    log.write("\n--- Preflight Error ---\nEmpty command\n")
                    return False
                exe = args[0]
                if not self._is_executable_available(exe):
                    log.write(f"\n--- Preflight Error ---\nExecutable not found on PATH: {exe}\n")
                    return False

                process = subprocess.Popen(
                    args,
                    cwd=self.case_directory,
                    shell=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                process.wait()
                return process.returncode == 0
        except Exception as e:
            with open(log_path, "a") as log:
                log.write(f"\n--- Python Exception ---\n{e}\n")
            return False

    def block_mesh(self) -> bool:
        """Runs the blockMesh utility."""
        ok, msg = self.validate_case()
        # blockMesh can create mesh, but we still require the case skeleton.
        if not ok:
            self._write_log_error("log.blockMesh", msg)
            return False
        return self.run_command("blockMesh", "log.blockMesh")

    def decompose_par(self) -> bool:
        """Runs the decomposePar utility for parallel processing."""
        ok, msg = self.validate_case()
        if not ok:
            self._write_log_error("log.decomposePar", msg)
            return False
        return self.run_command("decomposePar", "log.decomposePar")

    def reconstruct_par(self) -> bool:
        """Runs the reconstructPar utility to reassemble parallel results."""
        ok, msg = self.validate_case()
        if not ok:
            self._write_log_error("log.reconstructPar", msg)
            return False
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
        ok, msg = self.validate_case()
        if not ok:
            self._write_log_error(f"log.{solver}", msg)
            return False

        solver = (solver or "").strip()
        if not solver:
            self._write_log_error("log.solver", "Solver name is empty")
            return False

        if n_processors > 1:
            if not self._is_executable_available("mpirun"):
                self._write_log_error(f"log.{solver}", "mpirun not found on PATH (required for parallel runs)")
                return False
            command = f"mpirun -np {int(n_processors)} {solver} -parallel"
            return self.run_command(command, f"log.{solver}")

        return self.run_command(f"{solver}", f"log.{solver}")
