"""
SU2 Runner Module

Runs SU2 commands in a specified case directory.
Handles mesh generation, CFD solving, and solution processing.
"""

import os
import subprocess
import shlex
import shutil
from typing import Optional, Tuple, List


class SU2Runner:
    """Runs SU2 commands in a specified case directory."""

    def __init__(self, case_directory: str):
        if not os.path.isdir(case_directory):
            raise FileNotFoundError(f"Case directory not found: {case_directory}")
        self.case_directory = case_directory
        self.config_file = "config.cfg"

    def set_config_file(self, config_file: str) -> None:
        """Set the configuration file name."""
        self.config_file = config_file

    def get_config_path(self) -> str:
        """Get full path to config file."""
        return os.path.join(self.case_directory, self.config_file)

    def validate_case(self) -> Tuple[bool, str]:
        """Validate minimal SU2 case structure.

        Returns:
            (ok, message)
        """
        config_path = self.get_config_path()
        if not os.path.exists(config_path):
            return False, f"Missing configuration file: {self.config_file}"

        # Check for mesh file referenced in config
        mesh_file = self._get_mesh_filename()
        if mesh_file:
            mesh_path = os.path.join(self.case_directory, mesh_file)
            if not os.path.exists(mesh_path):
                return False, f"Missing mesh file: {mesh_file}"

        return True, "OK"

    def _get_mesh_filename(self) -> Optional[str]:
        """Read mesh filename from config file."""
        config_path = self.get_config_path()
        try:
            with open(config_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("MESH_FILENAME"):
                        # Format: MESH_FILENAME= mesh.su2
                        parts = stripped.split("=", 1)
                        if len(parts) >= 2:
                            return parts[1].strip()
        except Exception:
            pass
        return None

    def get_solver_from_config(self) -> Optional[str]:
        """Read solver type from config file.

        Returns None if the file is missing or cannot be parsed.
        """
        config_path = self.get_config_path()
        try:
            with open(config_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("%"):
                        continue
                    if stripped.startswith("SOLVER"):
                        # Format: SOLVER= RANS
                        parts = stripped.split("=", 1)
                        if len(parts) >= 2:
                            return parts[1].strip()
        except Exception:
            return None
        return None

    def _is_executable_available(self, exe: str) -> bool:
        """Check if executable is available on PATH."""
        return shutil.which(exe) is not None

    def _write_log_error(self, log_file: str, message: str) -> None:
        """Write error message to log file."""
        log_path = os.path.join(self.case_directory, log_file)
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else self.case_directory, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n--- Preflight Error ---\n{message}\n")

    def run_command(self, command: str, log_file: str = "log.txt", extra_env: dict = None) -> bool:
        """
        Runs a shell command within the case directory.

        Args:
            command (str): The command to execute.
            log_file (str): The file to which stdout and stderr will be redirected.
            extra_env (dict): Additional environment variables to set for the subprocess.

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

                # Merge extra environment variables with current environment
                env = os.environ.copy()
                if extra_env:
                    env.update(extra_env)

                process = subprocess.Popen(
                    args,
                    cwd=self.case_directory,
                    shell=False,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                )
                process.wait()
                return process.returncode == 0
        except Exception as e:
            with open(log_path, "a") as log:
                log.write(f"\n--- Python Exception ---\n{e}\n")
            return False

    def run_solver(self, n_processors: int = 1) -> bool:
        """
        Runs SU2_CFD solver (serial or parallel).

        Args:
            n_processors (int): Number of processors to use. If > 1, runs in parallel.

        Returns:
            bool: True if the command was successful, False otherwise.
        """
        ok, msg = self.validate_case()
        if not ok:
            self._write_log_error("log.SU2_CFD", msg)
            return False

        config = self.config_file
        # Environment variables to help with MPI/UCX issues
        mpi_env = {"UCX_TLS": "self,sm", "OMPI_MCA_btl": "self,sm"}

        if n_processors > 1:
            if not self._is_executable_available("mpirun"):
                self._write_log_error("log.SU2_CFD", "mpirun not found on PATH (required for parallel runs)")
                return False
            command = f"mpirun --oversubscribe -np {int(n_processors)} SU2_CFD {config}"
            return self.run_command(command, "log.SU2_CFD", extra_env=mpi_env)
        else:
            command = f"SU2_CFD {config}"
            return self.run_command(command, "log.SU2_CFD")

    def run_deformation(self, n_processors: int = 1) -> bool:
        """
        Runs SU2_DEF for mesh deformation.

        Args:
            n_processors (int): Number of processors to use.

        Returns:
            bool: True if the command was successful, False otherwise.
        """
        ok, msg = self.validate_case()
        if not ok:
            self._write_log_error("log.SU2_DEF", msg)
            return False

        config = self.config_file
        mpi_env = {"UCX_TLS": "self,sm", "OMPI_MCA_btl": "self,sm"}

        if n_processors > 1:
            if not self._is_executable_available("mpirun"):
                self._write_log_error("log.SU2_DEF", "mpirun not found on PATH (required for parallel runs)")
                return False
            command = f"mpirun --oversubscribe -np {int(n_processors)} SU2_DEF {config}"
            return self.run_command(command, "log.SU2_DEF", extra_env=mpi_env)
        else:
            command = f"SU2_DEF {config}"
            return self.run_command(command, "log.SU2_DEF")

    def run_solution_export(self, n_processors: int = 1) -> bool:
        """
        Runs SU2_SOL for solution file conversion/export.

        Args:
            n_processors (int): Number of processors to use.

        Returns:
            bool: True if the command was successful, False otherwise.
        """
        ok, msg = self.validate_case()
        if not ok:
            self._write_log_error("log.SU2_SOL", msg)
            return False

        config = self.config_file
        mpi_env = {"UCX_TLS": "self,sm", "OMPI_MCA_btl": "self,sm"}

        if n_processors > 1:
            if not self._is_executable_available("mpirun"):
                self._write_log_error("log.SU2_SOL", "mpirun not found on PATH (required for parallel runs)")
                return False
            command = f"mpirun --oversubscribe -np {int(n_processors)} SU2_SOL {config}"
            return self.run_command(command, "log.SU2_SOL", extra_env=mpi_env)
        else:
            command = f"SU2_SOL {config}"
            return self.run_command(command, "log.SU2_SOL")

    def get_available_restart_files(self) -> List[str]:
        """Get list of available restart files in case directory."""
        restart_files = []
        for f in os.listdir(self.case_directory):
            if f.endswith(".dat") or f.endswith(".csv") or "restart" in f.lower():
                restart_files.append(f)
        return sorted(restart_files)

    def get_convergence_history(self) -> Optional[str]:
        """Get path to convergence history file if it exists."""
        # SU2 typically writes history.csv or history.dat
        for name in ["history.csv", "history.dat", "convergence.csv"]:
            path = os.path.join(self.case_directory, name)
            if os.path.exists(path):
                return path
        return None

    def get_log_content(self, log_file: str = "log.SU2_CFD") -> Optional[str]:
        """Read content of a log file."""
        log_path = os.path.join(self.case_directory, log_file)
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                pass
        return None
