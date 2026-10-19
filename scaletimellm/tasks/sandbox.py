"""Best-effort restricted subprocess sandbox for generated Python code.

This is intentionally small: it runs candidate code in a separate interpreter
with ``-I`` (isolated mode), a minimal environment, a CPU/file-size limit and a
wall-clock timeout.  Candidate imports are screened by AST and only a whitelist
of pure-Python standard-library modules is allowed.  It is suitable for grading
research samples, not for running hostile code on a shared machine.
"""

from __future__ import annotations

import ast
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_IMPORTS = {
    "bisect",
    "collections",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "statistics",
    "string",
    "typing",
}
DENIED_IMPORTS = {
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "urllib",
}


class UnsafeCodeError(ValueError):
    """Raised when candidate code imports a disallowed module."""


def check_code_safety(code: str) -> None:
    """Reject syntax errors and imports outside :data:`ALLOWED_IMPORTS`."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise UnsafeCodeError(f"syntax error: {error}") from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS or root in DENIED_IMPORTS:
                    raise UnsafeCodeError(f"import not allowed: {root}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS or root in DENIED_IMPORTS:
                raise UnsafeCodeError(f"import not allowed: {root}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError("dunder attribute access is not allowed")


def _limits() -> None:  # pragma: no cover - executed in the child process
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    except (ValueError, OSError):
        pass


def run_python_tests(
    candidate_code: str,
    test_code: str,
    entry_point: str | None = None,
    timeout: float = 2.0,
) -> tuple[bool, str]:
    """Run candidate code plus trusted unit tests in a subprocess.

    Returns ``(passed, captured_output)``.  Failures, syntax errors, disallowed
    imports and timeouts return ``False`` rather than raising.
    """
    try:
        check_code_safety(candidate_code)
    except UnsafeCodeError as error:
        return False, f"unsafe code: {error}"

    if entry_point:
        test_code = f"{test_code.rstrip()}\n\ncheck({entry_point})\n"
    source = f"{candidate_code.rstrip()}\n\n{test_code}\n"
    with tempfile.TemporaryDirectory(prefix="scaletimellm_sandbox_") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": directory,
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=directory,
                env=environment,
                preexec_fn=_limits if os.name == "posix" else None,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout:.2f}s"
        output = (completed.stdout + completed.stderr)[-4000:]
        return completed.returncode == 0, output
