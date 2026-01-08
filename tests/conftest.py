import os
import sys
from pathlib import Path

import pytest


def pytest_configure():
    # Ensure we can import `core.*` from the repo's `src/` directory.
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(src_path))

    # Avoid Qt trying to connect to a display if any GUI code is imported.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication instance for Qt widget tests."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
