from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _new_destination(path):
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise ValueError('Output already exists; choose a new path')
    if destination.is_relative_to(PACKAGE_ROOT):
        runs = PACKAGE_ROOT / 'runs'
        if not destination.is_relative_to(runs) or destination == runs:
            raise ValueError('Inside this package, outputs must be new paths under runs/')
    if PACKAGE_ROOT.is_relative_to(destination):
        raise ValueError('Output may not contain the package')
    return destination


def new_output(path):
    """Create a new output directory without overwriting saved package results."""
    destination = _new_destination(path)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def new_report(path):
    """Prepare a new report file beside existing run outputs, never saved results."""
    destination = _new_destination(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def package_relative(path):
    """Use portable repository-relative provenance when paths share a drive."""
    destination = Path(path).expanduser().resolve()
    try:
        return Path(os.path.relpath(destination, PACKAGE_ROOT)).as_posix()
    except ValueError:
        # Windows cannot express a relative path across drive letters. Preserve
        # an explicitly supplied external location rather than inventing one.
        return destination.as_posix()
