"""Import functions/classes from marimo notebooks whose filenames aren't legal identifiers.

A marimo notebook is a valid Python module: symbols in the ``with app.setup:`` block and
cells defined with ``@app.function`` become module-level objects, while ``app.run()`` is
guarded by ``if __name__ == "__main__"`` — so importing the file is side-effect-light
(the per-cell ``@app.cell`` bodies are registered, not executed). But a filename like
``02-belief-extraction.py`` is not a legal identifier and can't be ``import``-ed by name.
This loads it by path::

    from zetteldev import notebook_module

    bex = notebook_module("02-belief-extraction.py")
    beliefs = bex.extract_beliefs(...)   # an @app.function cell, callable downstream

For a function/class to import cleanly it must live in its own cell and reference only
top-of-DAG symbols (the setup cell), not other cells.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

__all__ = ["notebook_module"]

_CACHE: dict[str, ModuleType] = {}


def _module_name(path: Path) -> str:
    """A legal module name from any filename (handles leading digits, dashes, dots)."""
    return "zetteldev_nb_" + re.sub(r"\W", "_", path.stem)


def notebook_module(path: str | Path, *, reload: bool = False) -> ModuleType:
    """Import a marimo notebook / numbered script by path and return its module.

    Args:
        path: path to the ``.py`` notebook (relative to the working directory, or absolute).
        reload: re-execute even if it was already imported this session (defaults to a
            per-path cache, so a notebook's heavy setup cell runs at most once).

    Returns:
        The imported module. Access its ``@app.function`` cells and setup-level symbols
        as attributes.

    Raises:
        FileNotFoundError: the path does not exist.
        ImportError: the file could not be turned into an import spec.
    """
    p = Path(path).resolve()
    key = str(p)
    if not reload and key in _CACHE:
        return _CACHE[key]
    if not p.exists():
        raise FileNotFoundError(f"notebook not found: {p}")

    name = _module_name(p)
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register first so dataclasses / pickling inside resolve
    try:
        spec.loader.exec_module(mod)  # runs setup + registers cells; app.run() is __main__-guarded
    except Exception:
        sys.modules.pop(name, None)
        raise
    _CACHE[key] = mod
    return mod
