"""Zetteldev shared infrastructure, importable as ``zetteldev``.

The package is rooted at the repo's ``.zetteldev/`` directory (see pyproject.toml);
the leading-dot directory name is decoupled from this legal import name. Submodules
are imported lazily — ``import zetteldev`` stays light and pulls no heavy deps until
you reach for a specific tool, e.g.::

    from zetteldev import figpub          # ergonomic figure publishing (lazy boto3)
    from zetteldev import publish_figure  # the underlying R2 machinery
    from zetteldev import notebook_module # import a function from a numbered marimo notebook

Heavy submodules (figpub, publish_figure) stay as lazy attribute access and pull no
boto3 / matplotlib until used; only stdlib-only helpers are re-exported eagerly here.
"""

from .nbimport import notebook_module

__version__ = "0.1.0"
__all__ = ["notebook_module"]
