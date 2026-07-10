"""Ergonomic figure publishing from notebooks to Cloudflare R2 (Obsidian-ready).

Wraps the sibling :mod:`zetteldev.publish_figure` machinery (R2 upload + provenance
metadata + a hash-based registry so unchanged figures are never re-uploaded) behind a
single function you call from a marimo / Jupyter cell::

    from zetteldev import figpub

    fig, ax = plt.subplots(); ax.plot(...)
    figpub.publish(fig, "my_result.png")     # -> renders inline + returns the URL

The returned :class:`PubResult` renders the figure inline (from the bytes you just
published) together with the stable R2 URL and a copy-paste Obsidian image line, so a
notebook cell ending in ``figpub.publish(...)`` shows you exactly what went up.

It accepts a matplotlib ``Figure`` (or ``Axes``), a great_tables ``GT`` table (rendered via
its chromedriver wrapper and autocropped), a path to an existing image, a ``PIL.Image``, or
raw ``bytes``. The experiment name, ``figures/`` directory, and source script are inferred
from the working directory; override per call or once via :func:`configure`.

Set credentials in the repo ``.env`` (``CLOUDFLARE_ACCOUNT_ID``,
``CLOUDFLARE_R2_ACCESS_KEY_ID``, ``CLOUDFLARE_R2_SECRET_ACCESS_KEY``) — loaded
automatically. The public URL pattern is ``https://blots.kincaid.ink/{repo}/{exp}/{file}``.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["publish", "configure", "PubResult"]

# Module-level defaults; override per call or via configure().
_DEFAULTS: dict[str, Any] = {
    "dpi": 200,
    "figure_dir": None,     # default: <cwd>/figures
    "source_script": None,  # default: inferred from the figure stem
    # great_tables render width (px) for the headless-chrome window. GT.save defaults to a 6000px
    # window, which lets tables sprawl absurdly wide; most tables are small, so 800px is a saner
    # default (long source notes wrap). Bump via configure(gt_window_width=...) for a wide table.
    "gt_window_width": 800,
}

_CONTENT_TYPES = {
    "png": "image/png", "svg": "image/svg+xml", "pdf": "application/pdf",
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
}
_RASTER = {"png", "jpg", "jpeg", "webp"}


def configure(**defaults: Any) -> None:
    """Set notebook-wide defaults, e.g. ``configure(source_script="03-analysis.py", dpi=220)``."""
    unknown = set(defaults) - set(_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown config keys {unknown}; valid: {sorted(_DEFAULTS)}")
    _DEFAULTS.update(defaults)


def _pub():
    """Lazy handle to the sibling R2 machinery (defers boto3 import to first publish)."""
    from . import publish_figure
    return publish_figure


def _render_bytes(figure: Any, suffix: str, dpi: int, gt_window_width: int = 800) -> bytes:
    """Render any supported figure input to image bytes of the given format."""
    if isinstance(figure, (bytes, bytearray)):
        return bytes(figure)
    if isinstance(figure, (str, Path)):
        return Path(figure).read_bytes()
    # great_tables GT -> render through its selenium/chromedriver wrapper (GT.save), then trim the
    # wide chromedriver whitespace margin with the shared autocrop, and return the cropped bytes.
    if (type(figure).__module__ or "").split(".", 1)[0] == "great_tables" and hasattr(figure, "save"):
        if suffix not in _RASTER:
            raise ValueError(
                f"a great_tables table renders to a raster image; pass a raster extension "
                f"(one of {sorted(_RASTER)}), not '.{suffix}'"
            )
        import tempfile

        from reason_reckon.autocrop import autocrop

        with tempfile.TemporaryDirectory() as _td:
            tmp = Path(_td) / f"table.{suffix}"
            # Constrain the chrome window WIDTH (GT.save defaults to 6000px, which sprawls); keep the
            # height generous so tall tables aren't clipped. autocrop then trims to the actual content.
            figure.save(str(tmp), window_size=(int(gt_window_width), 6000))
            autocrop(tmp, padding=20)  # in-place crop of the wide whitespace margin
            return tmp.read_bytes()
    # matplotlib Axes -> its Figure
    if hasattr(figure, "get_figure") and not hasattr(figure, "savefig"):
        figure = figure.get_figure()
    if hasattr(figure, "savefig"):  # matplotlib Figure
        buf = io.BytesIO()
        kwargs: dict[str, Any] = {"format": suffix, "bbox_inches": "tight"}
        if suffix in _RASTER:
            kwargs["dpi"] = dpi
        if suffix == "png":
            kwargs["metadata"] = {"Software": ""}  # deterministic bytes -> stable hash
        figure.savefig(buf, **kwargs)
        return buf.getvalue()
    if hasattr(figure, "save") and hasattr(figure, "mode"):  # PIL.Image
        buf = io.BytesIO()
        figure.save(buf, format="JPEG" if suffix in {"jpg", "jpeg"} else suffix.upper())
        return buf.getvalue()
    raise TypeError(
        f"unsupported figure type {type(figure)!r}; pass a matplotlib Figure/Axes, "
        "a PIL.Image, a path, or raw bytes"
    )


@dataclass
class PubResult:
    """Result of :func:`publish`; renders inline in marimo/Jupyter. ``str()`` is the URL."""

    path: Path
    url: str
    markdown: str
    changed: bool
    uploaded: bool
    reason: str
    _data: bytes = b""
    _suffix: str = "png"

    def __str__(self) -> str:
        return self.url

    def _badge(self) -> str:
        return {
            "changed": "🆕 uploaded", "registry-missing": "↑ uploaded",
            "unchanged": "✓ unchanged (cached)", "dry-run": "○ dry-run (not uploaded)",
        }.get(self.reason, self.reason)

    def _inline_img(self) -> str:
        """A self-contained <img>/<svg> from the just-rendered bytes (always shows)."""
        if not self._data:
            return f'<img src="{self.url}" style="max-width:100%">'
        if self._suffix == "svg":
            try:
                return self._data.decode("utf-8")
            except UnicodeDecodeError:
                pass
        ct = _CONTENT_TYPES.get(self._suffix, "image/png")
        b64 = base64.b64encode(self._data).decode("ascii")
        return f'<img src="data:{ct};base64,{b64}" style="max-width:100%">'

    def _repr_html_(self) -> str:
        return (
            f'<div>{self._inline_img()}'
            f'<div style="margin-top:6px;font-family:monospace;font-size:12px;color:#555">'
            f'{self._badge()} &middot; '
            f'<a href="{self.url}">{self.url}</a></div>'
            f'<div style="margin-top:2px;font-family:monospace;font-size:12px;'
            f'background:#f5f5f5;padding:4px 6px;border-radius:4px">{_html_escape(self.markdown)}</div>'
            f"</div>"
        )

    def _repr_markdown_(self) -> str:
        return f"{self.markdown}\n\n`{self.markdown}`  · {self._badge()}"


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def publish(
    figure: Any,
    name: str | Path,
    *,
    dpi: int | None = None,
    source_script: str | None = None,
    figure_dir: str | Path | None = None,
    dry_run: bool = False,
    alt: str | None = None,
) -> PubResult:
    """Publish a figure to R2 and return a :class:`PubResult` (renders inline).

    Args:
        figure: matplotlib ``Figure``/``Axes``, a great_tables ``GT`` table (raster
            extension only — rendered via GT's chromedriver wrapper, then autocropped),
            ``PIL.Image``, an existing image path, or raw image ``bytes``.
        name: output filename (``"result.png"``). Without an extension, ``.png`` is added.
            Relative names land in ``figure_dir``; absolute paths are used as-is.
        dpi: raster DPI (default 200, or the configured value).
        source_script: provenance tag stored on the R2 object (default: inferred).
        figure_dir: where to write the local file (default: ``<cwd>/figures``).
        dry_run: render + save locally and compute the URL, but do not upload.
        alt: alt text for the Obsidian markdown (default: the filename).

    Returns:
        :class:`PubResult` with ``.url``, ``.markdown``, ``.path``, ``.changed``,
        ``.uploaded``. Re-publishing identical bytes is a no-op (``reason="unchanged"``).
    """
    pub = _pub()
    git_root = pub.get_git_root()
    pub.load_dotenv(git_root / ".env")

    dpi = int(dpi if dpi is not None else _DEFAULTS["dpi"])
    figure_dir = figure_dir if figure_dir is not None else _DEFAULTS["figure_dir"]
    source_script = source_script if source_script is not None else _DEFAULTS["source_script"]

    # resolve the output path
    path = Path(name)
    if not path.suffix:
        path = path.with_suffix(".png")
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in _CONTENT_TYPES:
        raise ValueError(f"unsupported extension '.{suffix}'; use one of {sorted(_CONTENT_TYPES)}")
    if not path.is_absolute():
        base = Path(figure_dir) if figure_dir is not None else Path.cwd() / "figures"
        if path.parent == Path("."):
            path = base / path
        path = (Path.cwd() / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    rendered = _render_bytes(figure, suffix, dpi, int(_DEFAULTS["gt_window_width"]))
    rendered_hash = hashlib.sha256(rendered).hexdigest()[:16]
    existing_hash = (
        hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None
    )
    changed = rendered_hash != existing_hash

    rel = str(path.relative_to(git_root))
    registry = pub.load_registry()
    entry = registry.get(rel, {})
    registry_current = entry.get("hash") == rendered_hash and bool(entry.get("url"))
    url = entry.get("url") or _public_url(path, pub)

    alt = alt or path.name
    markdown = f"![{alt}]({url})"

    def _result(reason: str, uploaded: bool) -> PubResult:
        return PubResult(path, url, markdown, changed, uploaded, reason, rendered, suffix)

    # fast path: identical bytes already live
    if not changed and registry_current:
        return _result("unchanged", uploaded=False)

    if changed and not dry_run:
        path.write_bytes(rendered)
    if dry_run:
        if changed:
            path.write_bytes(rendered)  # local preview is still useful in dry-run
        return _result("dry-run", uploaded=False)

    client = pub.get_r2_client()
    pub.ensure_bucket_exists(client)
    url = pub.publish_figure(path, client, source_script=source_script, dry_run=False)
    registry[rel] = {"hash": pub.file_hash(path), "url": url, "commit": pub.get_git_commit()}
    pub.save_registry(registry)
    markdown = f"![{alt}]({url})"
    return PubResult(path, url, markdown, changed, True,
                     "changed" if changed else "registry-missing", rendered, suffix)


def _public_url(path: Path, pub) -> str:
    git_root = pub.get_git_root()
    repo_name = pub.get_repo_name()
    experiment = pub.infer_experiment(path, git_root)
    return f"{pub.PUBLIC_BASE}/{pub.make_key(path, repo_name, experiment)}"
