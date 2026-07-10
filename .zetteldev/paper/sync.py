"""Obsidian ↔ Overleaf paper bridge.

Canonical source is a single md file in ~/Pumberton/Workshop/. Outbound:
md → tex (Quarto + Lua filter) → Overleaf git. Inbound: pull Overleaf, diff,
draft md patches via LLM; land them in review/ for the human to accept.

Commands:
  new <slug>                       scaffold papers/<slug>/
  push <slug>                      md → tex, fetch figures, commit + push
  pull <slug>                      fetch Overleaf, draft review patches
  accept <slug> <patch-id>         apply a drafted patch to source md
  figures <slug>                   refresh only overleaf/figures/ from R2
  diff <slug>                      preview outbound without committing
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
import yaml
import questionary
import requests
from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Paths and helpers
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return Path(out.stdout.strip())


def paper_dir(slug: str) -> Path:
    return repo_root() / "papers" / slug


def list_papers() -> list[str]:
    """Return names of all scaffolded papers (papers/*/paper.yaml)."""
    root = repo_root() / "papers"
    if not root.exists():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "paper.yaml").exists()
    )


def resolve_slug(slug: Optional[str]) -> str:
    """If slug is given, return it; otherwise auto-detect from papers/.

    Fails with a clear message if zero or more than one paper exists.
    """
    if slug:
        return slug
    papers = list_papers()
    if len(papers) == 1:
        return papers[0]
    if not papers:
        typer.secho(
            "No papers scaffolded yet. Run `just new-paper <slug>` first.",
            fg="red",
        )
        raise typer.Exit(1)
    typer.secho("Multiple papers in this repo — pass <slug> explicitly:", fg="red")
    for p in papers:
        typer.echo(f"  {p}")
    raise typer.Exit(1)


def filter_path() -> Path:
    return repo_root() / ".zetteldev" / "paper" / "filters" / "obsidian-callouts.lua"


def templates_dir() -> Path:
    return repo_root() / ".zetteldev" / "paper" / "templates"


def expand(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_short(s: str | bytes, n: int = 16) -> str:
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:n]


def run(cmd: list[str], cwd: Optional[Path] = None, check: bool = True,
        capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, check=check,
        capture_output=capture, text=True,
    )


# ---------------------------------------------------------------------------
# paper.yaml
# ---------------------------------------------------------------------------

@dataclass
class PaperConfig:
    slug: str
    overleaf_git_url: str
    source_md: Path          # expanded absolute path
    refs_bib: Path           # expanded absolute path (may not exist)
    template: str = "article"
    main_tex: str = "main.tex"
    branch: str = "master"   # Overleaf's default branch

    @classmethod
    def load(cls, slug: str) -> "PaperConfig":
        cfg_path = paper_dir(slug) / "paper.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No paper.yaml at {cfg_path}. Run `just new-paper {slug}`.")
        raw = yaml.safe_load(cfg_path.read_text())
        return cls(
            slug=raw["slug"],
            overleaf_git_url=raw["overleaf_git_url"],
            source_md=expand(raw["source_md"]),
            refs_bib=expand(raw["refs_bib"]),
            template=raw.get("template", "article"),
            main_tex=raw.get("main_tex", "main.tex"),
            branch=raw.get("branch", "master"),
        )

    def save(self):
        cfg_path = paper_dir(self.slug) / "paper.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "slug": self.slug,
            "overleaf_git_url": self.overleaf_git_url,
            "source_md": str(self.source_md),
            "refs_bib": str(self.refs_bib),
            "template": self.template,
            "main_tex": self.main_tex,
            "branch": self.branch,
        }, sort_keys=False))


# ---------------------------------------------------------------------------
# Ledger — per-paper sync state
# ---------------------------------------------------------------------------

@dataclass
class Ledger:
    last_outbound_md_sha: Optional[str] = None
    last_outbound_tex_sha: Optional[str] = None
    last_outbound_tex_snapshot: Optional[str] = None  # path relative to paper_dir
    last_outbound_overleaf_commit: Optional[str] = None
    last_inbound_overleaf_commit: Optional[str] = None
    figures: dict[str, dict] = field(default_factory=dict)
    # url -> {hash, local_name, pulled_at}

    @classmethod
    def load(cls, slug: str) -> "Ledger":
        p = paper_dir(slug) / ".ledger.json"
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        return cls(**raw)

    def save(self, slug: str):
        p = paper_dir(slug) / ".ledger.json"
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Source map (fingerprint-based alignment between md and tex blocks)
# ---------------------------------------------------------------------------

def normalize_for_fp(text: str) -> str:
    """Collapse whitespace, strip tex/md markup so md and tex blocks compare."""
    t = text.lower()
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)  # \cmd, \cmd{x}
    t = re.sub(r"[\\{}%$#&_^~]", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fingerprint(text: str) -> str:
    return sha256_short(normalize_for_fp(text), 12)


@dataclass
class Block:
    start_line: int  # 1-based inclusive
    end_line: int    # 1-based inclusive
    text: str
    fp: str
    kind: str        # "paragraph", "heading", "code", "image", "callout"


def split_md_blocks(md: str) -> list[Block]:
    """Split md into logical blocks by blank lines. Skips YAML frontmatter."""
    lines = md.splitlines()
    i = 0
    # Strip frontmatter
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                i = j + 1
                break

    blocks: list[Block] = []
    n = len(lines)
    while i < n:
        # skip blank lines
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        start = i
        # consume until blank
        while i < n and lines[i].strip():
            i += 1
        end = i - 1
        text = "\n".join(lines[start:end + 1])
        kind = _classify_md_block(text)
        blocks.append(Block(
            start_line=start + 1, end_line=end + 1,
            text=text, fp=fingerprint(text), kind=kind,
        ))
    return blocks


def _classify_md_block(text: str) -> str:
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if first.startswith("#"):
        return "heading"
    if first.startswith("```"):
        return "code"
    if re.match(r"!\[[^\]]*\]\(", first) or first.startswith("!["):
        return "image"
    if first.startswith(">"):
        return "callout"
    return "paragraph"


def split_tex_blocks(tex: str) -> list[Block]:
    """Split tex body into logical blocks by blank lines. Strips comments."""
    lines = tex.splitlines()
    # Strip whole-line comments (but not sentinels we own)
    blocks: list[Block] = []
    i, n = 0, len(lines)
    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and lines[i].strip():
            i += 1
        end = i - 1
        text = "\n".join(lines[start:end + 1])
        # Skip pure-comment blocks (including sentinels and filter eliders)
        if all(ln.strip().startswith("%") for ln in text.splitlines()):
            continue
        # Classify
        first = text.lstrip().splitlines()[0]
        if first.startswith("\\section") or first.startswith("\\subsection") \
                or first.startswith("\\subsubsection"):
            kind = "heading"
        elif first.startswith("\\begin{verbatim}") or first.startswith("\\begin{lstlisting}"):
            kind = "code"
        elif "\\includegraphics" in text:
            kind = "image"
        elif first.startswith("\\begin{quote}"):
            kind = "callout"
        else:
            kind = "paragraph"
        blocks.append(Block(
            start_line=start + 1, end_line=end + 1,
            text=text, fp=fingerprint(text), kind=kind,
        ))
    return blocks


_MD_HEADING_RE = re.compile(r"^\s*#+\s*(.+?)\s*$", re.MULTILINE)
_TEX_HEADING_RE = re.compile(r"\\(?:sub)*section\*?\{([^}]+)\}")


def _md_heading_title(text: str) -> Optional[str]:
    m = _MD_HEADING_RE.search(text)
    return m.group(1).strip().lower() if m else None


def _tex_heading_title(text: str) -> Optional[str]:
    m = _TEX_HEADING_RE.search(text)
    return m.group(1).strip().lower() if m else None


def _find_anchor_pairs(md_blocks: list[Block],
                       tex_blocks: list[Block]) -> list[tuple[int, int]]:
    """Return (md_idx, tex_idx) pairs for heading blocks whose titles match,
    strictly increasing in both indices. Prevents cross-section drift."""
    md_hdrs = [
        (i, _md_heading_title(b.text))
        for i, b in enumerate(md_blocks) if b.kind == "heading"
    ]
    tex_hdrs = [
        (j, _tex_heading_title(b.text))
        for j, b in enumerate(tex_blocks) if b.kind == "heading"
    ]
    pairs: list[tuple[int, int]] = []
    tex_cursor = 0
    for md_idx, md_title in md_hdrs:
        if md_title is None:
            continue
        for tex_pos in range(tex_cursor, len(tex_hdrs)):
            tex_idx, tex_title = tex_hdrs[tex_pos]
            if tex_title is None:
                continue
            if md_title == tex_title or \
                    fuzz.ratio(md_title, tex_title) >= 85:
                pairs.append((md_idx, tex_idx))
                tex_cursor = tex_pos + 1
                break
    return pairs


def _align_segment(md_seg: list[Block], tex_seg: list[Block]) -> list[dict]:
    """Greedy in-order alignment within a segment (bounded by anchor pairs)."""
    mappings: list[dict] = []
    i = j = 0
    M, T = len(md_seg), len(tex_seg)
    while i < M and j < T:
        m, t = md_seg[i], tex_seg[j]
        if m.fp == t.fp:
            mappings.append(_mapping(m, t, 1.0)); i += 1; j += 1; continue
        score = fuzz.token_set_ratio(
            normalize_for_fp(m.text), normalize_for_fp(t.text)) / 100.0
        if score >= 0.55 and m.kind == t.kind:
            mappings.append(_mapping(m, t, score)); i += 1; j += 1; continue
        look_m = fuzz.token_set_ratio(
            normalize_for_fp(md_seg[i + 1].text), normalize_for_fp(t.text)) / 100.0 \
            if i + 1 < M else 0.0
        look_t = fuzz.token_set_ratio(
            normalize_for_fp(m.text), normalize_for_fp(tex_seg[j + 1].text)) / 100.0 \
            if j + 1 < T else 0.0
        if look_m > look_t and look_m >= 0.55:
            mappings.append(_mapping(m, None, 0.0)); i += 1
        elif look_t > look_m and look_t >= 0.55:
            mappings.append(_mapping(None, t, 0.0)); j += 1
        else:
            mappings.append(_mapping(m, t, max(score, 0.0))); i += 1; j += 1
    while i < M:
        mappings.append(_mapping(md_seg[i], None, 0.0)); i += 1
    while j < T:
        mappings.append(_mapping(None, tex_seg[j], 0.0)); j += 1
    return mappings


def align_blocks(md_blocks: list[Block], tex_blocks: list[Block]) -> list[dict]:
    """Section-anchored alignment between md and tex blocks.

    Section headings (md `# foo`, tex `\\section{foo}`) serve as hard anchors:
    alignment between two anchor pairs is done greedily, but alignment cannot
    cross an anchor. This prevents positional drift from propagating past a
    section boundary — the failure mode that landed a Background-section edit
    onto an Introduction-section hunk in the first real pull-paper run.
    """
    anchors = _find_anchor_pairs(md_blocks, tex_blocks)
    mappings: list[dict] = []
    prev_m, prev_t = -1, -1
    for md_idx, tex_idx in anchors:
        mappings.extend(_align_segment(
            md_blocks[prev_m + 1:md_idx], tex_blocks[prev_t + 1:tex_idx]))
        mappings.append(_mapping(md_blocks[md_idx], tex_blocks[tex_idx], 1.0))
        prev_m, prev_t = md_idx, tex_idx
    mappings.extend(_align_segment(
        md_blocks[prev_m + 1:], tex_blocks[prev_t + 1:]))
    return mappings


def _mapping(md: Optional[Block], tex: Optional[Block], conf: float) -> dict:
    return {
        "md_start": md.start_line if md else None,
        "md_end": md.end_line if md else None,
        "md_fp": md.fp if md else None,
        "md_kind": md.kind if md else None,
        "tex_start": tex.start_line if tex else None,
        "tex_end": tex.end_line if tex else None,
        "tex_fp": tex.fp if tex else None,
        "tex_kind": tex.kind if tex else None,
        "confidence": round(conf, 3),
    }


# ---------------------------------------------------------------------------
# Md preprocessing: Obsidian → Pandoc-friendly
# ---------------------------------------------------------------------------

R2_HOST = "https://blots.kincaid.ink"
R2_IMG_RE = re.compile(
    r"!\[([^\]]*)\]\(" + re.escape(R2_HOST) + r"/([^\s\)]+)\)"
)
CITE_WIKILINK_RE = re.compile(r"\[\[@([^\]|]+)\]\]")
ALIAS_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
PLAIN_WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\]@][^\]]*?)\]\]")
EMBED_IMG_RE = re.compile(r"!\[\[([^\]]+\.(?:png|jpg|jpeg|svg|pdf|webp))\]\]", re.IGNORECASE)
EMBED_NOTE_RE = re.compile(r"!\[\[([^\]]+)\]\]")

# Plain markdown image with a local (non-URL) target. Examples:
#   ![alt](some-file.png)
#   ![alt](path/to/some-file.png){width=...}
#   ![alt](file with spaces (1).pdf)
# Excludes http(s):// targets (those are handled by R2_IMG_RE) and Obsidian
# wikilink embeds (handled by EMBED_IMG_RE before this pattern runs).
# The URL portion allows spaces and one level of nested parens, matching
# pandoc's image-link parser.
_LOCAL_IMG_REF_RE = re.compile(
    r"(?<!!)!\[[^\]]*\]\("
    r"(?!https?://)"
    r"((?:[^()\n]|\([^()\n]*\))+\.(?:png|jpg|jpeg|svg|pdf|webp|gif))"
    r"\)",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"(?<![\w@\[])\[?@([A-Za-z][\w:-]*(?:\.[\w:-]+)*)\]?")

# Sidenote: bracketed text that isn't a citation, footnote reference, image,
# inline link, link-reference definition, nested link, or Obsidian callout
# marker. Body must not start with `@` (citation), `^` (footnote), or `!`
# (Obsidian callout type). Not followed by `(`, `[`, or `:`.
SIDENOTE_RE = re.compile(
    r"(?<![!\[\\])\[(?!\s*[@^!])([^\]\[\n]+?)\](?![\(\[:])"
)
SIDENOTE_CMD_DEFAULT = "snKincaid"  # bare `[text]` → \snKincaid{text}

# Multi-author sidenotes: `[Name: text]` → \sn<Name>{text} when Name appears
# in this set. Add new authors here and register a corresponding `\newcommand`
# in the template preamble.
SIDENOTE_AUTHORS = {"Kincaid", "Max", "Zeyu", "Boyi", "Peter", "Dilip"}

# Override the default `sn<Name>` macro on a per-author basis. Use when a
# co-author has renamed their macro on Overleaf (e.g. Peter renamed
# `\snPeter` → `\peter`). Inbound patches still recognize both forms; this
# only controls what outbound emits.
SIDENOTE_AUTHOR_CMDS = {
    "Peter": "peter",
    "Dilip": "dnote",
}

# Shielding patterns — applied in order before the sidenote regex runs, so
# brackets inside code / math can't be mistaken for sidenote syntax.
_FENCED_CODE_RE = re.compile(r"(^|\n)(```.*?\n.*?\n```)(?=\n|$)", re.DOTALL)
_CODE_PLACEHOLDER_RE = re.compile(r"(`+)([^\n`](?:[^\n]*?[^\n`])?)\1")
_DISPLAY_MATH_DOLLAR_RE = re.compile(r"\$\$[\s\S]+?\$\$")
_DISPLAY_MATH_BRACKET_RE = re.compile(r"\\\[[\s\S]+?\\\]")
_INLINE_MATH_DOLLAR_RE = re.compile(r"(?<!\$)\$(?!\s)(?:\\\$|[^\n$])+?\$(?!\$)")
_INLINE_MATH_PAREN_RE = re.compile(r"\\\([\s\S]+?\\\)")

# Display-math environments that are themselves complete math mode (must NOT
# be wrapped in `$$...$$` or `\[...\]`). When the user writes
# `$$\begin{align*}...\end{align*}$$` (necessary for Obsidian's renderer to
# display the math), we strip the outer `$$` before pandoc sees it.
_DISPLAY_MATH_ENV_NAMES = {
    "align", "align*", "alignat", "alignat*", "equation", "equation*",
    "gather", "gather*", "multline", "multline*", "eqnarray", "eqnarray*",
    "displaymath",
}
_DOUBLE_DOLLAR_ENV_RE = re.compile(
    r"\$\$\s*(\\begin\{([A-Za-z]+\*?)\}[\s\S]*?\\end\{\2\})\s*\$\$"
)
# Catch-all shield for math envs whether or not wrapped in `$$`.
_MATH_ENV_BLOCK_RE = re.compile(
    r"\\begin\{(align\*?|alignat\*?|equation\*?|gather\*?|multline\*?|"
    r"eqnarray\*?|displaymath)\}[\s\S]*?\\end\{\1\}"
)


# Obsidian's `tikz` fenced block holds a standalone-style LaTeX document
# (preamble + `\begin{document}`); the body must be lifted out and wrapped in
# a figure float for the templated paper. `\usepackage{tikz}` is provided by
# the template preamble — any `\usetikzlibrary{...}` lines are hoisted into the
# emitted raw block (modern PGF accepts them in the document body).
_TIKZ_BLOCK_RE = re.compile(
    r"(?:^|\n)```tikz[^\n]*\n([\s\S]*?)\n```(?=\n|$)"
)
_TIKZ_USEPACKAGE_RE = re.compile(r"^[ \t]*\\usepackage(?:\[[^\]]*\])?\{[^}]*\}[ \t]*\n?", re.M)
_TIKZ_DOCENV_RE = re.compile(r"^[ \t]*\\(?:begin|end)\{document\}[ \t]*\n?", re.M)
_TIKZ_LIBRARY_RE = re.compile(r"^[ \t]*(\\usetikzlibrary\{[^}]*\})[ \t]*\n?", re.M)


def _transform_tikz_blocks(md: str) -> str:
    """Convert Obsidian ```tikz ... ``` blocks into raw-LaTeX figure floats.

    Strips embedded `\\usepackage{...}` and `\\begin{document}`/`\\end{document}`
    lines (those belong in the template preamble), hoists any
    `\\usetikzlibrary{...}` declarations to the top of the emitted block, and
    wraps the remaining picture(s) in `\\begin{figure}[H]\\centering ... \\end{figure}`
    so they float properly.
    """
    def _sub(m: re.Match) -> str:
        body = m.group(1)
        libraries: list[str] = []

        def _lib(lm: re.Match) -> str:
            libraries.append(lm.group(1))
            return ""

        body = _TIKZ_LIBRARY_RE.sub(_lib, body)
        body = _TIKZ_USEPACKAGE_RE.sub("", body)
        body = _TIKZ_DOCENV_RE.sub("", body)
        body = body.strip("\n")

        lines: list[str] = []
        lines.extend(libraries)
        lines.append("\\begin{figure}[H]")
        lines.append("\\centering")
        lines.append(body)
        lines.append("\\end{figure}")
        block = "\n".join(lines)
        return "\n\n```{=latex}\n" + block + "\n```\n"

    return _TIKZ_BLOCK_RE.sub(_sub, md)


# Default Quarto/pandoc figure attributes auto-added to bare `![alt](url)`
# images on push-paper. Tunable per-paper later via paper.yaml; for now
# hardcoded to "fits at top of page, two-thirds width" which is a sensible
# default for NeurIPS-shaped single-column papers.
DEFAULT_FIG_ATTRS = "width=70%"  # fig-pos is set globally in _inject_quarto_meta

# `![alt](url)` not followed by `{...}` (i.e., no existing attributes).
# Lookahead allows trailing whitespace before EOL but stops at `{`.
_IMAGE_NO_ATTRS_RE = re.compile(
    r"(!\[[^\]]*\]\([^)\n]+\))(?![\s]*\{)"
)


def annotate_figures(md: str, attrs: str = DEFAULT_FIG_ATTRS) -> tuple[str, int]:
    """Append `{<attrs>}` to image refs that lack attribute braces.

    Skips images that already have a `{...}` block (so user overrides survive
    re-runs). Wikilink-style embeds `![[file.png]]` are not touched — they're
    rewritten by `preprocess_md` to plain `![](file.png)` later, which would
    catch them too if applied at the right point. We only touch
    `![alt](url)` form here so the source md stays human-friendly.

    Returns (mutated_md, count_of_images_annotated).
    """
    n = 0

    def _sub(m: re.Match) -> str:
        nonlocal n
        n += 1
        return f"{m.group(1)}{{{attrs}}}"

    new = _IMAGE_NO_ATTRS_RE.sub(_sub, md)
    return new, n


def _unwrap_display_math_envs(md: str) -> str:
    """Strip outer `$$...$$` around display-math environments.

    Obsidian's preview requires `$$...$$` to render any block as math, so the
    user wraps `align*` blocks in `$$`. LaTeX rejects this — `align*` is itself
    a display-math env, and nesting it inside `\\[ ... \\]` fails amsmath's
    structure check. We unwrap only when the entire content of the `$$` block
    IS a display-math env (so legitimate `$$\\begin{aligned}...\\end{aligned}$$`
    — `aligned` is sub-env, must stay wrapped — is left alone).
    """
    def _sub(m: re.Match) -> str:
        env = m.group(2)
        return m.group(1) if env in _DISPLAY_MATH_ENV_NAMES else m.group(0)
    return _DOUBLE_DOLLAR_ENV_RE.sub(_sub, md)


def preprocess_md(md: str) -> tuple[str, list[str], list[str]]:
    """Transform Obsidian syntax into Pandoc-friendly forms.

    Returns (preprocessed_md, r2_urls, local_embed_names).
    """
    r2_urls: list[str] = []

    def _r2_sub(m: re.Match) -> str:
        tail = m.group(2)
        r2_urls.append(f"{R2_HOST}/{tail}")
        return m.group(0)  # keep as-is; local rewrite happens after figure pull

    # Convert Obsidian `tikz` fenced blocks into raw-LaTeX figure floats
    # BEFORE any other transform — once they become raw blocks they're inert.
    out = _transform_tikz_blocks(md)
    out = R2_IMG_RE.sub(_r2_sub, out)
    out = CITE_WIKILINK_RE.sub(r"[@\1]", out)
    out = ALIAS_WIKILINK_RE.sub(r"\2", out)
    out = PLAIN_WIKILINK_RE.sub(r"\1", out)
    # Capture local embed filenames so _common_push can copy them from the
    # Obsidian vault into overleaf/figures/. Sources:
    #   - Obsidian wikilink embeds `![[file.png]]`
    #   - Plain markdown images `![alt](file.png)` with a local (non-URL) target
    out = EMBED_IMG_RE.sub(r"![](\1)", out)
    # Note-transclusion embeds: drop.
    out = EMBED_NOTE_RE.sub("", out)
    seen_embeds: list[str] = []
    for m in _LOCAL_IMG_REF_RE.finditer(out):
        name = m.group(1)
        if name not in seen_embeds:
            seen_embeds.append(name)
    local_embeds = seen_embeds
    # Unwrap `$$\begin{align*}...\end{align*}$$` BEFORE the sidenote pass so
    # the result is shielded as a math env block.
    out = _unwrap_display_math_envs(out)
    out = _apply_sidenotes(out)
    out = _normalize_crossref_prefixes(out)
    return out, r2_urls, local_embeds


# Quarto recognizes only specific id prefixes for theorem-like divs. Common
# human-friendly spellings are mapped to Quarto's canonical ones so the
# source md reads naturally even if Quarto's vocabulary is terser.
_CROSSREF_PREFIX_ALIASES = {
    "prop": "prp",
    "conj": "cnj",
    "algo": "alg",
}


def _normalize_crossref_prefixes(md: str) -> str:
    """Rewrite div ids / `@ref` crossrefs from human-friendly aliases to
    Quarto's canonical prefixes (e.g. `prop-` → `prp-`).

    Preserves match-casing of the rest of the id (we only rewrite the prefix).
    """
    out = md
    for alias, canonical in _CROSSREF_PREFIX_ALIASES.items():
        # Div attribute id inside `::: {#alias-x .cls}` and inline `{#alias-x}`.
        out = re.sub(
            rf"#({alias})-",
            f"#{canonical}-",
            out,
        )
        # Pandoc-style crossref `@alias-x` → `@canonical-x`
        out = re.sub(
            rf"(?<![\w@])@{alias}-",
            f"@{canonical}-",
            out,
        )
    return out


def _apply_sidenotes(md: str) -> str:
    """Rewrite `[sidenote text]` → inline raw LaTeX `\\snKincaid{...}`.

    Shields fenced and inline code spans AND math regions (``$...$``,
    ``$$...$$``, ``\\(...\\)``, ``\\[...\\]``) so brackets inside math like
    `[0,1]` intervals don't get misread as sidenote syntax.
    """
    shields: list[str] = []

    def _shield(m: re.Match) -> str:
        shields.append(m.group(0))
        return f"\0SHIELD{len(shields) - 1}\0"

    masked = md
    # Order matters: fenced code first (contains backticks that inline code
    # would otherwise match); display math before inline math (to avoid
    # matching one `$` of a `$$`).
    for pat in (_FENCED_CODE_RE, _CODE_PLACEHOLDER_RE,
                _MATH_ENV_BLOCK_RE,
                _DISPLAY_MATH_DOLLAR_RE, _DISPLAY_MATH_BRACKET_RE,
                _INLINE_MATH_DOLLAR_RE, _INLINE_MATH_PAREN_RE):
        masked = pat.sub(_shield, masked)

    def _sn_sub(m: re.Match) -> str:
        body = m.group(1).strip()
        if not body:
            return m.group(0)
        # Multi-author syntax: `[Name: text]` → \sn<Name>{text}, unless the
        # author has overridden their macro name in SIDENOTE_AUTHOR_CMDS.
        author_m = re.match(r"^([A-Z][A-Za-z]+)\s*:\s*(.+)$", body, re.DOTALL)
        if author_m and author_m.group(1) in SIDENOTE_AUTHORS:
            name = author_m.group(1)
            cmd = SIDENOTE_AUTHOR_CMDS.get(name, f"sn{name}")
            note_body = author_m.group(2)
        else:
            cmd = f"sn{SIDENOTE_CMD_DEFAULT[2:]}"  # snKincaid
            note_body = body
        # Escape `%` (tex comment) and `#` (tex param) in the body; other
        # specials (\, _, &, $) are left to the user — they're often present
        # intentionally (e.g. `\cite{}` inside a note).
        safe = note_body.replace("%", r"\%").replace("#", r"\#")
        return f"\\{cmd}{{{safe}}}"

    subbed = SIDENOTE_RE.sub(_sn_sub, masked)

    def _unshield(m: re.Match) -> str:
        idx = int(m.group(1))
        return shields[idx]

    return re.sub(r"\0SHIELD(\d+)\0", _unshield, subbed)


_CROSSREF_PREFIXES_FOR_FILTERING = (
    "thm-", "lem-", "cor-", "prp-", "cnj-", "def-", "exm-", "exr-", "sol-",
    "rem-", "alg-", "fig-", "tbl-", "sec-", "eq-",
)


def find_citekeys(md: str) -> set[str]:
    """Extract bibtex citekeys, excluding Quarto crossref-style `@thm-X`,
    `@fig-X`, etc., which are cross-references, not citations."""
    keys = set()
    for m in CITATION_RE.finditer(md):
        k = m.group(1)
        if not k.startswith(_CROSSREF_PREFIXES_FOR_FILTERING):
            keys.add(k)
    for m in CITE_WIKILINK_RE.finditer(md):
        k = m.group(1)
        if not k.startswith(_CROSSREF_PREFIXES_FOR_FILTERING):
            keys.add(k)
    return keys


def slugify_r2_key(url: str) -> str:
    """URL path after /<repo>/ → filename. Strips query strings."""
    # Drop query string / fragment first so they don't leak into filenames.
    core = url.split("?", 1)[0].split("#", 1)[0]
    tail = core[len(R2_HOST) + 1:]
    parts = tail.split("/", 1)
    rest = parts[1] if len(parts) == 2 else parts[0]
    return rest.replace("/", "__")


def canonicalize_r2_url(url: str) -> str:
    """Strip query strings / fragments so cache-buster params don't prevent
    registry hits."""
    return url.split("?", 1)[0].split("#", 1)[0]


def sanitize_local_asset(name: str) -> str:
    """Rewrite a vault filename to be LaTeX-friendly (no spaces/`?`/etc.)."""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    safe = re.sub(r"[^\w.\-]+", "_", stem).strip("_")
    return f"{safe}.{ext}" if ext else safe


def obsidian_vault_root(md_path: Path) -> Optional[Path]:
    """Walk up from the source md until we find a .obsidian/ directory."""
    for parent in [md_path.parent, *md_path.parents]:
        if (parent / ".obsidian").is_dir():
            return parent
    return None


def find_vault_asset(name: str, vault_root: Path) -> Optional[Path]:
    """Locate an Obsidian-embedded asset anywhere in the vault."""
    # Fast-path common locations before full-vault rglob.
    fast = [
        vault_root / name,
        vault_root / "Library" / name,
        vault_root / "Library" / "Attachments" / name,
        vault_root / "Attachments" / name,
        vault_root / "assets" / name,
    ]
    for c in fast:
        if c.exists():
            return c
    matches = list(vault_root.rglob(name))
    return matches[0] if matches else None


def copy_local_embeds(names: list[str], source_md: Path, slug: str,
                      ledger: Ledger) -> dict[str, str]:
    """Copy each embedded vault asset into overleaf/figures/.

    Returns {original_name: sanitized_local_name}.
    """
    if not names:
        return {}
    vault = obsidian_vault_root(source_md)
    if vault is None:
        typer.secho(
            f"  could not locate Obsidian vault root above {source_md}; "
            "skipping local embed resolution", fg="yellow")
        return {}
    fig_dir = paper_dir(slug) / "overleaf" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = {}
    for name in sorted(set(names)):
        src = find_vault_asset(name, vault)
        if src is None:
            typer.secho(f"  ✗ vault asset not found: {name}", fg="red")
            continue
        safe = sanitize_local_asset(name)
        dst = fig_dir / safe
        src_hash = sha256_short(src.read_bytes())
        prior = ledger.figures.get(f"vault://{name}", {})
        if not dst.exists() or prior.get("hash") != src_hash:
            typer.echo(f"  ← {src.relative_to(vault)}  →  figures/{safe}")
            shutil.copy2(src, dst)
            ledger.figures[f"vault://{name}"] = {
                "hash": src_hash, "local_name": safe,
                "source_path": str(src), "pulled_at": now_iso(),
            }
        resolved[name] = safe
    return resolved


def rewrite_local_embeds(md: str, name_to_safe: dict[str, str]) -> str:
    """Rewrite `![](original name.png)` → `![](safe_name.png)` for each mapping."""
    if not name_to_safe:
        return md
    out = md
    for original, safe in name_to_safe.items():
        out = out.replace(f"]({original})", f"]({safe})")
    return out


# ---------------------------------------------------------------------------
# Figure sync from R2
# ---------------------------------------------------------------------------

def load_figure_registry() -> dict:
    p = repo_root() / ".zetteldev" / "figure_registry.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def registry_hash_for_url(registry: dict, url: str) -> Optional[str]:
    for entry in registry.values():
        if entry.get("url") == url:
            return entry.get("hash")
    return None


def pull_r2_figures(urls: list[str], slug: str, ledger: Ledger) -> dict[str, str]:
    """Download changed R2 figures into overleaf/figures/.

    Returns {canonical_url: local_relative_name}.

    Freshness logic:
      - Same-repo figures (URL appears in this repo's `.zetteldev/figure_registry.json`):
        compare the registry's content-hash against the ledger's cached hash.
      - Cross-repo / hand-uploaded figures (no registry entry): HEAD-probe R2,
        compare ETag against the ledger's cached etag. Refetches whenever R2
        bytes change.
    """
    if not urls:
        return {}
    fig_dir = paper_dir(slug) / "overleaf" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    registry = load_figure_registry()
    url_to_local: dict[str, str] = {}
    # Deduplicate on the canonical (query-stripped) URL so cache-busters like
    # `?t=1776637214702` don't cause repeat fetches or per-query filenames.
    canonical_to_full: dict[str, str] = {}
    for u in urls:
        canonical_to_full.setdefault(canonicalize_r2_url(u), u)
    for canon in sorted(canonical_to_full):
        full = canonical_to_full[canon]
        local_name = slugify_r2_key(canon)
        local_path = fig_dir / local_name
        registry_hash = registry_hash_for_url(registry, canon)
        prior = ledger.figures.get(canon, {})

        current_etag: Optional[str] = None
        if registry_hash:
            need_fetch = (
                not local_path.exists() or prior.get("hash") != registry_hash
            )
        else:
            try:
                head = requests.head(full, timeout=15, allow_redirects=True)
                head.raise_for_status()
                current_etag = head.headers.get("ETag", "").strip('"') or None
            except requests.RequestException:
                current_etag = None
            if not local_path.exists():
                need_fetch = True
            elif current_etag is not None:
                need_fetch = current_etag != prior.get("etag")
            else:
                need_fetch = prior.get("hash") is None

        if need_fetch:
            typer.echo(f"  ↓ {full}")
            r = requests.get(full, timeout=30)
            r.raise_for_status()
            local_path.write_bytes(r.content)
            entry: dict = {
                "hash": registry_hash or sha256_short(r.content),
                "local_name": local_name,
                "pulled_at": now_iso(),
            }
            etag = current_etag or r.headers.get("ETag", "").strip('"')
            if etag:
                entry["etag"] = etag
            ledger.figures[canon] = entry
        url_to_local[canon] = local_name
    return url_to_local


def rewrite_r2_in_md(md: str, url_to_local: dict[str, str]) -> str:
    """Rewrite R2 URLs in md image refs to point at local figures/."""
    def _sub(m: re.Match) -> str:
        alt, tail = m.group(1), m.group(2)
        canon = canonicalize_r2_url(f"{R2_HOST}/{tail}")
        local = url_to_local.get(canon)
        if local is None:
            return m.group(0)
        return f"![{alt}]({local})"
    return R2_IMG_RE.sub(_sub, md)


# ---------------------------------------------------------------------------
# Bibliography merge
# ---------------------------------------------------------------------------

BIB_ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)


def parse_bib_entries(text: str) -> dict[str, str]:
    """Return {citekey: full_entry_text} from a bibtex blob."""
    entries: dict[str, str] = {}
    for m in BIB_ENTRY_RE.finditer(text):
        citekey = m.group(2)
        start = m.start()
        brace_start = text.find("{", m.start())
        if brace_start < 0:
            continue
        depth = 1
        j = brace_start + 1
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        entries[citekey] = text[start:j]
    return entries


def merge_bib(md: str, source_bib: Path, dest_bib: Path) -> list[str]:
    """Merge citations from md into dest_bib using source_bib as authority.

    Preserves any entries already in dest that weren't in source (manual adds).
    Returns list of missing citekeys (cited in md but not in source bib).
    """
    cited = find_citekeys(md)
    source_entries = parse_bib_entries(source_bib.read_text()) if source_bib.exists() else {}
    dest_entries = parse_bib_entries(dest_bib.read_text()) if dest_bib.exists() else {}

    merged: dict[str, str] = {}
    # Start with manual dest entries not present in source
    for k, v in dest_entries.items():
        if k not in source_entries:
            merged[k] = v
    missing: list[str] = []
    for k in cited:
        if k in source_entries:
            merged[k] = source_entries[k]
        elif k not in dest_entries:
            missing.append(k)

    header = "% Merged by push-paper. Manual entries (absent from Zotero source) are preserved.\n\n"
    body = "\n\n".join(merged[k] for k in sorted(merged))
    dest_bib.write_text(header + body + "\n")
    return missing


# ---------------------------------------------------------------------------
# Quarto rendering
# ---------------------------------------------------------------------------

BODY_BEGIN = "% <<< BODY BEGIN — generated, do not edit >>>"
BODY_END = "% <<< BODY END — generated, do not edit >>>"
APPENDIX_BEGIN = "% <<< APPENDIX BEGIN — generated, do not edit >>>"
APPENDIX_END = "% <<< APPENDIX END — generated, do not edit >>>"

# Pandoc emits this for a `# Appendix` md heading. Everything after this line
# in the rendered body should land in the post-bibliography appendix region
# rather than the main body.
_APPENDIX_SPLIT_RE = re.compile(
    r"^\\section\{Appendix\}.*$", re.MULTILINE
)


def split_appendix(tex_body: str) -> tuple[str, str]:
    """Split the rendered body at the first `\\section{Appendix}` line.

    Returns (main_body, appendix_body). The appendix body has the
    `\\section{Appendix}` heading itself stripped, since the post-body
    region of the template is already inside `\\appendix` (which renumbers
    sections as A, B, ...). If no appendix heading is found, the second
    element is empty.
    """
    m = _APPENDIX_SPLIT_RE.search(tex_body)
    if not m:
        return tex_body, ""
    main = tex_body[:m.start()].rstrip()
    # Drop the heading line itself so the user's `# Appendix` doesn't double
    # with the template's `\appendix` reset.
    appendix = tex_body[m.end():].lstrip()
    return main, appendix


def render_tex_body(preprocessed_md: str, paper_slug: str) -> str:
    """Render preprocessed md → tex body via `quarto render`.

    We invoke `quarto render` (not `quarto pandoc`) so Quarto's Lua-filter
    chain runs — this is what turns `::: {#thm-X .theorem}` into
    `\\begin{theorem}...\\end{theorem}`, resolves `@thm-X` crossrefs, and
    handles labeled equations. Quarto emits a standalone tex document; we
    extract the body from between `\\begin{document}` and `\\end{document}`
    (dropping `\\maketitle`, `\\bibliography{…}`, etc.) and splice that into
    the overleaf preamble via BODY sentinels.
    """
    work = paper_dir(paper_slug) / ".build"
    work.mkdir(exist_ok=True)
    # Quarto accepts .qmd natively; using .md can trip default-format inference
    # in some setups.
    md_file = work / "source.qmd"
    md_file.write_text(_inject_quarto_meta(preprocessed_md))

    tex_file = work / "source.tex"
    if tex_file.exists():
        tex_file.unlink()

    try:
        run([
            "quarto", "render", str(md_file),
            "--to", "latex",
            "--output", tex_file.name,
            "--no-execute-daemon",
        ], cwd=work)
    except subprocess.CalledProcessError as e:
        typer.secho("quarto render failed:", fg="red")
        typer.echo(e.stderr or e.stdout)
        raise typer.Exit(1)

    if not tex_file.exists():
        # Quarto occasionally writes to a different path (e.g. source.tex
        # beside the input). Fall back to scanning the work dir.
        for candidate in work.glob("*.tex"):
            if candidate.stat().st_mtime >= md_file.stat().st_mtime:
                tex_file = candidate; break

    full_tex = tex_file.read_text()
    body = _strip_pandocbounded(_extract_tex_body(full_tex))
    body = _normalize_inline_math(body)
    body = _format_align_blocks(body)
    return body


# Pandoc emits `\(x\)` for `$x$` inline math; co-authors prefer the `$x$`
# form on Overleaf, and reverting it on every push churns Git history. We
# normalize after rendering so the source md stays human-friendly while
# the tex matches the team's house style.
_PANDOC_INLINE_MATH_RE = re.compile(r"\\\(([\s\S]*?)\\\)")


def _normalize_inline_math(tex: str) -> str:
    """Rewrite Pandoc's `\\(x\\)` inline math to `$x$`.

    Display math (`\\[...\\]`) is left alone — co-authors sometimes prefer
    `\\begin{align*}` for those, but that's a per-block judgment call we
    don't try to automate.
    """
    return _PANDOC_INLINE_MATH_RE.sub(lambda m: f"${m.group(1)}$", tex)


# Pandoc emits `\begin{align*}` glued to the preceding text and the
# closing `\end{align*}` glued to the following text, with no indentation
# of contents. Max's house style for the proof region is: each align* on
# its own line, contents indented 4 spaces. We reformat after rendering.
_ALIGN_BLOCK_RE = re.compile(
    r"\\begin\{align\*\}\s*\n(.*?)\n\\end\{align\*\}",
    re.DOTALL,
)


def _format_align_blocks(tex: str) -> str:
    """Break `align*` envs onto their own lines and indent contents 4 sp."""
    def _indent(m: re.Match) -> str:
        body = "\n".join("    " + ln if ln.strip() else ln
                         for ln in m.group(1).splitlines())
        return f"\\begin{{align*}}\n{body}\n\\end{{align*}}"

    out = _ALIGN_BLOCK_RE.sub(_indent, tex)
    # Break `... text \begin{align*}` onto a new line.
    out = re.sub(r"(\S) (\\begin\{align\*\})", r"\1\n\2", out)
    # Break `\end{align*} text ...` onto a new line.
    out = re.sub(r"(\\end\{align\*\}) (\S)", r"\1\n\2", out)
    return out


def _inject_quarto_meta(md: str) -> str:
    """Merge quarto-specific frontmatter (our filter, latex format options,
    disabled code execution) into the source md's existing YAML frontmatter.

    The order of keys in the merged block is: existing keys, then additions,
    so the user's title/authors/etc. stay authoritative.
    """
    injection = (
        f"execute:\n"
        f"  enabled: false\n"
        f"filters:\n"
        f"  - {filter_path()}\n"
        f"format:\n"
        f"  latex:\n"
        f"    cite-method: natbib\n"
        f"    keep-tex: true\n"
        f"    fig-pos: t\n"
    )
    if md.startswith("---\n"):
        end = md.find("\n---", 4)
        if end > 0:
            head = md[4:end]
            body = md[end + 4:].lstrip("\n")
            # Drop a trailing newline on `head` if present so the merge is tidy.
            head = head.rstrip("\n")
            return f"---\n{head}\n{injection}---\n\n{body}"
    return f"---\n{injection}---\n\n{md}"


_DOC_BEGIN_RE = re.compile(r"\\begin\{document\}\s*\n")
_DOC_END_RE = re.compile(r"\\end\{document\}")


def _extract_tex_body(tex: str) -> str:
    """Strip Quarto's standalone preamble + document wrapper, leaving just the
    body that should land between our BODY sentinels in main.tex."""
    bm = _DOC_BEGIN_RE.search(tex)
    em = _DOC_END_RE.search(tex)
    if not bm or not em:
        return tex
    body = tex[bm.end():em.start()]
    # Drop Quarto-generated title/abstract/bibliography scaffolding that our
    # overleaf preamble supplies itself.
    body = re.sub(r"\\maketitle\s*\n?", "", body)
    body = re.sub(r"\\bibliography\{[^}]+\}\s*\n?", "", body)
    body = re.sub(r"\\bibliographystyle\{[^}]+\}\s*\n?", "", body)
    body = re.sub(r"\\printbibliography(?:\[[^\]]*\])?\s*\n?", "", body)
    return body.strip() + "\n"


_PANDOCBOUNDED_RE = re.compile(
    r"\\pandocbounded\{\s*\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}\s*\}",
    re.DOTALL,
)


def _strip_pandocbounded(tex: str) -> str:
    """Unwrap `\\pandocbounded{\\includegraphics[...]{...}}` → idiomatic
    `\\includegraphics[width=\\linewidth, keepaspectratio]{...}`.

    Recent pandoc auto-wraps figures with `\\pandocbounded` to scale them to
    the current text block. That macro is defined in pandoc's standalone
    template — which we're not using, since we only want a tex *fragment* for
    the BODY sentinels. So the macro is undefined, and compilation fails. The
    idiomatic replacement is what a human would write anyway.
    """
    def _sub(m: re.Match) -> str:
        opts = (m.group(1) or "").strip()
        fname = m.group(2)
        base = "width=\\linewidth"
        new_opts = base if not opts else f"{base},{opts}"
        return f"\\includegraphics[{new_opts}]{{{fname}}}"
    return _PANDOCBOUNDED_RE.sub(_sub, tex)


def parse_frontmatter(md: str) -> dict:
    lines = md.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i; break
    if end is None:
        return {}
    raw = "\n".join(lines[1:end])
    try:
        return yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}


def assemble_main_tex(main_tex: Path, tex_body: str, frontmatter: dict):
    """Insert generated body into main.tex between sentinels, preserving preamble."""
    if main_tex.exists():
        existing = main_tex.read_text()
    else:
        existing = (templates_dir() / "article" / "main.tex").read_text()

    # Title/author substitution: only if template placeholders are still present.
    title = frontmatter.get("title", "")
    author = frontmatter.get("author") or frontmatter.get("authors") or ""
    thanks = frontmatter.get("thanks") or frontmatter.get("affiliation") or ""
    if isinstance(author, (list, tuple)):
        names = [str(a) for a in author]
        if thanks and names:
            names[-1] = f"{names[-1]}\\thanks{{{thanks}}}"
        author_tex = " \\\\\n  ".join(names)
        if thanks:
            author_str = "%\n  " + author_tex
        else:
            author_str = "%\n  " + author_tex
    else:
        author_str = str(author)
        if thanks:
            author_str = f"{author_str}\\thanks{{{thanks}}}"
    existing = existing.replace("BODY_TITLE", str(title))
    existing = existing.replace("BODY_AUTHOR", author_str)

    # Pull out the `# Appendix` portion (if present) so it lands in the
    # post-bibliography region instead of before references. We always do
    # the split — if the host template doesn't have APPENDIX sentinels, the
    # appendix half falls back into the main body region with a warning.
    main_body, appendix_body = split_appendix(tex_body)

    # Substitute body region.
    if BODY_BEGIN in existing and BODY_END in existing:
        before, _, rest = existing.partition(BODY_BEGIN)
        _, _, after = rest.partition(BODY_END)
        if appendix_body and (APPENDIX_BEGIN not in after or APPENDIX_END not in after):
            typer.secho(
                f"  Warning: rendered body has an Appendix section but "
                f"{main_tex.name} has no APPENDIX sentinels; appendix will "
                f"render before references. Add APPENDIX_BEGIN/END to the "
                f"template after \\bibliography{{...}} to fix.",
                fg="yellow",
            )
            body_to_splice = tex_body
        else:
            body_to_splice = main_body
        new = f"{before}{BODY_BEGIN}\n{body_to_splice}\n{BODY_END}{after}"
        if appendix_body and APPENDIX_BEGIN in new and APPENDIX_END in new:
            head, _, rest2 = new.partition(APPENDIX_BEGIN)
            _, _, tail = rest2.partition(APPENDIX_END)
            new = f"{head}{APPENDIX_BEGIN}\n{appendix_body}\n{APPENDIX_END}{tail}"
    elif "%%% BODY %%%" in existing:
        new = existing.replace(
            "%%% BODY %%%",
            f"{BODY_BEGIN}\n{tex_body}\n{BODY_END}",
        )
    else:
        # Insert before \end{document}
        parts = existing.rsplit("\\end{document}", 1)
        if len(parts) != 2:
            raise RuntimeError(
                f"{main_tex} has neither BODY sentinels nor \\end{{document}}; "
                f"cannot splice generated body."
            )
        new = f"{parts[0]}{BODY_BEGIN}\n{tex_body}\n{BODY_END}\n\\end{{document}}{parts[1]}"
    main_tex.write_text(new)


def verify_tex(main_tex: Path) -> bool:
    """Compile-verify with tectonic if available. Returns True on success or skip."""
    if shutil.which("tectonic") is None:
        typer.secho("  (tectonic not installed; skipping verify)", fg="yellow")
        return True
    try:
        run(["tectonic", "-X", "compile", "--outfmt", "pdf",
             "--keep-logs", str(main_tex)], cwd=main_tex.parent)
        typer.secho("  ✓ tectonic verified", fg="green")
        return True
    except subprocess.CalledProcessError as e:
        typer.secho("  ✗ tectonic compile failed", fg="red")
        typer.echo(e.stderr or e.stdout)
        return False


# ---------------------------------------------------------------------------
# Overleaf git wrappers
# ---------------------------------------------------------------------------

def ov_git(slug: str, *args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    ov = paper_dir(slug) / "overleaf"
    return run(["git", "-C", str(ov), *args], check=check, capture=capture)


def ov_head(slug: str, ref: str = "HEAD") -> str:
    return ov_git(slug, "rev-parse", ref).stdout.strip()


def ov_remote_has_changes(slug: str, branch: str) -> bool:
    """True if origin/<branch> is ahead of local HEAD."""
    ov_git(slug, "fetch", "origin")
    local = ov_head(slug, "HEAD")
    remote = ov_git(slug, "rev-parse", f"origin/{branch}").stdout.strip()
    return local != remote


# ---------------------------------------------------------------------------
# LLM back-sync — shells out to `claude -p` (Claude Code CLI)
# ---------------------------------------------------------------------------

PATCH_PROMPT = """You are translating a LaTeX edit back into the Markdown source of an academic paper.

The workflow: Obsidian Markdown → Pandoc + custom Lua filter → LaTeX → Overleaf.
A collaborator edited the LaTeX on Overleaf, and you must produce the equivalent
Markdown edit.

## Output contract (your output is parsed by a script)

Your ENTIRE response must be one of these two forms — nothing else:

FORM A — a unified diff:
    --- a/<basename>
    +++ b/<basename>
    @@ -<start>,<count> +<start>,<count> @@
     <unchanged context>
    -<removed line>
    +<added line>
     <unchanged context>

FORM B — explicit refusal:
    CANNOT-REPRESENT: <one-sentence reason>

Rules:
- NO prose, explanations, apologies, notes, or meta-commentary.
- NO wrapping ``` fences — emit the raw diff directly.
- The `<basename>` is exactly the filename I give you, verbatim (do not prepend paths).
- Compute @@ line numbers by reading the absolute line-number prefix on each
  line of the windowed markdown source I provide in the user message (each
  line is formatted `<N>: <content>`). Strip the `<N>: ` prefix when emitting
  context/diff lines, but USE those numbers in the @@ header. Do NOT invent,
  guess, or placeholder them.
- Include enough context lines for `patch -p1` to apply cleanly (≥3 each side).
- If the tex edit is inside a filter-elided region (e.g. an elided warning callout),
  or is a LaTeX primitive with no md analogue (e.g. \\vspace, \\clearpage), or
  requires restructuring the md document, emit FORM B.

## Filter conventions (use the md form in your diff)

These md ↔ tex mappings happen automatically on the next push. When the tex
edit touched one of these, emit the md equivalent:

- `[[@citekey]]` / `[@citekey]` ↔ `\\cite{citekey}` / `\\citep{citekey}`
- `> [!info]` / `[!note]` / `[!quote]` callouts ↔ `\\begin{quote}...\\end{quote}`
- `> [!abstract]` / `[!summary]` callouts ↔ `\\begin{abstract}...\\end{abstract}`
- `[sidenote text]` (bracketed inline text) ↔ `\\snKincaid{sidenote text}`
- `[Name: sidenote text]` (with a known author prefix) ↔ `\\sn<Name>{sidenote text}`
  (known authors: Kincaid, Max, Zeyu, Boyi, Peter, Dilip)
  Some authors have aliased their macro to a shorter form; these are
  equivalent and BOTH go to/from `[Name: ...]` in md:
    - `\\peter{text}` ↔ `[Peter: text]`
    - `\\dnote{text}` ↔ `[Dilip: text]`
- `![alt](URL-or-filename)` ↔ `\\includegraphics{...}` (often wrapped in `\\begin{figure}`)
- `![[local.png]]` ↔ `\\includegraphics{local.png}`
- `# Name` / `## Name` ↔ `\\section{Name}` / `\\subsection{Name}`
- `$x$` / `$$x$$` ↔ `\\(x\\)` / `\\[x\\]`

If a tex edit changes prose inside `\\snKincaid{...}`, the md edit goes inside
the matching `[...]` brackets. If inside `\\begin{quote}`, it goes inside the
`> [!info]` callout body. Etc.
"""


_WINDOW_RADIUS = 80


def _md_window(full_md: str, md_start: int, radius: int = _WINDOW_RADIUS
               ) -> tuple[str, int, int]:
    """Slice `radius` lines on each side of `md_start` and prepend each line
    with its absolute 1-indexed line number.

    Returns (annotated_window, lo, hi) where `lo`/`hi` are the inclusive
    1-indexed line bounds of the slice. The annotated form looks like:

        185: As LLMs are deployed ever more in agentic harnesses, ...
        186:
        187: This creates the possibility of LLM agents which are highly ...

    The LLM can read absolute line numbers off each line and use them
    directly in the @@ hunk header it emits.
    """
    lines = full_md.splitlines()
    if md_start is None or md_start < 1:
        # No reliable anchor — return full doc with line numbers.
        lo, hi = 1, len(lines)
    else:
        lo = max(1, md_start - radius)
        hi = min(len(lines), md_start + radius)
    width = len(str(hi))
    annotated = "\n".join(
        f"{i:>{width}}: {line}" for i, line in enumerate(lines[lo - 1:hi], lo)
    )
    return annotated, lo, hi


def draft_md_patch(md_context: str, md_basename: str, md_start: int,
                   tex_hunk: str, full_md: str) -> str:
    """Call `claude -p` to translate a tex diff hunk into an md unified diff.

    Passes a windowed slice of the md (centered on `md_start`) with absolute
    line numbers prepended, so the LLM can both find the affected passage
    and read off the correct line numbers for the @@ hunk header. Falls
    back to the full md when `md_start` is None.
    """
    window, lo, hi = _md_window(full_md, md_start)
    full_lines = full_md.count("\n") + 1
    if (lo, hi) == (1, full_lines):
        scope = f"the full markdown source ({full_lines} lines)"
    else:
        scope = (f"a windowed slice of the markdown source covering lines "
                 f"{lo}–{hi} (out of {full_lines} total). The slice is "
                 f"centered on the source-map hint.")
    user_msg = (
        f"# Markdown source (line-numbered)\n\n"
        f"Filename for the diff headers (use verbatim): `{md_basename}`.\n"
        f"Below is {scope} Each line is prefixed with its absolute "
        f"1-indexed line number followed by `: `; strip that prefix when "
        f"composing diff context lines, but USE those numbers in the @@ "
        f"hunk header.\n\n"
        f"```\n{window}\n```\n\n"
        f"# LaTeX diff hunk\n\n"
        f"```diff\n{tex_hunk}\n```\n\n"
        f"# Source-map hint\n\n"
        f"A fingerprint alignment suggests the md region around line "
        f"{md_start} produced this tex hunk. The window above is centered "
        f"there. If the affected passage isn't in the window, emit "
        f"CANNOT-REPRESENT — don't guess line numbers outside the window."
    )
    try:
        proc = subprocess.run(
            [
                "claude", "-p", user_msg,
                "--append-system-prompt", PATCH_PROMPT,
                "--model", "sonnet",
            ],
            capture_output=True, text=True, check=True, timeout=300,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "`claude` CLI not found in PATH. Install Claude Code or "
            "re-run pull-paper with --dry-run."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("`claude -p` timed out after 300s")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"`claude -p` failed (exit {e.returncode}): "
            f"{(e.stderr or e.stdout or '').strip()[:500]}"
        )
    return _extract_diff(proc.stdout.strip())


_HUNK_HEADER_FIX_RE = re.compile(
    r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$"
)


def _recount_hunk_headers(diff: str) -> str:
    """Rewrite `@@ -N,M +P,Q @@` counts to match the actual hunk body lines.

    LLMs reliably miscount these (Sonnet's off-by-one is especially common),
    which `git apply` rejects as "corrupt patch". Recomputing server-side is
    safer than prompting harder.
    """
    lines = diff.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _HUNK_HEADER_FIX_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        pre_start, post_start, tail = m.group(1), m.group(2), m.group(3)
        body_start = i + 1
        j = body_start
        pre_count = 0
        post_count = 0
        while j < len(lines):
            ln = lines[j]
            if ln.startswith("@@") or ln.startswith("--- ") or ln.startswith("+++ "):
                break
            if ln.startswith("-") and not ln.startswith("---"):
                pre_count += 1
            elif ln.startswith("+") and not ln.startswith("+++"):
                post_count += 1
            elif ln.startswith(" ") or ln == "":
                pre_count += 1
                post_count += 1
            else:
                break
            j += 1
        new_hdr = (
            f"@@ -{pre_start},{pre_count} +{post_start},{post_count} @@{tail}"
        )
        out.append(new_hdr)
        out.extend(lines[body_start:j])
        i = j
    return "\n".join(out) + ("\n" if diff.endswith("\n") else "")


def _extract_diff(text: str) -> str:
    """Extract a unified diff (or CANNOT-REPRESENT sentinel) from a possibly
    noisy LLM response. Tries, in order:

      1. Response already looks like a raw diff (starts with `--- ` or `@@`).
      2. Response is a single wrapping ```[lang] ... ``` fence.
      3. Response contains a ``` ... ``` block whose content is a diff.
      4. Response contains a `CANNOT-REPRESENT:` line.
      5. Fallback: return a CANNOT-REPRESENT sentinel so `cmd_accept` refuses
         to apply nonsense.
    """
    s = text.strip()
    if s.startswith("--- ") or s.startswith("@@ "):
        return s
    m = re.match(r"^```[a-zA-Z]*\n(.*?)\n```\s*$", s, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        if inner.startswith("--- ") or inner.startswith("@@ "):
            return inner
        # Pure fenced but not a diff — maybe CANNOT-REPRESENT inside
        for line in inner.splitlines():
            if line.startswith("CANNOT-REPRESENT:"):
                return line
    for fence_m in re.finditer(r"```[a-zA-Z]*\n(.*?)\n```", s, re.DOTALL):
        inner = fence_m.group(1).strip()
        if inner.startswith("--- ") or inner.startswith("@@ "):
            return inner
    for line in s.splitlines():
        if line.startswith("CANNOT-REPRESENT:"):
            return line
    return (
        "CANNOT-REPRESENT: LLM response contained no parseable unified diff "
        f"(first 200 chars: {s[:200]!r})"
    )


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------

@dataclass
class TexHunk:
    pre_start: int       # line in the pre-edit tex file (1-based)
    pre_count: int
    post_start: int
    post_count: int
    body: str            # full hunk text including @@ header


HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(diff_text: str) -> list[TexHunk]:
    hunks: list[TexHunk] = []
    current_body: list[str] = []
    current_header: Optional[re.Match] = None
    for line in diff_text.splitlines(keepends=False):
        m = HUNK_HEADER_RE.match(line)
        if m:
            if current_header is not None:
                hunks.append(_mkhunk(current_header, current_body))
            current_header = m
            current_body = [line]
        else:
            if current_header is not None:
                current_body.append(line)
    if current_header is not None:
        hunks.append(_mkhunk(current_header, current_body))
    return hunks


def _mkhunk(header: re.Match, body_lines: list[str]) -> TexHunk:
    pre_start = int(header.group(1))
    pre_count = int(header.group(2) or "1")
    post_start = int(header.group(3))
    post_count = int(header.group(4) or "1")
    return TexHunk(
        pre_start=pre_start, pre_count=pre_count,
        post_start=post_start, post_count=post_count,
        body="\n".join(body_lines),
    )


def locate_md_range_for_tex_hunk(hunk: TexHunk, source_map: list[dict]) -> Optional[tuple[int, int, float]]:
    """Find md_start, md_end, confidence for a tex hunk using the source map."""
    lo = hunk.pre_start
    hi = hunk.pre_start + max(hunk.pre_count - 1, 0)
    best: Optional[tuple[int, int, float]] = None
    for entry in source_map:
        ts = entry.get("tex_start")
        te = entry.get("tex_end")
        ms = entry.get("md_start")
        me = entry.get("md_end")
        if ts is None or te is None or ms is None or me is None:
            continue
        if te < lo or ts > hi:
            continue
        overlap = min(te, hi) - max(ts, lo) + 1
        span = max(te - ts + 1, 1)
        score = overlap / span
        conf = float(entry.get("confidence", 0.0)) * score
        if best is None or conf > best[2]:
            best = (int(ms), int(me), conf)
    return best


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False, help=__doc__)


@app.command("import-template")
def cmd_import_template(name: str, source: str, force: bool = False):
    """Import a template zip into .zetteldev/paper/templates/<name>/.

    SOURCE may be an http(s) URL or a local zip path. Flattens a single
    top-level directory inside the zip so files land at the template root.
    """
    import tempfile
    import zipfile

    dst = templates_dir() / name
    if dst.exists():
        if not force and not questionary.confirm(
            f"{dst} exists. Replace?", default=False,
        ).ask():
            raise typer.Exit(1)
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    if source.startswith(("http://", "https://")):
        typer.echo(f"Fetching {source} …")
        r = requests.get(source, timeout=60)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        tmp.write(r.content)
        tmp.close()
        zip_path = Path(tmp.name)
        fetched_from = source
    else:
        zip_path = Path(source).expanduser().resolve()
        if not zip_path.exists():
            typer.secho(f"No such file: {zip_path}", fg="red")
            raise typer.Exit(1)
        fetched_from = str(zip_path)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dst)

    # Flatten if the zip contained a single top-level directory.
    contents = [c for c in dst.iterdir() if not c.name.startswith("__MACOSX")]
    for junk in dst.iterdir():
        if junk.name.startswith("__MACOSX"):
            shutil.rmtree(junk)
    contents = list(dst.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        inner = contents[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), str(dst / item.name))
        inner.rmdir()

    # Best-effort detect main tex (first .tex with \documentclass).
    main_guess = None
    for t in sorted(dst.rglob("*.tex")):
        if "\\documentclass" in t.read_text(errors="ignore"):
            main_guess = str(t.relative_to(dst))
            break

    meta = {"name": name, "source": fetched_from}
    if main_guess:
        meta["main_tex"] = main_guess
    (dst / "template.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    typer.secho(f"✓ imported template '{name}' from {fetched_from}", fg="green")
    typer.echo(f"  main_tex guess: {main_guess or '(none — set manually in template.yaml)'}")
    typer.echo(f"  next: just apply-paper-template <slug> {name}")


@app.command("apply-template")
def cmd_apply_template(slug: Optional[str] = typer.Argument(None),
                       template: Optional[str] = typer.Argument(None),
                       force: bool = False):
    """Copy a template's files into overleaf/, preserving existing files.

    Existing files in overleaf/ (including subfolders from prior drafts) are
    NOT touched unless --force. Only new paths land. main_tex and template
    fields in paper.yaml are updated to match.

    If there's only one paper in the repo, slug may be omitted:
      apply-template <template>
    """
    # Positional shift: if only one arg provided, it's the template.
    if template is None and slug is not None:
        template = slug
        slug = None
    if template is None:
        typer.secho("template is required", fg="red")
        raise typer.Exit(1)
    slug = resolve_slug(slug)

    cfg = PaperConfig.load(slug)
    ov = paper_dir(slug) / "overleaf"
    tpl = templates_dir() / template
    if not tpl.exists():
        typer.secho(
            f"No template at {tpl}.\n"
            f"Import one first: just import-paper-template {template} <url-or-zip>",
            fg="red",
        )
        raise typer.Exit(1)

    meta_path = tpl / "template.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}

    copied, skipped = [], []
    for src in sorted(tpl.rglob("*")):
        if src.is_dir():
            continue
        if src.name == "template.yaml":
            continue
        rel = src.relative_to(tpl)
        dst = ov / rel
        if dst.exists() and not force:
            skipped.append(str(rel))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(rel))

    for f in copied:
        typer.echo(f"  + {f}")
    for f in skipped:
        typer.echo(f"  = {f} (exists; --force to overwrite)")

    # Resolve main tex: template.yaml wins, else auto-detect, else ask.
    main_tex_name = meta.get("main_tex")
    if not main_tex_name:
        candidates = []
        for t in sorted(ov.glob("*.tex")):
            if "\\documentclass" in t.read_text(errors="ignore"):
                candidates.append(t.name)
        if len(candidates) == 1:
            main_tex_name = candidates[0]
        elif len(candidates) > 1:
            main_tex_name = questionary.select(
                "Multiple top-level .tex files with \\documentclass. Main entry?",
                choices=candidates,
            ).ask()
    if main_tex_name:
        cfg.main_tex = main_tex_name
    cfg.template = template
    cfg.save()

    typer.secho(
        f"✓ applied '{template}' to {slug} (main_tex: {cfg.main_tex})",
        fg="green",
    )
    typer.echo("Next: just push-paper " + slug)


@app.command("new")
def cmd_new(slug: str):
    """Scaffold papers/<slug>/ and clone the Overleaf repo."""
    pdir = paper_dir(slug)
    if pdir.exists():
        typer.secho(f"{pdir} already exists. Aborting.", fg="red")
        raise typer.Exit(1)

    url = questionary.text("Overleaf git URL (https://git.overleaf.com/<project_id>)").ask()
    if not url:
        typer.secho("Overleaf URL is required.", fg="red")
        raise typer.Exit(1)

    default_md = Path.home() / "Pumberton" / "Workshop" / f"{slug}.md"
    source_md = questionary.text(
        f"Source md path", default=str(default_md)
    ).ask()
    refs_bib = questionary.text(
        "Zotero refs.bib path",
        default=str(Path.home() / "Drive" / "references.bib"),
    ).ask()
    template = questionary.select(
        "Template (used if the Overleaf repo is empty)",
        choices=["article"],  # expand as more templates land
        default="article",
    ).ask()
    branch = questionary.text("Overleaf default branch", default="master").ask()

    pdir.mkdir(parents=True)
    (pdir / "review").mkdir()

    # Clone
    typer.echo(f"Cloning {url} → {pdir}/overleaf/ …")
    try:
        run(["git", "clone", url, str(pdir / "overleaf")])
    except subprocess.CalledProcessError:
        shutil.rmtree(pdir)
        typer.secho("git clone failed. Paper dir removed.", fg="red")
        raise typer.Exit(1)

    # Seed template if repo is empty
    ov = pdir / "overleaf"
    tex_files = list(ov.glob("*.tex"))
    if not tex_files:
        if questionary.confirm(
            f"Overleaf repo looks empty. Seed with {template} template?",
            default=True,
        ).ask():
            src = templates_dir() / template
            for item in src.iterdir():
                shutil.copy2(item, ov / item.name)
            typer.echo(f"  seeded {template} template")

    cfg = PaperConfig(
        slug=slug,
        overleaf_git_url=url,
        source_md=expand(source_md),
        refs_bib=expand(refs_bib),
        template=template,
        main_tex="main.tex",
        branch=branch,
    )
    cfg.save()

    if not cfg.source_md.exists():
        if questionary.confirm(
            f"Source md {cfg.source_md} doesn't exist. Create a stub?",
            default=True,
        ).ask():
            cfg.source_md.parent.mkdir(parents=True, exist_ok=True)
            cfg.source_md.write_text(
                f"---\ntitle: {slug}\nauthor: \n---\n\n# Introduction\n\n"
            )

    typer.secho(f"✓ paper '{slug}' scaffolded at {pdir}", fg="green")


def _common_push(slug: str, commit: bool) -> int:
    slug = resolve_slug(slug)
    cfg = PaperConfig.load(slug)
    ledger = Ledger.load(slug)
    pdir = paper_dir(slug)
    ov = pdir / "overleaf"

    if not cfg.source_md.exists():
        typer.secho(f"Source md not found: {cfg.source_md}", fg="red")
        raise typer.Exit(1)

    # Refuse if Overleaf has unpulled changes.
    try:
        if ov_remote_has_changes(slug, cfg.branch):
            last = ledger.last_inbound_overleaf_commit
            head = ov_head(slug)
            remote = ov_git(slug, "rev-parse", f"origin/{cfg.branch}").stdout.strip()
            if remote != last and remote != head:
                typer.secho(
                    "Overleaf has unreviewed changes. Run `just pull-paper "
                    f"{slug}` first.", fg="red")
                raise typer.Exit(1)
    except subprocess.CalledProcessError:
        typer.secho("  (could not check remote; continuing)", fg="yellow")

    md_text = cfg.source_md.read_text()
    annotated, n_annotated = annotate_figures(md_text)
    if n_annotated > 0:
        typer.echo(
            f"  annotated {n_annotated} image(s) with default attrs "
            f"({DEFAULT_FIG_ATTRS}) — overwrite per-image with manual `{{…}}`"
        )
        cfg.source_md.write_text(annotated)
        md_text = annotated
    md_sha = sha256_short(md_text)
    typer.echo(f"source md sha: {md_sha}")

    pre_md, r2_urls, local_embeds = preprocess_md(md_text)
    typer.echo(f"R2 figures referenced: {len({canonicalize_r2_url(u) for u in r2_urls})}")
    if local_embeds:
        typer.echo(f"Local embed assets: {len(set(local_embeds))}")

    url_to_local = pull_r2_figures(r2_urls, slug, ledger)
    pre_md = rewrite_r2_in_md(pre_md, url_to_local)

    name_to_safe = copy_local_embeds(local_embeds, cfg.source_md, slug, ledger)
    pre_md = rewrite_local_embeds(pre_md, name_to_safe)

    tex_body = render_tex_body(pre_md, slug)

    main_tex_path = ov / cfg.main_tex
    frontmatter = parse_frontmatter(md_text)
    assemble_main_tex(main_tex_path, tex_body, frontmatter)

    missing = merge_bib(md_text, cfg.refs_bib, ov / "refs.bib")
    if missing:
        typer.secho(f"  missing bib entries for: {', '.join(sorted(missing))}", fg="yellow")

    # Build source map between raw md and generated tex body.
    md_blocks = split_md_blocks(md_text)
    tex_blocks = split_tex_blocks(tex_body)
    mappings = align_blocks(md_blocks, tex_blocks)

    # Compute the offset between body-coordinates (used in mappings) and
    # main.tex file-coordinates (used in git diff hunks at pull time).
    # `body_line_offset` is the file line that comes immediately BEFORE body
    # line 1 — i.e. `file_line = body_line_offset + body_line`.
    main_tex_content = main_tex_path.read_text()
    body_line_offset = 0
    for idx, line in enumerate(main_tex_content.splitlines(), 1):
        if BODY_BEGIN in line:
            body_line_offset = idx
            break

    (pdir / ".source-map.json").write_text(json.dumps({
        "generated_at": now_iso(),
        "md_sha": md_sha,
        "tex_sha": sha256_short(main_tex_content),
        "md_source": str(cfg.source_md),
        "main_tex": cfg.main_tex,
        "body_line_offset": body_line_offset,
        "blocks": mappings,
    }, indent=2) + "\n")

    # Snapshot the pre-edit tex for future back-sync.
    snapdir = pdir / ".tex-snapshots"
    snapdir.mkdir(exist_ok=True)
    tex_sha = sha256_short(main_tex_path.read_text())
    snap = snapdir / f"{tex_sha}.tex"
    snap.write_text(main_tex_path.read_text())

    # Verify.
    ok = verify_tex(main_tex_path)
    if not ok and commit:
        typer.secho(
            "Refusing to push: tectonic verification failed. Fix the errors "
            "above, or re-run with `just paper-diff` to iterate locally.",
            fg="red",
        )
        raise typer.Exit(1)

    if not commit:
        typer.secho("[diff] would commit + push here (skipped)", fg="cyan")
        return 0

    # Commit + push.
    ov_git(slug, "add", "-A", check=True)
    status = ov_git(slug, "status", "--porcelain").stdout
    if not status.strip():
        typer.secho("No changes to push.", fg="yellow")
        return 0
    ov_git(slug, "commit", "-m", f"sync from obsidian @ {md_sha}")
    ov_git(slug, "push", "origin", cfg.branch)
    head = ov_head(slug)
    typer.secho(f"✓ pushed {head} to Overleaf", fg="green")

    ledger.last_outbound_md_sha = md_sha
    ledger.last_outbound_tex_sha = tex_sha
    ledger.last_outbound_tex_snapshot = str(snap.relative_to(pdir))
    ledger.last_outbound_overleaf_commit = head
    ledger.save(slug)
    return 0


@app.command("push")
def cmd_push(slug: Optional[str] = typer.Argument(None)):
    """Render md → tex, fetch R2 figures, commit + push to Overleaf."""
    raise typer.Exit(_common_push(resolve_slug(slug), commit=True))


@app.command("diff")
def cmd_diff(slug: Optional[str] = typer.Argument(None)):
    """Preview outbound changes (md → tex + figures) without committing."""
    raise typer.Exit(_common_push(resolve_slug(slug), commit=False))


@app.command("figures")
def cmd_figures(slug: Optional[str] = typer.Argument(None)):
    """Refresh only overleaf/figures/ from R2 (no tex regeneration)."""
    slug = resolve_slug(slug)
    cfg = PaperConfig.load(slug)
    ledger = Ledger.load(slug)
    md_text = cfg.source_md.read_text()
    _, urls, local_embeds = preprocess_md(md_text)
    pull_r2_figures(urls, slug, ledger)
    copy_local_embeds(local_embeds, cfg.source_md, slug, ledger)
    ledger.save(slug)
    total = len({canonicalize_r2_url(u) for u in urls}) + len(set(local_embeds))
    typer.secho(f"✓ figures refreshed ({total} assets)", fg="green")


@app.command("pull")
def cmd_pull(slug: Optional[str] = typer.Argument(None), dry_run: bool = False):
    """Pull Overleaf, diff against last outbound, draft md review patches.

    With --dry-run, parses the incoming hunks and maps them to md ranges but
    skips the LLM calls — useful for previewing a large incoming diff without
    spending tokens.
    """
    slug = resolve_slug(slug)
    cfg = PaperConfig.load(slug)
    ledger = Ledger.load(slug)
    pdir = paper_dir(slug)

    if not dry_run and shutil.which("claude") is None:
        typer.secho(
            "`claude` CLI not found in PATH. Install Claude Code, or re-run "
            "with --dry-run to map hunks without drafting patches.",
            fg="red",
        )
        raise typer.Exit(1)

    # Reset the working tree before pulling. paper-diff and earlier push runs
    # leave the generated tex in the working tree; those regenerations are
    # always reproducible from the md, so discarding them is safe. Without
    # this, `git pull --ff-only` refuses when there are unstaged changes.
    ov_git(slug, "checkout", "--", ".", check=False)
    ov_git(slug, "fetch", "origin")
    try:
        ov_git(slug, "pull", "--ff-only", "origin", cfg.branch)
    except subprocess.CalledProcessError as e:
        typer.secho(
            "Overleaf pull is not fast-forward. Local overleaf/ has diverged "
            "from the remote — inspect `git -C papers/"
            + slug + "/overleaf status` and resolve before re-running.",
            fg="red",
        )
        typer.echo(e.stderr or e.stdout)
        raise typer.Exit(1)

    last = ledger.last_outbound_overleaf_commit
    current = ov_head(slug)
    if last is None:
        typer.secho("No prior outbound sync — nothing to diff against.", fg="yellow")
        return
    if last == current:
        typer.echo("No new Overleaf commits.")
        return

    try:
        name_status = ov_git(
            slug, "diff", "--name-status", f"{last}..{current}",
        ).stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        typer.secho(
            f"Could not diff {last[:7]}..{current[:7]}. Overleaf history may "
            "have been rewritten. Inspect the repo manually.", fg="red")
        raise typer.Exit(1)

    other_changes = []
    main_changed = False
    for line in name_status:
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status, path = parts[0], parts[1]
        if path == cfg.main_tex:
            main_changed = True
        else:
            other_changes.append((status, path))

    if other_changes:
        typer.secho(
            f"  Overleaf also touched {len(other_changes)} non-main file(s):",
            fg="cyan",
        )
        for status, path in other_changes:
            typer.echo(f"    [{status}] {path}")
        typer.echo(
            "  These are NOT round-tripped to md. Inspect and merge by hand.")

    if not main_changed:
        typer.echo(f"No changes to {cfg.main_tex}; nothing to draft.")
        ledger.last_inbound_overleaf_commit = current
        ledger.save(slug)
        return

    diff = ov_git(slug, "diff", "--unified=3",
                  f"{last}..{current}", "--", cfg.main_tex).stdout
    hunks = parse_unified_diff(diff)
    if not hunks:
        typer.echo(f"No hunks in {cfg.main_tex} between {last[:7]}..{current[:7]}.")
        ledger.last_inbound_overleaf_commit = current
        ledger.save(slug)
        return

    sm_path = pdir / ".source-map.json"
    if not sm_path.exists():
        typer.secho("No source map; cannot back-sync. Did you `just push-paper`?", fg="red")
        raise typer.Exit(1)
    sm_doc = json.loads(sm_path.read_text())
    source_map = sm_doc.get("blocks", [])
    body_line_offset = sm_doc.get("body_line_offset", 0)

    md_text = cfg.source_md.read_text()
    md_sha = sha256_short(md_text)
    md_lines = md_text.splitlines()
    review_dir = pdir / "review"
    review_dir.mkdir(exist_ok=True)

    date_prefix = datetime.now().strftime("%Y-%m-%d")
    written = []
    for idx, hunk in enumerate(hunks, start=1):
        # The source map indexes tex BODY lines; git diff hunks are in full
        # main.tex coordinates. Translate before lookup.
        body_hunk = TexHunk(
            pre_start=max(1, hunk.pre_start - body_line_offset),
            pre_count=hunk.pre_count,
            post_start=max(1, hunk.post_start - body_line_offset),
            post_count=hunk.post_count,
            body=hunk.body,
        )
        hit = locate_md_range_for_tex_hunk(body_hunk, source_map)
        if hit is None:
            typer.secho(f"  hunk {idx}: no md mapping (insertion or lost context)",
                        fg="yellow")
            md_start = md_end = None
            md_context = ""
            confidence = 0.0
        else:
            md_start, md_end, confidence = hit
            md_context = "\n".join(md_lines[md_start - 1:md_end])

        if dry_run:
            typer.echo(f"  hunk {idx}/{len(hunks)}: md {md_start}-{md_end} "
                       f"(confidence {confidence:.2f}) [dry-run, skipping LLM]")
            patch_body = "DRY-RUN: re-run without --dry-run to draft"
        else:
            typer.echo(f"  hunk {idx}/{len(hunks)}: md {md_start}-{md_end} "
                       f"(confidence {confidence:.2f}) → drafting patch")
            try:
                patch_body = draft_md_patch(
                    md_context=md_context,
                    md_basename=cfg.source_md.name,
                    md_start=md_start or 1,
                    tex_hunk=hunk.body,
                    full_md=md_text,
                )
            except Exception as e:
                typer.secho(f"    LLM error: {e}", fg="red")
                patch_body = f"CANNOT-REPRESENT: {e}"

        # Content-hash-tag the filename so repeat pulls don't overwrite prior drafts.
        hunk_tag = sha256_short(hunk.body, 6)
        pf = review_dir / (
            f"{date_prefix}-{current[:7]}-hunk-{idx:02d}-{hunk_tag}.patch.md"
        )
        pf.write_text(_render_patch_doc(
            slug=slug, commit=current, hunk=hunk, md_range=(md_start, md_end),
            confidence=confidence, md_context=md_context, patch=patch_body,
            md_sha_at_draft=md_sha,
        ))
        written.append(pf)

    if not dry_run:
        ledger.last_inbound_overleaf_commit = current
        ledger.save(slug)
    typer.secho(f"{'[dry-run] ' if dry_run else '✓ '}wrote "
                f"{len(written)} patch(es) to {review_dir}",
                fg="green")
    if not dry_run:
        typer.echo("Review them, then `just paper-accept <slug> <patch-id>`.")


@app.command("pending")
def cmd_pending(slug: Optional[str] = typer.Argument(None)):
    """List drafted review patches awaiting acceptance."""
    slug = resolve_slug(slug)
    pdir = paper_dir(slug)
    review_dir = pdir / "review"
    if not review_dir.exists():
        typer.echo("(no review/ dir yet)")
        return
    patches = sorted(
        p for p in review_dir.glob("*.patch.md") if p.is_file()
    )
    if not patches:
        typer.echo("No pending review patches.")
        return
    for p in patches:
        txt = p.read_text()
        commit = _field(txt, "Overleaf commit") or "?"
        conf = _field(txt, "Source-map confidence") or "?"
        status = "CANNOT-REPRESENT" if "CANNOT-REPRESENT" in txt else (
            "DRY-RUN" if "DRY-RUN" in txt else "ready"
        )
        typer.echo(f"  {p.name}")
        typer.echo(f"      commit {commit}  conf {conf}  status: {status}")


def _field(doc: str, label: str) -> Optional[str]:
    m = re.search(rf"- {re.escape(label)}: `?([^`\n]+)`?", doc)
    return m.group(1).strip() if m else None


def _render_patch_doc(slug: str, commit: str, hunk: TexHunk,
                      md_range: tuple, confidence: float,
                      md_context: str, patch: str,
                      md_sha_at_draft: str) -> str:
    md_start, md_end = md_range
    return (
        f"# Review patch — {slug}\n\n"
        f"- Overleaf commit: `{commit}`\n"
        f"- Tex hunk at tex lines {hunk.pre_start}–"
        f"{hunk.pre_start + hunk.pre_count - 1}\n"
        f"- Md range: {md_start}–{md_end}\n"
        f"- Source-map confidence: {confidence:.2f}\n"
        f"- Md sha at draft: `{md_sha_at_draft}`\n\n"
        "## Md context\n\n"
        f"```markdown\n{md_context}\n```\n\n"
        "## Tex hunk\n\n"
        f"```diff\n{hunk.body}\n```\n\n"
        "## Drafted md patch\n\n"
        f"```diff\n{patch}\n```\n"
    )


@app.command("accept")
def cmd_accept(slug: Optional[str] = typer.Argument(None),
               patch_id: Optional[str] = typer.Argument(None),
               force: bool = False):
    """Apply a drafted patch in review/ to the source md.

    Refuses to apply if the md has been edited since the patch was drafted
    (the md_sha_at_draft field no longer matches). --force overrides.

    If there's only one paper in the repo, slug may be omitted:
      accept <patch_id>
    """
    # Positional shift: one arg → it's the patch_id, auto-resolve slug.
    if patch_id is None and slug is not None:
        patch_id = slug
        slug = None
    if patch_id is None:
        typer.secho("patch_id is required", fg="red")
        raise typer.Exit(1)
    slug = resolve_slug(slug)
    cfg = PaperConfig.load(slug)
    pdir = paper_dir(slug)
    review_dir = pdir / "review"
    candidates = list(review_dir.glob(f"*{patch_id}*"))
    if not candidates:
        typer.secho(f"No patch matching '{patch_id}' in {review_dir}", fg="red")
        raise typer.Exit(1)
    if len(candidates) > 1:
        typer.secho("Ambiguous patch id:", fg="red")
        for c in candidates:
            typer.echo(f"  {c.name}")
        raise typer.Exit(1)
    pfile = candidates[0]
    text = pfile.read_text()

    # Freshness check.
    draft_sha = _field(text, "Md sha at draft")
    if draft_sha:
        current_sha = sha256_short(cfg.source_md.read_text())
        if current_sha != draft_sha and not force:
            typer.secho(
                f"Source md has changed since this patch was drafted "
                f"(draft sha {draft_sha}, current {current_sha}).\n"
                f"Re-run `just pull-paper {slug}` to regenerate patches, "
                f"or pass --force to apply anyway.",
                fg="red",
            )
            raise typer.Exit(1)

    m = re.search(r"## Drafted md patch\s*\n\n```diff\n(.*?)\n```", text, re.DOTALL)
    if not m:
        typer.secho("Could not extract diff from patch doc.", fg="red")
        raise typer.Exit(1)
    diff_text = m.group(1)
    if diff_text.startswith(("CANNOT-REPRESENT", "DRY-RUN")):
        typer.secho(f"Patch is not applyable: {diff_text}", fg="red")
        raise typer.Exit(1)

    diff_text = _recount_hunk_headers(diff_text)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f:
        f.write(diff_text if diff_text.endswith("\n") else diff_text + "\n")
        diff_path = f.name
    try:
        # First try `git apply` — strict, fast, handles filenames with spaces
        # via --directory.
        ga = subprocess.run([
            "git", "apply", "--unsafe-paths",
            "--directory", str(cfg.source_md.parent) + "/",
            diff_path,
        ], capture_output=True, text=True)
        if ga.returncode == 0:
            pass
        else:
            # Fallback: GNU patch with fuzzy context matching. Crucial when
            # the md has drifted (lines shifted) since the patch was drafted.
            # Pass the target file explicitly so patch ignores the diff
            # header filename (which `patch -p1` would otherwise mangle on
            # filenames with spaces).
            pa = subprocess.run([
                "patch", str(cfg.source_md), "-i", diff_path,
                "-p1", "--fuzz=3", "--no-backup-if-mismatch", "--silent",
            ], capture_output=True, text=True)
            if pa.returncode != 0:
                typer.secho(
                    f"git apply failed:\n  {(ga.stderr or ga.stdout).strip()}\n"
                    f"patch fallback failed:\n  {(pa.stderr or pa.stdout).strip()}",
                    fg="red",
                )
                typer.echo("Drafted diff was:")
                typer.echo(diff_text)
                raise typer.Exit(1)
            typer.secho("  (applied via patch with fuzzy matching)", fg="cyan")
    finally:
        os.unlink(diff_path)

    archive = pdir / "review" / "applied"
    archive.mkdir(exist_ok=True)
    shutil.move(str(pfile), archive / pfile.name)
    typer.secho(f"✓ applied {pfile.name} to {cfg.source_md}", fg="green")
    typer.echo("Re-run `just push-paper` to close the loop.")


if __name__ == "__main__":
    app()
