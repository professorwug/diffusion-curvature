# Paper Bridge

Obsidian markdown ↔ Overleaf LaTeX sync. See
[[2026-04-22]] daily note for the full design.

## Flow

```
~/Pumberton/Workshop/<paper>.md                 ← canonical source (human edits)
         │
         │  just push-paper <slug>
         ▼
preprocess (wikilinks, embeds, citations)
         │
         ▼
quarto pandoc + obsidian-callouts.lua           → tex body
         │
         ▼
assemble into overleaf/main.tex between BODY sentinels
         │
         ▼
pull R2 figures referenced in md → overleaf/figures/
         │
         ▼
merge cited bibtex entries from Zotero refs.bib → overleaf/refs.bib
         │
         ▼
[tectonic verify]  →  commit + push to Overleaf remote
         │
         ▼
.ledger.json updated; .source-map.json written
```

Inbound (`just pull-paper <slug>`): git pull Overleaf → diff main.tex against
last outbound snapshot → for each hunk, look up owning md range in the source
map → LLM drafts a unified md diff → land in `review/` for human review.
Never auto-applied; `just paper-accept` applies after review.

## Layout

```
papers/<slug>/
├── paper.yaml              overleaf_url, source_md path, refs_bib path
├── overleaf/               git clone of the Overleaf project
│   ├── main.tex            generated body between BODY sentinels
│   ├── figures/            pulled from R2
│   └── refs.bib            merged from Zotero + manual adds
├── review/                 LLM-drafted inbound patches (one .patch.md each)
│   └── applied/            accepted patches are archived here
├── .ledger.json            last_outbound_{md_sha,tex_sha,overleaf_commit}, figure hashes
├── .source-map.json        md↔tex block alignment for back-sync
├── .tex-snapshots/         pre-edit tex snapshots keyed by sha
└── .build/                 transient preprocessed md + quarto output
```

## Filter conventions

The custom Lua filter at `.zetteldev/paper/filters/obsidian-callouts.lua`
translates Obsidian blockquote idioms:

| Md | Tex |
|---|---|
| `> [!info]` / `[!note]` / `[!quote]` | `\begin{quote}...\end{quote}` |
| `> [!abstract]` / `[!summary]` | prose (no wrapper) |
| `> [!warning]` / `[!caution]` / `[!danger]` | `% callout elided` |
| `>` without `[!type]` | `% researcher-commentary elided` |

Wikilinks, embeds, and citations are handled by `preprocess_md` in `sync.py`
before Pandoc sees the document:

| Md | Pandoc-land |
|---|---|
| `[[@smith2020]]` | `[@smith2020]` → `\cite{smith2020}` |
| `[[Name\|alias]]` | `alias` |
| `[[Name]]` | `Name` |
| `![[foo.png]]` | `![](foo.png)` |
| `![...](https://blots.kincaid.ink/...)` | kept; R2 URL pulled to `overleaf/figures/` and rewritten to local path |
