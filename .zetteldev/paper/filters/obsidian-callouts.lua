-- obsidian-callouts.lua
-- Pandoc/Quarto Lua filter that translates Obsidian-specific blockquote idioms
-- into LaTeX, while leaving all other AST transformations to Pandoc's defaults.
--
-- Handles two blockquote cases:
--   1. `> [!type] optional title\n> body...` — Obsidian callout.
--   2. `> body...` with no `[!type]` marker — researcher commentary; elided.
--
-- See papers/README.md in the repo for the invocation contract.

local stringify = (require 'pandoc.utils').stringify

-- Map of callout type -> LaTeX rendering strategy.
-- `quote`    wraps in \begin{quote}...\end{quote}.
-- `abstract` wraps in \begin{abstract}...\end{abstract}.
-- `prose`    emits the body as plain blocks (no wrapper).
-- `elide`    drops the content with a provenance comment.
local STRATEGY = {
  info     = "quote",
  note     = "quote",
  quote    = "quote",
  tip      = "quote",
  example  = "quote",
  abstract = "abstract",
  summary  = "abstract",
  tldr     = "prose",
  warning  = "elide",
  caution  = "elide",
  danger   = "elide",
  bug      = "elide",
  todo     = "elide",
  question = "elide",
  failure  = "elide",
  success  = "elide",
}

-- Detect an Obsidian callout marker at the start of a Para's inlines and
-- return (typename, remaining_inlines_after_marker_and_title). Returns
-- (nil, nil) if no marker is present.
--
-- Pandoc parses `> [!info] title\n> body` as a Para whose first inline is a
-- single Str `"[!info]"` (brackets fused with the marker text). Following
-- inlines are an optional Space + Str title, a SoftBreak, then body.
-- The optional `-`/`+` fold indicator from Obsidian lives inside the brackets
-- as `[!info]-` or `[!info]+`.
local function split_marker(inlines)
  if #inlines < 1 then return nil, nil end
  if inlines[1].t ~= "Str" then return nil, nil end
  local typename = inlines[1].text:match("^%[!([%w_-]+)%][-+]?$")
  if not typename then return nil, nil end

  -- Find the first SoftBreak/LineBreak (end of marker+title line).
  local break_idx = nil
  for k = 2, #inlines do
    if inlines[k].t == "SoftBreak" or inlines[k].t == "LineBreak" then
      break_idx = k; break
    end
  end
  local rest = {}
  local start = break_idx and (break_idx + 1) or (#inlines + 1)
  for k = start, #inlines do rest[#rest + 1] = inlines[k] end
  return typename:lower(), rest
end

function BlockQuote(elem)
  local first = elem.content[1]
  if first == nil or first.t ~= "Para" then
    -- Non-para first block (list, code, etc.) inside a blockquote is treated
    -- as researcher commentary unless it starts with a marker — which can't
    -- happen unless the first block is a Para.
    return pandoc.RawBlock("latex", "% researcher-commentary elided")
  end

  local typename, rest_inlines = split_marker(first.content)
  if typename == nil then
    return pandoc.RawBlock("latex", "% researcher-commentary elided")
  end

  local strategy = STRATEGY[typename] or "quote"
  if strategy == "elide" then
    return pandoc.RawBlock("latex", "% callout elided: " .. typename)
  end

  -- Reassemble body: rest_inlines as a Para (if non-empty), then all blocks
  -- after the first.
  local body = {}
  if #rest_inlines > 0 then
    body[#body + 1] = pandoc.Para(rest_inlines)
  end
  for i = 2, #elem.content do body[#body + 1] = elem.content[i] end

  if strategy == "prose" then
    return body
  end

  local env = (strategy == "abstract") and "abstract" or "quote"
  local out = { pandoc.RawBlock("latex", "\\begin{" .. env .. "}") }
  for _, b in ipairs(body) do out[#out + 1] = b end
  out[#out + 1] = pandoc.RawBlock("latex", "\\end{" .. env .. "}")
  return out
end
