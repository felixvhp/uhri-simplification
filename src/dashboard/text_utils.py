"""
Pkg 3: Shared text-formatting helpers [last updated 2026-09-13]

Usage:
    from src.dashboard.text_utils import slugify, format_legal_bullets

slugify() turns a metric or prompt name typed into the GUI into its key in
the settings files. format_legal_bullets() is used by the Streamlit GUI's
"auto-format original text" toggle (app.py)
so the raw legalese displays with one lettered clause per line, closer to how
the simplified output usually already reads.

Open Issues:
    - format_legal_bullets() only recognises single-letter "(a)"-style
      markers, so nested sub-points like "(a)(i)" stay attached to their
      parent clause rather than getting their own line too.

Resumption:
    - If this grows past a couple of helpers, keep it as the one place for
      dashboard string/formatting utilities rather than scattering them.
"""

import re


def slugify(name: str) -> str:
    """Turn a free-text name into a lowercase, underscore-separated slug.

    Used anywhere a user-entered name needs to become a safe identifier —
    e.g. a new metric's column prefix, or a new prompt's `prompt` value.

    Args:
        name: Human-entered name, e.g. "Tone: formal enough?".

    Returns:
        Slug, e.g. "tone_formal_enough".

    Raises:
        ValueError: If `name` has no alphanumeric characters to slugify.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"'{name}' has no usable characters for a slug")
    return slug


# Matches the position right before a "(a) ", "(b) ", ... marker — a
# zero-width lookahead, so splitting on it keeps the marker attached to the
# clause that follows rather than the text before it.
_LETTER_MARKER = re.compile(r"(?=\([a-z]\)\s)")


def format_legal_bullets(text):
    """Put each lettered "(a)/(b)/(c)..." clause in legal text on its own line.

    Source text in this dataset runs consecutive lettered sub-clauses
    together with only a single newline or space between them — e.g.
    "...the State Party: (a) Do X; (b) Do Y; (c) Do Z." A single newline
    renders as just a space in Markdown, so as-is it reads as one dense
    paragraph, while the LLM-simplified version of the same text usually
    already breaks each point onto its own line. This closes that gap so
    the two are easier to compare side by side.

    Requires at least two markers to do anything — a single "(a)" more
    often turns out to be a stray parenthetical (e.g. a citation like
    "(a) 2019 report") than the start of a lettered list, and splitting on
    just one would add a line break for no real reason.

    Args:
        text: Raw text, e.g. an ``original_text`` cell. Returned unchanged
            if it isn't a string (a stray NaN from pandas, say).

    Returns:
        Text with a blank line before each marker, so it renders in
        Markdown as one paragraph per clause. Unchanged if fewer than two
        markers are found.
    """
    if not isinstance(text, str):
        return text

    parts = [p.strip() for p in _LETTER_MARKER.split(text) if p.strip()]
    marker_count = sum(1 for p in parts if p.startswith("("))
    if marker_count < 2:
        return text
    return "\n\n".join(parts)
