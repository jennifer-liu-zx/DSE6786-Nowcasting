"""Shared helpers for scripted, targeted edits to Technical Documentation.docx."""


def find_paragraph_index(doc, substring: str) -> int:
    matches = [i for i, p in enumerate(doc.paragraphs) if substring in p.text]
    if not matches:
        raise ValueError(f"No paragraph contains: {substring!r}")
    if len(matches) > 1:
        raise ValueError(f"Substring is ambiguous, found in paragraphs {matches}: {substring!r}")
    return matches[0]


def replace_paragraph_text(doc, para_index: int, old: str, new: str) -> None:
    """Replace `old` with `new` within a single paragraph's text, preserving
    the paragraph's style but not per-run formatting (acceptable for these
    plain-body-text corrections)."""
    para = doc.paragraphs[para_index]
    if old not in para.text:
        raise ValueError(f"Paragraph {para_index} does not contain: {old!r}")
    new_text = para.text.replace(old, new)
    for run in list(para.runs):
        run.text = ""
    if para.runs:
        para.runs[0].text = new_text
    else:
        para.add_run(new_text)


def remove_paragraph(doc, para_index: int) -> None:
    para = doc.paragraphs[para_index]
    para._element.getparent().remove(para._element)
