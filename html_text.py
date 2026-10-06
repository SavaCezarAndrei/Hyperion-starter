"""HTML page -> Markdown text. Used by fetch_docs.py and by the web fallback."""

import html
import re


def cell(raw: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", "", raw))
    return re.sub(r"\s+", " ", text).strip().replace("|", "/")


def table(match: re.Match) -> str:
    rows = re.findall(r"(?s)<tr[^>]*>(.*?)</tr>", match.group(0))
    lines = []
    for i, row in enumerate(rows):
        cells = [cell(c) for c in re.findall(r"(?s)<t[dh][^>]*>(.*?)</t[dh]>", row)]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "---|" * len(cells))
    return "\n\n" + "\n".join(lines) + "\n\n"


GENERIC_IMAGES = {"home", "eu-flag-white", "image", "logo"}


def image_text(match: re.Match) -> str:
    """Logos carry information (e.g. the partner list): keep their alt text."""
    alt = re.search(r'alt="([^"]*)"', match.group(0))
    name = re.sub(r"\s*\blogo\b\s*", " ", html.unescape(alt.group(1)) if alt else "", flags=re.I)
    name = name.strip()
    return f"\n- {name}\n" if name and name.lower() not in GENERIC_IMAGES else ""


def html_to_markdown(page: str, container: str) -> str:
    """Main content of a page as Markdown: headings, lists, tables, code blocks."""
    page = re.sub(r"(?s)<(script|style|nav|header|footer|form|noscript)\b.*?</\1>", "", page)
    main = re.search(rf"(?s)<{container}\b.*?</{container}>", page)
    page = main.group(0) if main else page
    page = re.sub(r"<img\b[^>]*>", image_text, page)
    page = re.sub(r'(?s)<a[^>]*class="headerlink"[^>]*>.*?</a>', "", page)
    page = re.sub(r"(?s)<table.*?</table>", table, page)
    for level in range(1, 5):
        page = re.sub(rf"<h{level}[^>]*>", "\n\n" + "#" * level + " ", page)
    page = re.sub(r"<pre[^>]*>", "\n```\n", page).replace("</pre>", "\n```\n")
    page = re.sub(r"<li[^>]*>", "\n- ", page)
    page = re.sub(r"<br\s*/?>|</(p|h\d|div|section)>", "\n", page)
    text = html.unescape(re.sub(r"<[^>]+>", "", page))
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    # Website quirks: the coordinator label follows its logo; decorative "//" headings.
    text = re.sub(r"- (.+)\n\nCoordinator\n", r"- \1 (project coordinator)\n", text)
    return re.sub(r"\n## //.*\n", "\n", text)
