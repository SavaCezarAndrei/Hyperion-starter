"""Build the RAG index from data/docs/ into data/index.json.

    uv run build_index.py

Run it locally whenever data/docs/ changes and commit data/index.json: the image
ships the finished index, so the container never embeds documents at startup.
"""

import re
from pathlib import Path

import docx2txt
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from llm import embeddings

DOCS = Path("data/docs")
INDEX = Path("data/index.json")

CHUNK_SIZE = 1200  # characters, ~300 tokens: 5 chunks fit easily in the 8192 context
CHUNK_OVERLAP = 150

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def docx_title(path: Path) -> str:
    """D4.2_3.1_High_Level_Architecture.docx -> 'D4.2 Section 3.1 High Level Architecture'."""
    deliverable, *rest = path.stem.split("_")
    if rest and re.fullmatch(r"\d+(\.\d+)*", rest[0]):
        rest[0] = "Section " + rest[0]
    return " ".join([deliverable, *rest])


def join_lines(text: str) -> str:
    """The deliverables store every visual line as its own paragraph; rebuild them.

    A line continues the previous one unless the previous one ends a sentence or
    is a short heading-like line.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    paragraphs: list[str] = []
    for line in lines:
        prev = paragraphs[-1] if paragraphs else ""
        if prev and not prev.endswith((".", ":", "!", "?")) and len(prev.split("\n")[-1]) > 60:
            paragraphs[-1] = f"{prev} {line}"
        else:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def load_docx(path: Path) -> list[Document]:
    text = join_lines(docx2txt.process(str(path)))
    return [Document(text, metadata={"source": docx_title(path), "section": ""})]


SITES = {"tutorial": "IDE Tutorial", "website": "HYPER-AI website"}


def markdown_source(path: Path, text: str) -> str:
    """tutorial_cookbook.md -> 'IDE Tutorial: Cookbook & Examples' (its # title),
    website_use_case_healthcare.md -> 'HYPER-AI website: Use Case Healthcare'."""
    site, _, page = path.stem.partition("_")
    title = re.search(r"^# (.+)$", text, re.M)
    name = title.group(1).strip() if title else page.replace("_", " ").title()
    return f"{SITES.get(site, site)}: {name}"


def load_markdown(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    source = markdown_source(path, text)
    sections = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False).split_text(text)
    return [
        Document(
            s.page_content,
            metadata={
                "source": source,
                "section": " > ".join(s.metadata[h] for _, h in HEADERS[1:] if h in s.metadata),
            },
        )
        for s in sections
    ]


def main() -> None:
    documents = []
    for path in sorted(DOCS.iterdir()):
        if path.suffix == ".docx":
            documents += load_docx(path)
        elif path.suffix == ".md":
            documents += load_markdown(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        # The heading travels with the text: it helps both search and citations.
        where = " > ".join(filter(None, [chunk.metadata["source"], chunk.metadata["section"]]))
        chunk.page_content = f"[{where}]\n{chunk.page_content}"

    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    INDEX.parent.mkdir(exist_ok=True)
    store.dump(str(INDEX))

    sources = sorted({c.metadata["source"] for c in chunks})
    print(f"{len(chunks)} chunks from {len(sources)} sources -> {INDEX}")
    for s in sources:
        print(f"  {sum(c.metadata['source'] == s for c in chunks):3d}  {s}")


if __name__ == "__main__":
    main()
