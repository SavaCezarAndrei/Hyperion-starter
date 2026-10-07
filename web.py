"""Web fallback for questions the HYPER-AI documents do not cover.

Only reached after the router accepted the question, so off-topic requests are
still refused - except through /btw, which may ask about anything. Evidence = search snippets + the most relevant paragraphs of the
top result pages. Answers are drafted only from that evidence, verified by the
question worker, and always labelled as not coming from the documentation.
"""

import asyncio
import logging
from dataclasses import dataclass, field

import httpx
from ddgs import DDGS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import prompts
import rag
from html_text import html_to_markdown
from llm import llm

log = logging.getLogger("hyperion.web")

# Free search endpoints rate-limit and often return nothing: try several. Google
# blocks most automated queries but is tried first.
BACKENDS = ("google", "duckduckgo", "bing", "auto")
MAX_RESULTS = 5
SEARCH_TIMEOUT = 10  # seconds per backend
PAGES_TO_READ = 2  # full pages fetched from the top results
PAGE_TIMEOUT = 8
MAX_LINK_CHARS = 200
PASSAGE_CHARS = 1500  # relevant text kept per page (the model has 8192 tokens)
# "Hyper AI" is also an unrelated product name; anchor searches to the EU project.
PROJECT_HINT = "EU Horizon project computing continuum"

web_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.SYSTEM + "\n\n" + prompts.WEB_INSTRUCTIONS),
        ("human", "Web search results:\n\n{evidence}\n\nQuestion: {question}{feedback}"),
    ]
)
web_chain = web_prompt | llm | StrOutputParser()

# /btw may be about anything, so its web answers are not limited to the EU project.
general_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.BTW_WEB_INSTRUCTIONS),
        ("human", "Web search results:\n\n{evidence}\n\nQuestion: {question}{feedback}"),
    ]
)
general_chain = general_prompt | llm | StrOutputParser()


@dataclass
class WebEvidence:
    text: str = ""  # numbered results, given to the writer and the verifier
    links: list[str] = field(default_factory=list)


def web_query(question: str) -> str:
    query = question.strip().rstrip("?")
    if "hyper" in query.lower():
        query += f" {PROJECT_HINT}"
    return query


def _search(query: str) -> list[dict]:
    for backend in BACKENDS:
        try:
            results = DDGS(timeout=SEARCH_TIMEOUT).text(
                query, max_results=MAX_RESULTS, backend=backend
            )
            if results:
                log.info("web search via %s: %d results for %r", backend, len(results), query)
                return results
        except Exception as exc:
            log.warning("web search via %s failed: %s", backend, exc)
    return []


def relevant_passages(page_text: str, question: str) -> str:
    """The paragraphs of a page that share the most words with the question."""
    wanted = set(rag.tokenize(question))
    paragraphs = [p.strip() for p in page_text.split("\n\n") if len(p.strip()) > 40]
    ranked = sorted(paragraphs, key=lambda p: -len(wanted & set(rag.tokenize(p))))
    kept, size = [], 0
    for p in ranked:
        if size + len(p) > PASSAGE_CHARS:
            break
        kept.append(p)
        size += len(p)
    return "\n".join(kept)


async def read_page(client: httpx.AsyncClient, url: str, question: str) -> str:
    try:
        response = await client.get(url)
        if "html" not in response.headers.get("content-type", ""):
            return ""
        return relevant_passages(html_to_markdown(response.text, "body"), question)
    except Exception as exc:
        log.info("could not read %s: %s", url, exc)
        return ""


async def gather(question: str) -> WebEvidence:
    results = await asyncio.to_thread(_search, web_query(question))
    if not results:
        return WebEvidence()
    async with httpx.AsyncClient(
        timeout=PAGE_TIMEOUT, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}
    ) as client:
        pages = await asyncio.gather(
            *(read_page(client, r["href"], question) for r in results[:PAGES_TO_READ])
        )
    blocks = []
    for i, r in enumerate(results, 1):
        block = f"[{i}] {r['title']}\n{r['href']}\n{r['body']}"
        if i <= len(pages) and pages[i - 1]:
            block += f"\nFrom the page:\n{pages[i - 1]}"
        blocks.append(block)
    # Ad and tracking redirects (bing.com/aclick?..., hundreds of characters) are not
    # sources a user can check; show real page links only.
    links = [r["href"] for r in results if len(r["href"]) <= MAX_LINK_CHARS and "/aclick?" not in r["href"]]
    return WebEvidence("\n\n".join(blocks), links[:3])


async def draft(question: str, evidence: str, feedback: str = "", general: bool = False) -> str:
    return await (general_chain if general else web_chain).ainvoke(
        {
            "evidence": evidence,
            "question": question,
            "feedback": f"\n\n{feedback}" if feedback else "",
        }
    )
