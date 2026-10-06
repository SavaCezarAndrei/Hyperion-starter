"""Web fallback for on-topic questions the HYPER-AI documents do not cover.

Only reached after the router accepted the question, so off-topic requests are
still refused. The answer is built from search snippets only and is always
labelled as not coming from the documentation.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

from ddgs import DDGS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import prompts
from llm import llm

log = logging.getLogger("hyperion.web")

# Free search endpoints rate-limit and often return nothing: try several. Google
# blocks most automated queries but is tried first.
BACKENDS = ("google", "duckduckgo", "bing", "auto")
MAX_RESULTS = 5
TIMEOUT = 10  # seconds per backend
# "Hyper AI" is also an unrelated product name; anchor searches to the EU project.
PROJECT_HINT = "EU Horizon project computing continuum"


def web_query(question: str) -> str:
    query = question.strip().rstrip("?")
    if "hyper" in query.lower():
        query += f" {PROJECT_HINT}"
    return query

web_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.SYSTEM + "\n\n" + prompts.WEB_INSTRUCTIONS),
        ("human", "Web search results:\n\n{results}\n\nQuestion: {question}"),
    ]
)
web_chain = web_prompt | llm | StrOutputParser()


def _search(query: str) -> list[dict]:
    for backend in BACKENDS:
        try:
            results = DDGS(timeout=TIMEOUT).text(query, max_results=MAX_RESULTS, backend=backend)
            if results:
                log.info("web search via %s: %d results for %r", backend, len(results), query)
                return results
        except Exception as exc:
            log.warning("web search via %s failed: %s", backend, exc)
    return []


async def search(query: str) -> list[dict]:
    return await asyncio.to_thread(_search, query)


async def answer(query: str, question: str) -> AsyncIterator[dict]:
    """Stream a red-marked answer from the web, or nothing if the search finds nothing."""
    results = await search(web_query(query))
    if not results:
        yield {"response": prompts.WEB_UNAVAILABLE}
        return
    snippets = "\n\n".join(
        f"[{i}] {r['title']}\n{r['href']}\n{r['body']}" for i, r in enumerate(results, 1)
    )
    yield {"response": prompts.WEB_HEADER}
    async for chunk in web_chain.astream({"results": snippets, "question": question}):
        yield {"response": chunk}
    links = "\n".join(f"  {r['href']}" for r in results[:3])
    yield {"response": f"\n\n🔴 Web sources (not verified by HYPER-AI):\n{links}"}
