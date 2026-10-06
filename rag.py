"""RAG over the HYPER-AI documents (index built by build_index.py).

Hybrid retrieval: vector search (nomic-embed-text) finds paraphrases, BM25 finds
exact terms like "DeviceNode" or "self-CHOP"; the two rankings are merged with
reciprocal rank fusion. The answer is generated only from the retrieved chunks.
"""

import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from rank_bm25 import BM25Okapi

import memory
import prompts
import web
from llm import embeddings, llm

log = logging.getLogger("hyperion.rag")

INDEX = Path(__file__).parent / "data" / "index.json"

TOP_K = 5  # chunks given to the model
CANDIDATES = 10  # per retriever, before fusion
RRF_K = 60  # standard reciprocal-rank-fusion constant
# Below this cosine similarity the question is not covered by the documents
MIN_SIMILARITY = 0.58  # measured: documented questions 0.60-0.86, off-topic 0.47-0.61
# Follow-ups shorter than this are searched together with the previous question.
SHORT_FOLLOW_UP = 8


# Question words carry no meaning for keyword search and match every chunk.
STOPWORDS = set(
    "a an and are as at be by can do doe does for from how i in is it its me my "
    "of on or the that this to was what when where which who why will with you your "
    "tell explain about there their they them".split()
)


def stem(word: str) -> str:
    """Light plural stemming so "controllers" matches "Controller"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


class HybridRetriever(BaseRetriever):
    """Vector + BM25 retrieval fused by rank. Puts the best vector score in metadata."""

    store: InMemoryVectorStore
    documents: list[Document]
    bm25: BM25Okapi
    k: int = TOP_K

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_store(cls, store: InMemoryVectorStore) -> "HybridRetriever":
        documents = [
            Document(id=doc_id, page_content=d["text"], metadata=d["metadata"])
            for doc_id, d in store.store.items()
        ]
        bm25 = BM25Okapi([tokenize(d.page_content) for d in documents])
        return cls(store=store, documents=documents, bm25=bm25)

    def _fuse(self, query: str, vector_hits: list[tuple[Document, float]]) -> list[Document]:
        scores: dict[str, float] = {}
        similarity = {doc.id: sim for doc, sim in vector_hits}
        for rank, (doc, _) in enumerate(vector_hits):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (RRF_K + rank)
        keyword = self.bm25.get_scores(tokenize(query))
        best = sorted(range(len(self.documents)), key=lambda i: -keyword[i])[:CANDIDATES]
        for rank, i in enumerate(b for b in best if keyword[b] > 0):
            doc_id = self.documents[i].id
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)
        by_id = {d.id: d for d in self.documents}
        top = sorted(scores, key=lambda doc_id: -scores[doc_id])[: self.k]
        return [
            Document(
                id=doc_id,
                page_content=by_id[doc_id].page_content,
                metadata={**by_id[doc_id].metadata, "similarity": similarity.get(doc_id, 0.0)},
            )
            for doc_id in top
        ]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = self.store.similarity_search_with_score(query, k=CANDIDATES)
        return self._fuse(query, hits)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = await self.store.asimilarity_search_with_score(query, k=CANDIDATES)
        return self._fuse(query, hits)


def load_retriever() -> HybridRetriever | None:
    if not INDEX.exists():
        log.error("no RAG index at %s - run build_index.py", INDEX)
        return None
    store = InMemoryVectorStore.load(str(INDEX), embedding=embeddings)
    log.info("RAG index loaded: %d chunks", len(store.store))
    return HybridRetriever.from_store(store)


retriever = load_retriever()

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts.SYSTEM + "\n\n" + prompts.RAG_INSTRUCTIONS),
        MessagesPlaceholder("history"),
        ("human", "Context from the HYPER-AI documents:\n\n{context}\n\nQuestion: {question}"),
    ]
)
answer_chain = answer_prompt | llm | StrOutputParser()


def search_query(session: memory.Session, text: str) -> str:
    """Short follow-ups ("and how does it register them?") need the previous question."""
    previous = [m["content"] for m in session.messages if m["role"] == "user"]
    if previous and len(text.split()) < SHORT_FOLLOW_UP:
        return f"{previous[-1]} {text}"
    return text


async def retrieve(session: memory.Session, text: str) -> list[Document]:
    """Relevant chunks, or [] when the documents do not cover the question."""
    if retriever is None:
        return []
    docs = await retriever.ainvoke(search_query(session, text))
    best = max((d.metadata["similarity"] for d in docs), default=0.0)
    log.info("retrieved %d chunks, best similarity %.3f for %r", len(docs), best, text)
    return docs if best >= MIN_SIMILARITY else []


# Characters held back before streaming, so the NOT_COVERED marker never reaches the user.
HOLD = 20
# Backup for when the model explains instead of answering with the marker.
NOT_COVERED_PHRASES = re.compile(
    r"(do not|don't|does not|doesn't) (cover|mention|contain)|not (mentioned|covered|"
    r"provided|specified|included) in|no information",
    re.I,
)


def cite(docs: list[Document]) -> str:
    sources = list(dict.fromkeys(d.metadata["source"] for d in docs))
    return "\n\nSources (HYPER-AI documentation): " + "; ".join(sources[:3])


async def answer(session: memory.Session, text: str) -> AsyncIterator[dict]:
    query = search_query(session, text)
    docs = await retrieve(session, text)
    covered = False
    if docs:
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        written = ""
        async for chunk in answer_chain.astream(
            {"context": context, "question": text, "history": memory.history(session)}
        ):
            written += chunk
            if prompts.NOT_COVERED in written:
                break
            if len(written) >= HOLD:
                yield {"response": written if len(written) - len(chunk) < HOLD else chunk}
        else:
            if written and len(written) < HOLD:  # short answer, never released above
                yield {"response": written}
            covered = bool(written.strip()) and not NOT_COVERED_PHRASES.search(written)
        if covered:
            yield {"response": cite(docs)}
            return
    if not docs or prompts.NOT_COVERED in written:
        yield {"response": prompts.NOT_IN_DOCS}
    # The documents fall short: fall back to the web, clearly marked.
    async for event in web.answer(query, text):
        yield event
