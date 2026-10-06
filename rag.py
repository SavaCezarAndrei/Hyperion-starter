"""RAG over the HYPER-AI documents (index built by build_index.py).

Hybrid retrieval: vector search (nomic-embed-text) finds paraphrases, BM25 finds
exact terms like "DeviceNode" or "self-CHOP"; the two rankings are merged with
reciprocal rank fusion. Answers are drafted only from the retrieved chunks; the
question worker (workers.py) verifies them and falls back to the web.
"""

import logging
import re
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
from llm import embeddings, llm

log = logging.getLogger("hyperion.rag")

INDEX = Path(__file__).parent / "data" / "index.json"

TOP_K = 6  # chunks given to the model (~1800 tokens)
CANDIDATES = 15  # per retriever, before fusion
PER_SOURCE = 2  # max chunks from one document, so comparisons see both sides
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
    """Light suffix stripping, applied to both questions and chunks:
    controllers/Controller -> controll, coordinates/coordinator -> coordinat."""
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    for suffix in ("ing", "ed", "or", "er", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
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
        top, taken = [], {}
        for doc_id in sorted(scores, key=lambda doc_id: -scores[doc_id]):
            source = by_id[doc_id].metadata["source"]
            if taken.get(source, 0) < PER_SOURCE:
                top.append(doc_id)
                taken[source] = taken.get(source, 0) + 1
            if len(top) == self.k:
                break
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
        (
            "human",
            "Context from the HYPER-AI documents:\n\n{context}\n\nQuestion: {question}{feedback}",
        ),
    ]
)
answer_chain = answer_prompt | llm | StrOutputParser()

# Backup for when the model explains instead of answering with the NOT_COVERED marker.
NOT_COVERED_PHRASES = re.compile(
    r"(do not|don't|does not|doesn't) (cover|mention|contain)|not (mentioned|covered|"
    r"provided|specified|included) in|no information",
    re.I,
)


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


def evidence(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def sources(docs: list[Document]) -> list[str]:
    return list(dict.fromkeys(d.metadata["source"] for d in docs))[:3]


def not_covered(answer: str) -> bool:
    return prompts.NOT_COVERED in answer or bool(NOT_COVERED_PHRASES.search(answer))


async def draft(
    session: memory.Session, question: str, context: str, feedback: str = ""
) -> str:
    """One answer drafted only from `context`; `feedback` carries the verifier's objections."""
    return await answer_chain.ainvoke(
        {
            "context": context,
            "question": question,
            "feedback": f"\n\n{feedback}" if feedback else "",
            "history": memory.history(session),
        }
    )
