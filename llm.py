"""Shared clients for the legion1 LLM server (OpenAI-compatible)."""

import os

import httpx
import openai
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

load_dotenv()

API_KEY = os.environ.get("API_KEY", "")
# The OpenAI client refuses to even be created without a key, which would crash the
# server at startup; main.py answers "not configured" instead when API_KEY is empty.
CLIENT_KEY = API_KEY or "not-set"
BASE_URL = "https://legion1.di.uoa.gr/v1"
MODEL = "llama3.1"  # Llama 3.1 8B, 8192-token context
# The shared server is sometimes slow. Calls are NOT cut short: the user sees live
# progress meanwhile (progress.py), and main.SAFETY_LIMIT only ends a dead reply.
# A failed connection is retried once.
TIMEOUT = httpx.Timeout(600, connect=15)
MAX_RETRIES = 1

# Free-text answers (streamed to the user).
llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=CLIENT_KEY,
    max_completion_tokens=2048,
    temperature=0.2,
    timeout=TIMEOUT,
    max_retries=MAX_RETRIES,
)

# Structured output for the router and the planner: deterministic JSON.
llm_json = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=CLIENT_KEY,
    max_completion_tokens=512,
    temperature=0,
    timeout=TIMEOUT,
    max_retries=MAX_RETRIES,
).bind(response_format={"type": "json_object"})


def is_fatal(exc: BaseException) -> bool:
    """Errors that retrying cannot fix: the server rejects our key."""
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in (401, 403)


class LegionEmbeddings(Embeddings):
    """nomic-embed-text on legion1.

    Not OpenAIEmbeddings: the OpenAI SDK always sends `encoding_format`, which
    legion1 rejects. nomic also expects task prefixes on every text.
    """

    model = "nomic-embed-text"  # 768 dimensions
    batch_size = 32
    timeout = 60

    def _payload(self, texts: list[str]) -> dict:
        return {"model": self.model, "input": texts}

    @staticmethod
    def _vectors(response: httpx.Response) -> list[list[float]]:
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        with httpx.Client(timeout=self.timeout) as client:
            for i in range(0, len(texts), self.batch_size):
                response = client.post(
                    f"{BASE_URL}/embeddings",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json=self._payload(texts[i : i + self.batch_size]),
                )
                vectors += self._vectors(response)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"search_query: {text}"])[0]

    async def aembed_query(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=self._payload([f"search_query: {text}"]),
            )
        return self._vectors(response)[0]


embeddings = LegionEmbeddings()
