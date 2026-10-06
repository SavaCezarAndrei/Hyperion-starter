"""Shared clients for the legion1 LLM server (OpenAI-compatible)."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = "https://legion1.di.uoa.gr/v1"
MODEL = "llama3.1"  # Llama 3.1 8B, 8192-token context

# Free-text answers (streamed to the user).
llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    max_completion_tokens=2048,
    temperature=0.2,
)

# Structured output for the router and the planner: the model must return JSON.
llm_json = llm.bind(response_format={"type": "json_object"})
