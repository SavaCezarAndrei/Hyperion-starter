"""End-to-end tests against a running agent (docker compose up --build -d first).

    uv run pytest tests -v

One test per grading criterion. Tests for phases not built yet are marked xfail:
when a phase lands, delete its marker and the test must pass.
"""

import uuid

import pytest

from client import send

todo = pytest.mark.xfail(reason="not implemented yet", strict=False)


def new_user() -> str:
    return str(uuid.uuid4())


# --- Phase 1: streaming + actions -------------------------------------------


def test_streams_a_reply():
    text, actions = send(new_user(), "hello")
    assert text.strip()
    assert actions == []


def test_action_event_format():
    _, actions = send(new_user(), "/test-action")
    assert actions == [{"action": "create_folder", "path": "hyperion-test"}]


# --- Phase 2: memory ---------------------------------------------------------


def test_memory_within_session():
    user = new_user()
    send(user, "My name is Zorblax. Just reply ok.")
    text, _ = send(user, "What is my name?")
    assert "zorblax" in text.lower()


def test_sessions_are_separate():
    send(new_user(), "My name is Zorblax. Just reply ok.")
    text, _ = send(new_user(), "What is my name?")
    assert "zorblax" not in text.lower()


# --- Phase 3: guardrails -----------------------------------------------------


def test_rejects_off_topic():
    text, actions = send(new_user(), "What is the weather today?")
    assert "only help with HYPER-AI" in text
    assert actions == []
    assert "🔴" not in text  # off-topic is refused, never searched on the web


def test_chitchat_is_not_refused():
    text, _ = send(new_user(), "what is your name?")
    assert "only help with HYPER-AI" not in text


def test_valid_request_is_not_refused():
    text, _ = send(new_user(), "Create a deployment YAML for nginx")
    assert "only help with HYPER-AI" not in text


# --- Phase 4: RAG ------------------------------------------------------------


def test_answers_about_hyperai_with_sources():
    text, _ = send(new_user(), "What is HyperAI?")
    assert any(w in text.lower() for w in ("continuum", "distributed", "edge", "cloud"))
    assert "Sources (HYPER-AI documentation)" in text
    assert "🔴" not in text  # documented answers are never marked


def test_answer_comes_from_the_docs():
    # Only in D4.2 §3.1 — the bare model cannot know this.
    text, _ = send(new_user(), "Which two controllers make up the HYPER-AI Open Connectors?")
    assert "device controller" in text.lower()
    assert "application controller" in text.lower()


def test_marks_web_answers_when_docs_lack_it():
    # On-topic, but no document covers it: web answer, clearly marked red.
    text, _ = send(new_user(), "What was the total budget of the HYPER-AI project in euros?")
    assert "not cover" in text
    assert "Sources (HYPER-AI documentation)" not in text
    if "returned no results" in text:  # free search engines are rate-limited at times
        pytest.skip("web search unavailable right now")
    assert "🔴 NOT FROM THE HYPER-AI DOCUMENTATION 🔴" in text
    assert "http" in text  # web sources listed


def test_rag_follow_up_uses_memory():
    user = new_user()
    send(user, "What are the Open Connectors?")
    text, _ = send(user, "which controllers do they have?")
    assert "device controller" in text.lower()


# --- Phase 5: IDE actions ----------------------------------------------------


@todo
def test_creates_nginx_yaml():
    _, actions = send(
        new_user(), "Create a deployment YAML for a service using the nginx Docker image"
    )
    files = [a for a in actions if a["action"] == "create_file"]
    assert files and files[0]["path"].endswith((".yaml", ".yml"))
    assert "nginx" in files[0]["content"]


# --- Phase 6: human in the loop ----------------------------------------------


@todo
def test_delete_needs_confirmation():
    user = new_user()
    _, actions = send(user, "delete nginx.yaml")
    assert not any(a["action"] == "delete_file" for a in actions)
    _, actions = send(user, "yes")
    assert any(a["action"] == "delete_file" for a in actions)
