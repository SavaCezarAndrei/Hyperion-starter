"""Verifier and worker anti-hallucination checks (calls the LLM, no Docker needed).

    uv run pytest tests/test_verifier.py -v
"""

import memory
import orchestrator
import verifier
import workers

from conftest import loop  # the one shared event loop, see conftest.py

EVIDENCE = (
    "[D4.2 Section 3.1 High Level Architecture]\nThere are 2 main discrete modules that "
    "form the connectors, the Device Controller and the Application Controller. Both "
    "controllers communicate directly with the Kubernetes control plane."
)
QUESTION = "Which controllers make up the Open Connectors?"


def check(answer: str) -> verifier.Verdict:
    return loop.run_until_complete(verifier.check_answer(QUESTION, EVIDENCE, answer))


def test_accepts_a_correct_answer():
    assert check("The Device Controller and the Application Controller.").ok


def test_accepts_a_paraphrase():
    assert check(
        "Two modules: a Device Controller and an Application Controller, "
        "both talking to the Kubernetes control plane."
    ).ok


def test_catches_invented_facts():
    verdict = check(
        "Three controllers: the Device, Application and Network Controller, written in Go."
    )
    assert not verdict.ok and verdict.problems


def test_catches_an_answer_to_another_question():
    assert not check("Kubernetes is an open-source container orchestration system.").ok


def test_code_checks_reject_empty_answer():
    assert not verifier.code_checks("  ").ok


def test_remove_flagged_keeps_the_verified_rest():
    answer = (
        "The Open Connectors have two controllers. They were written in Go by Google "
        "in 2019. Both talk to the Kubernetes control plane."
    )
    kept = workers.remove_flagged(answer, ["written in Go by Google in 2019"])
    assert "Go" not in kept and "two controllers" in kept and "control plane" in kept


def test_planner_splits_compound_questions():
    plan = loop.run_until_complete(
        orchestrator.plan_questions("What is a DeviceNode and who coordinates HYPER-AI?")
    )
    assert len(plan) == 2


def test_planner_keeps_single_questions():
    for text in (
        "What is a DeviceNode?",
        "What is the difference between native and device applications?",
    ):
        assert len(loop.run_until_complete(orchestrator.plan_questions(text))) == 1


def test_worker_answers_from_docs():
    result = loop.run_until_complete(
        workers.question_worker(memory.Session(user_id="t"), "Who coordinates the HYPER-AI project?")
    )
    assert result.kind == "docs" and "CERTH" in result.text
