"""The project knowledge graph (knowledge.py). Unit tests: no LLM, no IDE, no Redis.

    uv run pytest tests/test_knowledge.py -v
"""

import pytest

import knowledge
from conftest import loop

NATIVE = """
applicationProfile:
  metadata:
    name: "web-shop"
  specs:
    runtime:
      containerImage:
        uri: "docker.io/library/nginx"
        tag: "1.25"
    resources:
      cpu: "500m"
      memory: "1Gi"
    network:
      ports:
        - port: 80
        - port: 443
"""

DEVICE = """
apiVersion: hyper-ai.eu/v1
kind: Application
metadata:
  name: sensor-cache
spec:
  workload:
    kind: DockerImage
    dockerImage:
      image: redis:7
"""

OTHER_NATIVE = NATIVE.replace("web-shop", "api").replace("nginx", "python").replace("80", "8080")


@pytest.fixture(autouse=True)
def empty_graph(monkeypatch):
    monkeypatch.setattr(knowledge, "_files", {})
    monkeypatch.setattr(knowledge, "_answers", {})
    monkeypatch.setattr(knowledge, "_meta", {})
    monkeypatch.setattr(knowledge, "_redis", None)


def learn(path: str, content: str) -> None:
    loop.run_until_complete(knowledge.apply({"action": "create_file", "path": path, "content": content}))


def test_extracts_native_and_device_profiles():
    native = knowledge.extract("demo/web.yaml", NATIVE)
    assert native.kind == "native" and native.name == "web-shop"
    assert native.images == ["docker.io/library/nginx:1.25"]
    assert native.ports == ["80", "443"]
    assert native.resources == {"cpu": "500m", "memory": "1Gi"}
    device = knowledge.extract("cache.yaml", DEVICE)
    assert device.kind == "device" and device.name == "sensor-cache"
    assert device.images == ["redis:7"]


def test_broken_yaml_is_kept_as_a_plain_file():
    facts = knowledge.extract("bad.yaml", "a: [unclosed")
    assert facts.kind is None and knowledge.describe(facts) == "bad.yaml: not an application profile"


def test_actions_update_the_graph():
    learn("demo/web.yaml", NATIVE)
    learn("demo/api.yaml", OTHER_NATIVE)
    assert knowledge.known_files() == {"demo/web.yaml", "demo/api.yaml"}
    loop.run_until_complete(knowledge.apply({"action": "delete_file", "path": "demo/api.yaml"}))
    assert knowledge.known_files() == {"demo/web.yaml"}
    loop.run_until_complete(knowledge.apply({"action": "delete_folder", "path": "demo"}))
    assert knowledge.known_files() == set()


def test_graph_links_apps_through_shared_images_and_ports():
    learn("a.yaml", NATIVE)
    learn("b.yaml", NATIVE.replace("web-shop", "web-shop-2"))
    graph = knowledge.graph()
    image_edges = [e for e in graph["edges"] if e["rel"] == "uses_image"]
    assert {e["to"] for e in image_edges} == {"image:docker.io/library/nginx:1.25"}
    assert len(image_edges) == 2
    assert "Shared by several apps" in knowledge.overview()


def test_files_are_identified_by_app_name_or_unique_image():
    learn("demo/web.yaml", NATIVE)
    learn("demo/api.yaml", OTHER_NATIVE)
    assert knowledge.identifies("demo/web.yaml", "change the port of web-shop to 8081")
    assert knowledge.identifies("demo/web.yaml", "change the port of the nginx app")
    assert not knowledge.identifies("demo/api.yaml", "change the port of the nginx app")
    assert not knowledge.identifies("demo/web.yaml", "change the port to 8081")


def test_project_questions_use_the_graph_but_docs_questions_do_not():
    learn("demo/web.yaml", NATIVE)
    evidence, paths = knowledge.project_evidence("which of my apps expose port 443?")
    assert "port" in evidence and paths == ["demo/web.yaml"]
    assert knowledge.project_evidence("how much memory does web-shop request?")[1] == ["demo/web.yaml"]
    assert knowledge.project_evidence("how do I deploy nginx on HYPER-AI?") == ("", [])
    assert knowledge.project_evidence("What is a DeviceNode?") == ("", [])


def fake_vectors(monkeypatch, vectors: dict[str, list[float]]):
    async def embed(question: str) -> list[float]:
        return vectors[question]

    monkeypatch.setattr(knowledge.embeddings, "aembed_query", embed)


def test_learned_answers_are_reused_only_for_the_same_question(monkeypatch):
    fake_vectors(monkeypatch, {
        "What is a DeviceNode?": [1.0, 0.0],
        "what is a devicenode": [0.99, 0.05],
        "What is a NativeNode?": [0.98, 0.1],  # embeds closely, but other words
    })
    loop.run_until_complete(
        knowledge.remember("What is a DeviceNode?", "docs", "An edge device.", ["D4.2"])
    )
    same = loop.run_until_complete(knowledge.recall("what is a devicenode"))
    assert same and same.text == "An edge device." and same.sources == ["D4.2"]
    assert loop.run_until_complete(knowledge.recall("What is a NativeNode?")) is None
    assert knowledge.stats()["answers_reused"] == 1


def test_unverified_answers_are_never_learned(monkeypatch):
    fake_vectors(monkeypatch, {"q?": [1.0]})
    for kind in ("unverified", "none", "web_no_answer", "project", "file"):
        loop.run_until_complete(knowledge.remember("q?", kind, "text", []))
    assert knowledge.stats()["learned_answers"] == 0


def test_old_web_answers_expire(monkeypatch):
    fake_vectors(monkeypatch, {"Who won?": [1.0]})
    loop.run_until_complete(knowledge.remember("Who won?", "web", "Argentina [1]", ["https://x"]))
    knowledge._answers[knowledge._key("Who won?")].created -= knowledge.WEB_TTL + 1
    assert loop.run_until_complete(knowledge.recall("Who won?")) is None


def test_docs_answers_expire_when_the_index_is_rebuilt(monkeypatch):
    fake_vectors(monkeypatch, {"What is X?": [1.0]})
    loop.run_until_complete(knowledge.remember("What is X?", "docs", "X is Y.", ["D3.3"]))
    monkeypatch.setattr(knowledge, "INDEX_VERSION", "rebuilt")
    assert loop.run_until_complete(knowledge.recall("What is X?")) is None


def test_structured_project_questions_are_answered_in_code():
    learn("demo/web.yaml", NATIVE)
    learn("demo/api.yaml", OTHER_NATIVE)
    learn("cache.yaml", DEVICE)
    text, paths = knowledge.lookup("which of my apps use the nginx image?")
    assert paths == ["demo/web.yaml"] and "1 app(s)" in text
    assert knowledge.lookup("which of my apps expose port 8080?")[1] == ["demo/api.yaml"]
    assert knowledge.lookup("which of my apps are device apps?")[1] == ["cache.yaml"]
    text, paths = knowledge.lookup("which of my apps use port 9999?")
    assert paths == [] and "None of the 3 apps" in text
    assert knowledge.lookup("how do I deploy nginx?") is None  # a documentation question
