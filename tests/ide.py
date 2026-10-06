"""Stand-in for the IDE frontend in tests: applies streamed actions to the backend.

In real use the browser executes our actions. This replays them with the same
backend calls and the same name matching as the frontend's code (ide-gui bundle:
createFile -> POST /file/create, saveFile -> POST /file, deleteEntry -> DELETE /delete,
names resolved by exact path or a unique trailing name).
"""

import os

import httpx

BACKEND = os.environ.get("IDE_BACKEND_URL", "http://localhost:3001/api")


def entries(kind: str) -> list[str]:
    found, pending = [], [""]
    with httpx.Client(timeout=10) as client:
        while pending:
            for item in client.get(f"{BACKEND}/files", params={"path": pending.pop()}).json():
                if item["type"] == "folder":
                    pending.append(item["path"])
                if item["type"] == kind:
                    found.append(item["path"])
    return found


def resolve(name: str, kind: str) -> str | None:
    paths = entries(kind)
    if name in paths:
        return name
    matches = [p for p in paths if p.endswith("/" + name)]
    return matches[0] if len(matches) == 1 else None


def apply(action: dict) -> None:
    op, path = action["action"], action.get("path", "")
    with httpx.Client(timeout=10) as client:
        if op == "create_folder":
            client.post(f"{BACKEND}/folder/create", json={"path": path})
        elif op == "create_file":
            client.post(f"{BACKEND}/file/create", json={"path": path, "content": action["content"]})
        elif op == "edit_file" and (found := resolve(path, "file")):
            client.post(f"{BACKEND}/file", json={"path": found, "content": action["content"]})
        elif op in ("delete_file", "delete_folder"):
            if found := resolve(path, "file" if op == "delete_file" else "folder"):
                client.delete(f"{BACKEND}/delete", params={"path": found})


def read(path: str) -> str | None:
    response = httpx.get(f"{BACKEND}/agent/file", params={"path": path}, timeout=10)
    return response.json().get("content") if response.status_code == 200 else None


def delete(path: str) -> None:
    httpx.delete(f"{BACKEND}/delete", params={"path": path}, timeout=10)
