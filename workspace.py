"""The user's IDE workspace, as seen through the ide-backend.

Facts measured against the backend (Phase 5, step 0):
- create_file creates missing parent folders itself;
- creating a file or folder that already exists fails (so overwriting = edit_file);
- the IDE runs the actions we stream concurrently, not one after another;
- `GET /files?path=` lists one level only;
- validate_file only works on a file that exists - hence `validate_content`, which
  checks a draft through a scratch file before the user ever sees it.
"""

import logging
import posixpath
import re
import uuid
from dataclasses import dataclass, field

import httpx

from helpers import IDE_BACKEND_URL, ReadFileError, ValidateFileError, read_file, validate_file

log = logging.getLogger("hyperion.workspace")

TIMEOUT = 10
SCRATCH_PREFIX = ".hyperion-check-"  # never shown to the user; deleted right after use
YAML_SUFFIXES = (".yaml", ".yml")


class PathError(ValueError):
    """A path the agent must not touch; the message can be shown to the user."""


@dataclass
class Entries:
    files: set[str] = field(default_factory=set)
    folders: set[str] = field(default_factory=set)


async def list_entries() -> Entries:
    """Every file and folder in the workspace (walks the tree level by level)."""
    entries = Entries()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        pending = [""]
        while pending:
            folder = pending.pop()
            response = await client.get(f"{IDE_BACKEND_URL}/files", params={"path": folder})
            response.raise_for_status()
            for item in response.json():
                if item["path"].split("/")[-1].startswith(SCRATCH_PREFIX):
                    continue
                if item["type"] == "folder":
                    entries.folders.add(item["path"])
                    pending.append(item["path"])
                else:
                    entries.files.add(item["path"])
    return entries


def clean_path(raw: str, *, is_file: bool) -> str:
    """A safe path relative to the workspace root, or PathError."""
    path = str(raw or "").strip().strip("`'\"").replace("\\", "/")
    if re.match(r"^[a-zA-Z]:/", path) or path.startswith("~"):
        raise PathError(f"'{raw}' is an absolute path - paths must be inside the workspace")
    path = posixpath.normpath("/" + path).lstrip("/")  # resolves "a/./b", "a//b"
    if not path or path == ".":
        raise PathError("the path is empty")
    if ".." in str(raw).replace("\\", "/").split("/"):
        raise PathError(f"'{raw}' leaves the workspace ('..' is not allowed)")
    if any(part.startswith(SCRATCH_PREFIX) for part in path.split("/")):
        raise PathError(f"'{raw}' is reserved")
    if is_file and not path.lower().endswith(YAML_SUFFIXES):
        if "." not in path.split("/")[-1]:
            path += ".yaml"
    return path


def resolve(name: str, candidates: set[str]) -> list[str]:
    """Paths matching `name` the way the IDE matches them: exact, or by trailing name."""
    name = name.strip().strip("/")
    if name in candidates:
        return [name]
    return sorted(p for p in candidates if p.endswith("/" + name) or p == name)


async def read(path: str) -> str:
    return await read_file(path)


async def validate_content(content: str) -> dict | None:
    """Validate a draft before it is shown: write it to a scratch file, validate, delete.

    Returns the backend's report ({"valid", "type", "errors", "warnings"}), or None
    when the backend is unavailable (the caller falls back to local checks).
    """
    scratch = f"{SCRATCH_PREFIX}{uuid.uuid4().hex[:12]}.yaml"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            created = await client.post(
                f"{IDE_BACKEND_URL}/file/create", json={"path": scratch, "content": content}
            )
            created.raise_for_status()
            try:
                return await validate_file(scratch)
            finally:
                await client.delete(f"{IDE_BACKEND_URL}/delete", params={"path": scratch})
    except (httpx.HTTPError, ValidateFileError) as exc:
        log.warning("backend validation unavailable: %s", exc)
        return None


__all__ = [
    "Entries",
    "PathError",
    "ReadFileError",
    "clean_path",
    "list_entries",
    "read",
    "resolve",
    "validate_content",
]
