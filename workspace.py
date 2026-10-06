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
import time
import uuid
from dataclasses import dataclass, field

import httpx

from helpers import IDE_BACKEND_URL, ReadFileError, ValidateFileError, read_file, validate_file

log = logging.getLogger("hyperion.workspace")

TIMEOUT = httpx.Timeout(10, connect=3)
SCRATCH_PREFIX = ".hyperion-check-"  # never shown to the user; deleted right after use
YAML_SUFFIXES = (".yaml", ".yml")
# Circuit breaker: after a connection failure, skip the backend for this long instead
# of paying the connection attempt again on every call (listing, each validation...).
DOWN_FOR = 30

_down_until = 0.0


class PathError(ValueError):
    """A path the agent must not touch; the message can be shown to the user."""


class BackendDown(Exception):
    """The ide-backend could not be reached recently."""


def backend_available() -> bool:
    return time.monotonic() >= _down_until


def _mark_down(exc: Exception) -> None:
    global _down_until
    if backend_available():
        log.warning("ide-backend unreachable (%s) - skipping it for %ss", exc, DOWN_FOR)
    _down_until = time.monotonic() + DOWN_FOR


def _mark_up() -> None:
    global _down_until
    _down_until = 0.0


@dataclass
class Entries:
    files: set[str] = field(default_factory=set)
    folders: set[str] = field(default_factory=set)


async def list_entries() -> Entries:
    """Every file and folder in the workspace (walks the tree level by level)."""
    if not backend_available():
        raise BackendDown("the IDE backend was unreachable a moment ago")
    entries = Entries()
    try:
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
    except httpx.TransportError as exc:  # connection refused, timeout...
        _mark_down(exc)
        raise
    _mark_up()
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


def unsafe_paths(text: str) -> list[str]:
    """Absolute or '..' paths the user wrote. Checked on the user's own words, because
    the planner tends to "repair" them (C:/x.yaml -> x.yaml) before clean_path sees them."""
    found = []
    for token in text.split():
        token = token.strip("\"'`()[]{},;?!")
        parts = re.split(r"[\\/]", token)
        if (
            re.match(r"^[a-zA-Z]:[\\/]", token)  # C:\temp, C:/temp
            or token.startswith(("~/", "~\\"))  # home folder
            or (token.startswith("/") and token.count("/") >= 2)  # /etc/passwd
            or ".." in parts  # ../x, a/../../b, ..
        ):
            found.append(token)
    return found


def mentioned(path: str, text: str) -> bool:
    """Whether the user's message names this file or folder (by path, name or stem)."""
    said = text.lower()
    name = path.lower().rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    if path.lower() in said or name in said or stem in said:
        return True
    words = set(re.findall(r"[a-z0-9]+", said))
    parts = [p for p in re.split(r"[-_.]", stem) if len(p) >= 3 and p not in GENERIC_NAME_PARTS]
    return any(p in words for p in parts)


# Name parts too common to identify a file on their own ("the deployment file").
GENERIC_NAME_PARTS = {"yaml", "yml", "file", "app", "application", "profile", "deployment",
                      "service", "config", "device", "native"}


def resolve(name: str, candidates: set[str]) -> list[str]:
    """Paths matching `name` the way the IDE matches them: exact, or by trailing name."""
    name = name.strip().strip("/")
    if name in candidates:
        return [name]
    return sorted(p for p in candidates if p.endswith("/" + name) or p == name)


async def read(path: str) -> str:
    if not backend_available():
        raise ReadFileError("the IDE backend is unreachable right now")
    try:
        return await read_file(path)
    except ReadFileError as exc:
        if "cannot reach" in str(exc):  # helpers.read_file's connection-failure message
            _mark_down(exc)
        raise


async def validate_content(content: str) -> dict | None:
    """Validate a draft before it is shown: write it to a scratch file, validate, delete.

    Returns the backend's report ({"valid", "type", "errors", "warnings"}), or None
    when the backend is unavailable (the caller falls back to local checks).
    """
    if not backend_available():
        return None
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
    except httpx.TransportError as exc:  # unreachable: remember it for the next calls
        _mark_down(exc)
        return None
    except (httpx.HTTPError, ValidateFileError) as exc:
        if "cannot reach" in str(exc):
            _mark_down(exc)
        log.warning("backend validation unavailable: %s", exc)
        return None


__all__ = [
    "BackendDown",
    "Entries",
    "PathError",
    "backend_available",
    "ReadFileError",
    "clean_path",
    "list_entries",
    "read",
    "resolve",
    "validate_content",
]
