"""The HYPER-AI application-profile DSL: templates, required fields, local checks.

Everything is read from the official tutorial pages in data/docs (copied into the
image), so it cannot drift from the specification:
- templates: the cookbook examples (native nginx, device Docker, device Android);
- required fields: the ✅ rows of the DSL reference tables.

The ide-backend's validator (workspace.validate_content) is the real check;
`check` here is the fallback when the backend cannot be reached.
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import yaml

DOCS = Path(__file__).parent / "data" / "docs"

Kind = Literal["native", "device"]
Workload = Literal["DockerImage", "AndroidApk", "esp32Binary"]
WORKLOADS: tuple[Workload, ...] = ("DockerImage", "AndroidApk", "esp32Binary")

SPEC_PAGE = {"native": "tutorial_dsl_native_apps.md", "device": "tutorial_dsl_device_apps.md"}
# Sections that are not written by the user (runtime state) or are only examples.
SKIPPED_SECTIONS = ("status", "Examples", "Custom Configuration")


@dataclass
class Field:
    path: str  # e.g. "specs.runtime.containerImage.uri"
    row: str  # the specification's table row, used as evidence for the writer
    workload: str | None = None  # device fields that apply to one workload kind only


@cache
def required_fields(kind: Kind) -> list[Field]:
    """The ✅ rows of the specification, with their full dotted path."""
    section, workload, fields = "", None, []
    for line in (DOCS / SPEC_PAGE[kind]).read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^(#{3,4}) (\S+)", line)
        if heading:
            level, name = heading.groups()
            if level == "###":
                section, workload = name, None
            else:
                workload = name if name.lower() in {w.lower() for w in WORKLOADS} else None
            continue
        if not line.startswith("| ") or "✅" not in line or section.startswith(SKIPPED_SECTIONS):
            continue
        name = line.split("|")[1].strip()
        if name.startswith(("apiVersion", "kind", "metadata", "spec")) and section in ("", "##"):
            path = name
        elif kind == "device" and section == "spec" and "." not in name:
            path = f"spec.{name}"
        else:
            path = f"{section}.{name}" if not name.startswith(section) else name
        # Native rows are relative to applicationProfile, except the two root rows.
        path = path.removeprefix("applicationProfile.")
        fields.append(Field(path, line, workload))
    return fields


@cache
def cookbook() -> dict[str, str]:
    blocks = re.findall(
        r"```\n(.*?)\n```", (DOCS / "tutorial_cookbook.md").read_text(encoding="utf-8"), re.S
    )
    templates = {}
    for block in blocks:
        doc = yaml.safe_load(block)
        if "applicationProfile" in doc:
            templates.setdefault("native", block)
        else:
            workload = doc["spec"]["workload"]["kind"]
            templates.setdefault(workload, block)
    return templates


def template(kind: Kind, workload: Workload | None = None) -> str:
    """The closest official example; ESP32 has none, so it gets the Docker one."""
    examples = cookbook()
    if kind == "native":
        return examples["native"]
    return examples.get(workload or "DockerImage", examples["DockerImage"])


def evidence(kind: Kind, workload: Workload | None = None) -> str:
    """What the writer gets: the template and the required fields that apply."""
    rows = [
        f.row
        for f in required_fields(kind)
        if f.workload is None or (workload and f.workload.lower() == workload.lower())
    ]
    return (
        f"Official example ({kind}{' / ' + workload if workload else ''} application):\n"
        f"```yaml\n{template(kind, workload)}\n```\n\n"
        "Required fields (from the HYPER-AI DSL specification):\n"
        "| Field | Type | Required | Example | Description |\n" + "\n".join(rows)
    )


def strip_fences(text: str) -> str:
    """Models like to wrap YAML in ``` fences or add a sentence around it."""
    fenced = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", text, re.S)
    return (fenced.group(1) if fenced else text).strip() + "\n"


def kind_of(doc: object) -> Kind | None:
    if isinstance(doc, dict) and "applicationProfile" in doc:
        return "native"
    if isinstance(doc, dict) and doc.get("kind") == "Application":
        return "device"
    return None


def _get(doc: object, dotted: str) -> object:
    for key in dotted.split("."):
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def check(text: str, kind: Kind) -> list[str]:
    """Local fallback validation: YAML syntax, root type, required fields."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [f"invalid YAML: {exc}".splitlines()[0]]
    found = kind_of(doc)
    if found != kind:
        root = "applicationProfile" if kind == "native" else "apiVersion: hyper.ai/v1 + kind: Application"
        return [f"expected a {kind} application ({root} at the root)"]
    errors = []
    base = doc["applicationProfile"] if kind == "native" else doc
    workload = _get(doc, "spec.workload.kind")
    for f in required_fields(kind):
        if "[]" in f.path or (f.workload and f.workload.lower() != str(workload).lower()):
            continue
        parent = f.path.rsplit(".", 1)[0] if "." in f.path else ""
        # Fields of optional sections (e.g. network) are only required if the section exists.
        if parent and _get(base, parent) is None and parent.count(".") >= 1 and kind == "device":
            continue
        if _get(base, f.path) in (None, ""):
            errors.append(f"{f.path} is required")
    return errors
