"""Download every RAG source into data/docs/, then run build_index.py.

    uv run fetch_docs.py && uv run build_index.py

Sources: the official document set (Google Drive folder from the kick-off slides),
the IDE tutorial site and the public HYPER-AI website.
"""

import html
import re
from pathlib import Path

import httpx

from html_text import html_to_markdown

DOCS = Path("data/docs")

DRIVE_FOLDER = "1uaNu4Pn_QsyXmRDS-DqQ-1mHzvX3zT1r"

TUTORIAL = "https://ide-tutorial.hyperai.di.uoa.gr/"
TUTORIAL_PAGES = {
    "": "tutorial_introduction",
    "quick-start-demo/": "tutorial_quick_start",
    "ide-guide/": "tutorial_ide_guide",
    "dsl/native-apps/": "tutorial_dsl_native_apps",
    "dsl/devices/": "tutorial_dsl_device_apps",
    "cookbook/": "tutorial_cookbook",
    "hyperion-agent/": "tutorial_hyperion_agent",
}

WEBSITE = "https://hyper-ai-project.eu/"
WEBSITE_PAGES = {
    "": "website_home",
    "about-us": "website_about_us",
    "cognitive-computing-continuum-cluster": "website_computing_continuum_cluster",
    "use-cases": "website_use_cases",
    "use-cases/farming-and-agriculture": "website_use_case_agriculture",
    "use-cases/green-energy": "website_use_case_green_energy",
    "use-cases/healthcare": "website_use_case_healthcare",
    "use-cases/industry-40": "website_use_case_industry_40",
    "use-cases/mobility-and-automotive": "website_use_case_mobility_automotive",
    "publications-and-deliverables": "website_publications_and_deliverables",
    "events/veleshack-riding-edge-cloud-continuum": "website_event_veleshack",
    "news/veleshack-2026-hyper-ai-brings-agentic-ai-three-day-edge-cloud-hackathon": "website_news_veleshack",
    "news/lcf-bench-reproducible-benchmark-feasibility-aware-logical-cluster-formation-cloud-edge-iot": "website_news_lcf_bench",
}


def fetch_drive(client: httpx.Client) -> None:
    listing = client.get(f"https://drive.google.com/embeddedfolderview?id={DRIVE_FOLDER}").text
    files = re.findall(
        r'href="https://drive.google.com/file/d/([^/]+)/view[^"]*".*?class="flip-entry-title">([^<]+)<',
        listing,
        re.S,
    )
    for file_id, name in files:
        name = html.unescape(name).strip()
        data = client.get(
            "https://drive.usercontent.google.com/download",
            params={"id": file_id, "export": "download"},
        ).content
        (DOCS / name).write_bytes(data)
        print(f"  drive     {name} ({len(data)} bytes)")


def fetch_pages(client: httpx.Client, base: str, pages: dict, container: str) -> None:
    for path, name in pages.items():
        response = client.get(base + path)
        if response.status_code != 200:
            print(f"  SKIPPED   {base + path}: HTTP {response.status_code}")
            continue
        text = html_to_markdown(response.text, container)
        (DOCS / f"{name}.md").write_text(text + "\n", encoding="utf-8")
        print(f"  {container:9s} {name}.md ({len(text.split())} words)")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    transport = httpx.HTTPTransport(retries=3)  # retries failed connections (DNS blips)
    with httpx.Client(timeout=60, follow_redirects=True, transport=transport) as client:
        fetch_drive(client)
        fetch_pages(client, TUTORIAL, TUTORIAL_PAGES, "article")
        fetch_pages(client, WEBSITE, WEBSITE_PAGES, "main")


if __name__ == "__main__":
    main()
