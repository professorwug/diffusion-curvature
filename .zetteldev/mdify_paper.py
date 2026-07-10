"""Convert a paper PDF/arXiv URL to markdown via Datalab Marker."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from base64 import b64decode
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv


DATALAB_API = "https://www.datalab.to/api/v1/marker"
POLL_SECONDS = 3
TIMEOUT_SECONDS = 600
DATALAB_PARAMS = {
    "output_format": "markdown",
    "force_ocr": False,
    "format_lines": False,
    "paginate": False,
    "use_llm": False,
    "strip_existing_ocr": False,
    "disable_image_extraction": False,
    "max_pages": None,
    "page_range": None,
}


def normalize_paper_url(url: str) -> str:
    """Return a PDF URL for direct PDF or arXiv abstract URLs."""
    parsed = urlparse(url)
    if parsed.netloc.endswith("arxiv.org") and parsed.path.startswith("/abs/"):
        paper_id = parsed.path.removeprefix("/abs/").strip("/")
        return f"https://arxiv.org/pdf/{paper_id}.pdf"
    return url


def output_slug(url: str) -> str:
    """Create a readable markdown filename from a paper URL."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if parsed.netloc.endswith("arxiv.org"):
        match = re.search(r"(?:abs|pdf)/([^/]+?)(?:\.pdf)?$", path)
        if match:
            return f"arxiv-{match.group(1)}"

    stem = Path(path).stem or parsed.netloc or "paper"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return slug or "paper"


def datalab_headers() -> dict[str, str]:
    """Load Datalab credentials from the environment."""
    key = os.getenv("DATALAB_KEY")
    if not key:
        raise RuntimeError(
            "Missing Datalab credentials. Set DATALAB_KEY in the environment "
            "or repo .env file."
        )
    return {"X-Api-Key": key}


def download_to_temp(url: str) -> Path:
    """Download a paper URL to a temporary PDF file."""
    source_url = normalize_paper_url(url)
    suffix = Path(urlparse(source_url).path).suffix or ".pdf"
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        response = client.get(source_url)
        response.raise_for_status()

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(response.content)
        return Path(tmp.name)


async def submit_marker(pdf_path: Path, headers: dict[str, str]) -> dict:
    """Submit a PDF file to the Datalab Marker API."""
    with pdf_path.open("rb") as file:
        files = {"file": (pdf_path.name, file, "application/pdf")}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                DATALAB_API,
                files=files,
                data={k: v for k, v in DATALAB_PARAMS.items() if v is not None},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()

    if not payload.get("success", True):
        raise RuntimeError(f"Datalab submission failed: {payload.get('error')}")
    if "request_check_url" not in payload:
        raise RuntimeError(f"Datalab did not return request_check_url: {payload}")
    return payload


async def poll_marker(submission: dict, headers: dict[str, str]) -> dict:
    """Poll the Datalab Marker API until conversion completes."""
    check_url = submission["request_check_url"]
    deadline = time.monotonic() + TIMEOUT_SECONDS
    async with httpx.AsyncClient(timeout=60) as client:
        while time.monotonic() < deadline:
            response = await client.get(check_url, headers=headers)
            response.raise_for_status()
            payload = response.json()
            status = payload.get("status")
            if status == "complete":
                return payload
            if status == "failed":
                raise RuntimeError(f"Datalab conversion failed: {payload.get('error')}")
            print(f"Datalab status: {status or 'unknown'}", file=sys.stderr)
            await asyncio.sleep(POLL_SECONDS)

    raise TimeoutError(f"Datalab conversion did not finish within {TIMEOUT_SECONDS}s")


def save_markdown(result: dict, output_path: Path) -> None:
    """Write markdown and any extracted image artifacts returned by Datalab."""
    markdown = result.get("markdown")
    if markdown is None:
        raise RuntimeError(f"Datalab result did not include markdown: {result.keys()}")

    output_path.write_text(markdown, encoding="utf-8")
    for name, encoded in result.get("images", {}).items():
        image_path = output_path.parent / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b64decode(encoded))


async def convert(url: str, output_dir: Path) -> Path:
    """Convert a URL and write markdown to output_dir."""
    load_dotenv()
    headers = datalab_headers()
    source_url = normalize_paper_url(url)
    pdf_path = download_to_temp(source_url)
    try:
        submission = await submit_marker(pdf_path, headers)
        print(
            f"Submitted {source_url} to Datalab Marker",
            file=sys.stderr,
        )
        result = await poll_marker(submission, headers)
    finally:
        pdf_path.unlink(missing_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_slug(source_url)}.md"
    save_markdown(result, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="PDF URL or arXiv abs/pdf URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where the markdown file should be written.",
    )
    args = parser.parse_args()

    try:
        output_path = asyncio.run(convert(args.url, args.output_dir))
    except Exception as exc:
        raise SystemExit(f"mdify-paper failed: {exc}") from exc

    print(output_path)


if __name__ == "__main__":
    main()
