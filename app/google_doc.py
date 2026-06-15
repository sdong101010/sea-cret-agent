"""Push a meeting summary to Google Docs.

Uses the same `claude -p` subprocess pattern as the answer engine, leveraging
the user's already-configured google-adc MCP server (which exposes
mcp__google-adc__docs_create).

We send Claude:
  - The meeting summary (markdown the SummaryGenerator produced)
  - A title for the doc
  - Instructions to call docs_create and return only the resulting doc URL

Returns: a dict {"url": "...", "title": "..."} on success, or {"error": "..."}.
"""

import asyncio
import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


CLAUDE_TIMEOUT_SECONDS = 240.0


PROMPT_TEMPLATE = """\
You have access to the google-adc MCP server. Create a new Google Doc with \
the following content. Title and content are below.

INSTRUCTIONS:
1. Call mcp__google-adc__docs_create with the given title.
2. The docs_create tool returns the new document. Note its documentId.
3. Insert the markdown content using mcp__google-adc__docs_batch_update. \
IMPORTANT: split the body into chunks of roughly 6000-8000 characters and \
make MULTIPLE docs_batch_update calls -- one per chunk -- so no single \
request times out. Each chunk should contain a coherent section break (end \
on a paragraph or heading boundary, never mid-sentence). Insert chunks in \
order.
4. Return ONLY a single line of valid JSON in this exact shape:
   {{"url": "<full https://docs.google.com/... URL>", "title": "<the title>"}}

Do NOT include any other text, explanation, or markdown fencing. Just the JSON.

If a chunked write partially succeeds (doc was created but some chunks \
failed), still return the URL with a "partial": true flag like \
{{"url": "...", "title": "...", "partial": true, "note": "<which chunks failed>"}}.

If the doc could not be created at all, return: {{"error": "<short reason>"}}.

TITLE:
{title}

CONTENT (markdown):
{content}
"""


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _strip_full_transcript(md: str) -> str:
    """Drop the `## Full Transcript` section. Long transcripts blow past the
    Google Docs request budget even when chunked, and the user only wants the
    summary + Q&A in the doc."""
    lines = md.splitlines()
    out: list[str] = []
    skipping = False
    for line in lines:
        if line.startswith("## Full Transcript"):
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            out.append(line)
    # Trim trailing `---` separators that would otherwise dangle.
    while out and out[-1].strip() in ("", "---"):
        out.pop()
    return "\n".join(out) + "\n"


async def create_doc(summary_markdown: str, meeting_title: str | None = None) -> dict:
    """Create a Google Doc from the meeting summary.

    Returns {"url": "...", "title": "..."} on success, {"error": "..."} otherwise.
    """
    now = datetime.now()
    if meeting_title:
        title = f"{meeting_title} - {now.strftime('%Y-%m-%d %H:%M')}"
    else:
        title = f"Sea-cret Agent - {now.strftime('%Y-%m-%d %H:%M')}"

    doc_body = _strip_full_transcript(summary_markdown)
    prompt = PROMPT_TEMPLATE.format(title=title, content=doc_body)

    cmd = [
        "claude",
        "--print",
        "--output-format", "json",
        "--no-session-persistence",
        "--permission-mode", "bypassPermissions",
        prompt,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=CLAUDE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.warning("Google Doc creation timed out after %.0fs", CLAUDE_TIMEOUT_SECONDS)
        return {"error": "Timed out talking to Google"}
    except FileNotFoundError:
        logger.error("`claude` CLI not found on PATH")
        return {"error": "claude CLI not installed"}
    except Exception as e:
        logger.exception("Google Doc creation failed")
        return {"error": str(e)}

    if proc.returncode != 0:
        logger.error("claude -p exited %d. stderr: %s",
                     proc.returncode, stderr.decode("utf-8", "replace")[:500])
        return {"error": f"claude exited {proc.returncode}"}

    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        logger.error("claude envelope not JSON: %s", stdout[:300])
        return {"error": "Bad envelope from claude"}

    inner = envelope.get("result", "")
    data = _extract_json_object(inner) if isinstance(inner, str) else None
    if data is None:
        logger.error("Couldn't parse JSON from claude result: %s", str(inner)[:300])
        return {"error": "Couldn't parse Google Docs response"}

    if "error" in data:
        return {"error": data["error"]}

    url = data.get("url")
    if not url:
        return {"error": "No URL in response"}

    result = {"url": url, "title": data.get("title") or title}
    if data.get("partial"):
        result["partial"] = True
        if data.get("note"):
            result["note"] = data["note"]
        logger.warning("Created Google Doc with PARTIAL body: %s -> %s (%s)",
                       title, url, data.get("note", ""))
    else:
        logger.info("Created Google Doc: %s -> %s", title, url)
    return result
