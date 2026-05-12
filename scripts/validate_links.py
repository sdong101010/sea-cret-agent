#!/usr/bin/env python3
"""Validate all URLs in the RFP reference markdown files.

Scans for https:// links, checks each with a HEAD request, and reports
broken, redirected, and duplicate URLs.

Usage:
    python scripts/validate_links.py
    python scripts/validate_links.py --fix   # auto-update redirected URLs in-place
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx

REFERENCE_DIR = Path.home() / ".cursor/skills/rfp-answering/reference"
URL_PATTERN = re.compile(r"https?://[^\s\)\"'>]+")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) link-checker/1.0",
}


def extract_urls(filepath: Path) -> list[tuple[int, str]]:
    """Return [(line_number, url), ...] from a markdown file."""
    results = []
    for i, line in enumerate(filepath.read_text().splitlines(), start=1):
        for match in URL_PATTERN.finditer(line):
            url = match.group().rstrip(".,;:)")
            results.append((i, url))
    return results


def check_url(client: httpx.Client, url: str) -> dict:
    """HEAD-request a URL and return status info."""
    try:
        resp = client.head(url, follow_redirects=False, timeout=10)
        final_url = url
        status = resp.status_code

        if status in (301, 302, 303, 307, 308):
            location = resp.headers.get("location", "")
            if location and not location.startswith("http"):
                from urllib.parse import urljoin
                location = urljoin(url, location)
            try:
                resp2 = client.get(location, follow_redirects=True, timeout=10)
                final_url = str(resp2.url)
                status = resp2.status_code
            except Exception:
                final_url = location

            return {
                "status": "REDIRECT",
                "code": resp.status_code,
                "final_url": final_url,
                "final_code": status,
            }

        if 200 <= status < 400:
            return {"status": "OK", "code": status}
        return {"status": "BROKEN", "code": status}

    except httpx.TimeoutException:
        return {"status": "TIMEOUT", "code": 0}
    except Exception as e:
        return {"status": "ERROR", "code": 0, "error": str(e)[:80]}


def find_duplicates(all_urls: dict[str, list[tuple[str, int]]]) -> list[tuple[str, list]]:
    """Find URLs appearing more than once across all files."""
    url_locations: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for filepath, entries in all_urls.items():
        for line_no, url in entries:
            url_locations[url].append((filepath, line_no))
    return [(url, locs) for url, locs in url_locations.items() if len(locs) > 1]


def main():
    parser = argparse.ArgumentParser(description="Validate reference file URLs")
    parser.add_argument("--fix", action="store_true", help="Auto-update redirected URLs in-place")
    args = parser.parse_args()

    if not REFERENCE_DIR.exists():
        print(f"Reference directory not found: {REFERENCE_DIR}")
        sys.exit(1)

    md_files = sorted(REFERENCE_DIR.glob("*.md"))
    if not md_files:
        print("No .md files found")
        sys.exit(1)

    all_urls: dict[str, list[tuple[int, str]]] = {}
    for f in md_files:
        urls = extract_urls(f)
        if urls:
            all_urls[f.name] = urls

    total = sum(len(v) for v in all_urls.values())
    print(f"\nScanning {total} URLs across {len(all_urls)} files...\n")

    broken = []
    redirected = []
    ok_count = 0
    fixes: dict[str, list[tuple[str, str]]] = defaultdict(list)

    with httpx.Client(headers=HEADERS) as client:
        for filename, entries in all_urls.items():
            print(f"  {filename}")
            seen_in_file = set()
            for line_no, url in entries:
                if url in seen_in_file:
                    continue
                seen_in_file.add(url)

                result = check_url(client, url)
                status = result["status"]

                if status == "OK":
                    ok_count += 1
                    print(f"    L{line_no:>3}  OK     {url[:80]}")
                elif status == "REDIRECT":
                    redirected.append((filename, line_no, url, result))
                    final = result.get("final_url", "?")
                    print(f"    L{line_no:>3}  REDIR  {url[:60]}")
                    print(f"           -> {final[:80]}")
                    if args.fix and final != url and final.startswith("http"):
                        fixes[filename].append((url, final))
                elif status == "BROKEN":
                    broken.append((filename, line_no, url, result["code"]))
                    print(f"    L{line_no:>3}  BROKEN ({result['code']})  {url[:80]}")
                elif status == "TIMEOUT":
                    print(f"    L{line_no:>3}  TIMEOUT  {url[:80]}")
                else:
                    err = result.get("error", "unknown")
                    print(f"    L{line_no:>3}  ERROR  {url[:60]}  ({err})")

    dupes = find_duplicates(all_urls)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  OK:         {ok_count}")
    print(f"  Redirected: {len(redirected)}")
    print(f"  Broken:     {len(broken)}")
    print(f"  Duplicates: {len(dupes)}")

    if broken:
        print(f"\n{'BROKEN URLS':=^70}")
        for fname, line, url, code in broken:
            print(f"  {fname}:{line}  [{code}]  {url}")

    if redirected:
        print(f"\n{'REDIRECTED URLS':=^70}")
        for fname, line, url, result in redirected:
            print(f"  {fname}:{line}")
            print(f"    OLD: {url}")
            print(f"    NEW: {result.get('final_url', '?')}")

    if dupes:
        print(f"\n{'DUPLICATE URLS':=^70}")
        for url, locations in dupes:
            print(f"  {url[:80]}")
            for fname, line in locations:
                print(f"    -> {fname}:{line}")

    if args.fix and fixes:
        print(f"\n{'APPLYING FIXES':=^70}")
        for filename, replacements in fixes.items():
            filepath = REFERENCE_DIR / filename
            content = filepath.read_text()
            for old_url, new_url in replacements:
                content = content.replace(old_url, new_url)
                print(f"  {filename}: {old_url[:50]} -> {new_url[:50]}")
            filepath.write_text(content)
        print(f"\nUpdated {len(fixes)} file(s).")

    if broken:
        sys.exit(1)


if __name__ == "__main__":
    main()
