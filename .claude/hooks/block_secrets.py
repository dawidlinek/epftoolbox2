#!/usr/bin/env python3
"""PreToolUse hook for Bash: stop a `git commit` that would include a real-looking API key.

`DataPipeline.save()` writes `api_key` into the YAML in plain text, and the repo tracks pipeline
YAMLs, so a real ENTSO-E token can end up in a commit. The hook runs before the whole Bash command,
so a `git add ... && git commit` has not staged anything yet; it therefore scans every change
against HEAD plus untracked files, which over-approximates what will be committed.

Exit code 2 blocks the command and shows stderr to Claude.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

GIT_COMMIT = re.compile(r"\bgit\s+(?:-C\s+\S+\s+|-c\s+\S+\s+)*commit\b")
SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (api[_-]?key|security[_-]?token|access[_-]?token|secret|password)   # key name
    ["']?\s*[:=]\s*["']                                                  # : or = then a quote
    ([A-Za-z0-9_\-]{20,})                                                # long literal value
    """
)
YAML_SECRET = re.compile(r"(?i)^\s*(api_key|security_token|token|secret|password)\s*:\s*['\"]?([A-Za-z0-9_\-]{20,})")
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
PLACEHOLDER = re.compile(r"(?i)your|here|example|dummy|placeholder|changeme|xxxx|test|fake")
SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".md", ".mdx", ".txt", ".cfg", ".ini", ".sh", ".env"}


def git(project: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=project, capture_output=True, text=True).stdout


def added_lines(project: Path):
    """Yield (path, line) for lines that differ from HEAD, including untracked files."""
    current = None
    for line in git(project, "diff", "HEAD", "-U0", "--no-color").splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("+") and current:
            yield current, line[1:]
    for name in git(project, "ls-files", "--others", "--exclude-standard").splitlines():
        path = project / name
        if path.suffix in SCANNED_SUFFIXES and path.is_file() and path.stat().st_size < 1_000_000:
            for text in path.read_text(errors="ignore").splitlines():
                yield name, text


def mask(value: str) -> str:
    return value[:4] + "…" if len(value) > 4 else "…"


def main() -> int:
    payload = json.load(sys.stdin)
    command = (payload.get("tool_input") or {}).get("command") or ""
    if not GIT_COMMIT.search(command):
        return 0

    project = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()).resolve()
    findings = []

    for name in git(project, "status", "--porcelain", "--untracked-files=all").splitlines():
        path = name[3:].strip()
        env_name = Path(path).name
        if env_name == ".env" or (env_name.startswith(".env.") and not env_name.endswith(".example")):
            findings.append(f"{path}: env file would be committed")

    for path, line in added_lines(project):
        if Path(path).suffix not in SCANNED_SUFFIXES or path.endswith("uv.lock"):
            continue
        for pattern in (SECRET_ASSIGNMENT, YAML_SECRET):
            match = pattern.search(line)
            if match and not PLACEHOLDER.search(match.group(2)):
                findings.append(f"{path}: {match.group(1)} = {mask(match.group(2))}")
        if UUID.search(line) and re.search(r"(?i)key|token|secret", line):
            findings.append(f"{path}: UUID next to a key/token name (ENTSO-E tokens are UUIDs)")

    if findings:
        unique = "\n".join(f"  - {f}" for f in dict.fromkeys(findings))
        print(
            "Blocked git commit: possible secrets in changes vs HEAD or untracked files:\n"
            f"{unique}\n"
            "Replace real keys with a placeholder or ${ENTSOE_API_KEY} before committing. "
            "If this is a false positive, tell the user and let them commit.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
