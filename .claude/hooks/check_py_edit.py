"""PostToolUse hook: lint + Python 3.11 compile check for any .py file Claude edits.

Exit code 2 feeds the error back to Claude so it fixes the problem right away.
"""

import json
import py_compile
import subprocess
import sys

payload = json.load(sys.stdin)
path = (payload.get("tool_input") or {}).get("file_path") or ""
if not path.endswith(".py"):
    sys.exit(0)

problems = []
try:
    py_compile.compile(path, doraise=True)
except py_compile.PyCompileError as e:
    problems.append(f"Python 3.11 compile error:\n{e.msg}")

lint = subprocess.run(
    ["uv", "run", "ruff", "check", "--quiet", path], capture_output=True, text=True
)
if lint.returncode != 0:
    problems.append(f"ruff:\n{lint.stdout}{lint.stderr}")

if problems:
    print("\n\n".join(problems), file=sys.stderr)
    sys.exit(2)
