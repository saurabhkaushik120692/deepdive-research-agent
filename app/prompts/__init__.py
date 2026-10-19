from functools import cache
from pathlib import Path

_DIR = Path(__file__).parent


@cache
def load_prompt(name: str) -> str:
    """System prompt for an agent, from app/prompts/<name>.md (one file per agent)."""
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")
