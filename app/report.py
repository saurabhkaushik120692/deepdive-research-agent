"""Render the final markdown report. Pure function: everything it needs is passed in."""

from app.graph.schemas import Conflict, SynthesisOutput
from app.models import Claim, Source, Verdict

PARTIAL_MARKER = "⚠ partially verified"


def render_report(
    question: str,
    synthesis: SynthesisOutput,
    claims: dict[int, Claim],
    verdicts: dict[int, Verdict],
    chunk_sources: dict[int, Source],
    thin_sq_texts: list[str],
    conflicts: list[Conflict],
) -> str:
    numbers: dict[str, int] = {}  # source url -> citation number, in order of first citation
    cited: list[Source] = []

    def cite(claim_id: int) -> list[int]:
        out = []
        for chunk_id in claims[claim_id].chunk_ids:
            src = chunk_sources.get(chunk_id)
            if src is None:
                continue
            if src.url not in numbers:
                numbers[src.url] = len(numbers) + 1
                cited.append(src)
            if numbers[src.url] not in out:
                out.append(numbers[src.url])
        return out

    lines = [f"# {synthesis.title}", "", f"*Question:* {question}", "", synthesis.summary, ""]

    for section in synthesis.sections:
        lines += [f"## {section.heading}", ""]
        for p in section.paragraphs:
            refs: list[int] = []
            for cid in p.claim_ids:
                refs += [n for n in cite(cid) if n not in refs]
            text = p.text + (" " + "".join(f"[{n}]" for n in refs) if refs else "")
            if any(verdicts[cid].label == "partial" for cid in p.claim_ids if cid in verdicts):
                text += f" {PARTIAL_MARKER}"
            lines += [text, ""]

    if thin_sq_texts:
        lines += [
            "## Limited evidence",
            "",
            "These sub-questions could not be answered with verified evidence, so the report says "
            "nothing about them rather than guessing:",
            "",
        ]
        lines += [f"- {t}" for t in thin_sq_texts] + [""]

    if conflicts:
        lines += ["## Conflicting sources", ""]
        for c in conflicts:
            a, b = claims.get(c.claim_a), claims.get(c.claim_b)
            if a and b:
                lines.append(f'- "{a.text}" vs "{b.text}": {c.explanation}')
        lines.append("")

    lines += ["## Rejected claims", ""]
    rejected = [(claims[cid], v) for cid, v in sorted(verdicts.items()) if v.label == "unsupported"]
    if rejected:
        lines += [f'- "{c.text}": {v.reason}' for c, v in rejected]
    else:
        lines.append("None. Every drafted claim was supported by its cited source.")
    lines.append("")

    lines += ["## Sources", ""]
    lines += [f"[{numbers[s.url]}] {s.title or s.domain} ({s.url}), tier {s.tier}" for s in cited]
    return "\n".join(lines).rstrip() + "\n"
