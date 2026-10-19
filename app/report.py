"""Render the final markdown report. Pure function: everything it needs is passed in."""

from app.graph.schemas import Conflict, SynthesisOutput
from app.models import Claim, Source, Verdict

PARTIAL_MARKER = "⚠ partially verified"


class Citations:
    """Numbers sources in order of first citation; shared by the markdown and HTML views."""

    def __init__(self, claims: dict[int, Claim], chunk_sources: dict[int, Source]) -> None:
        self.claims, self.chunk_sources = claims, chunk_sources
        self.numbers: dict[str, int] = {}
        self.cited: list[Source] = []

    def sources_for(self, claim_id: int) -> list[tuple[int, Source]]:
        out: list[tuple[int, Source]] = []
        for chunk_id in self.claims[claim_id].chunk_ids:
            src = self.chunk_sources.get(chunk_id)
            if src is None:
                continue
            if src.url not in self.numbers:
                self.numbers[src.url] = len(self.numbers) + 1
                self.cited.append(src)
            n = self.numbers[src.url]
            if all(n != m for m, _ in out):
                out.append((n, src))
        return out


def render_report(
    question: str,
    synthesis: SynthesisOutput,
    claims: dict[int, Claim],
    verdicts: dict[int, Verdict],
    chunk_sources: dict[int, Source],
    thin_sq_texts: list[str],
    conflicts: list[Conflict],
) -> str:
    citations = Citations(claims, chunk_sources)

    lines = [f"# {synthesis.title}", "", f"*Question:* {question}", "", synthesis.summary, ""]

    for section in synthesis.sections:
        lines += [f"## {section.heading}", ""]
        for p in section.paragraphs:
            refs: list[int] = []
            for cid in p.claim_ids:
                refs += [n for n, _ in citations.sources_for(cid) if n not in refs]
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
    lines += [
        f"[{citations.numbers[s.url]}] {s.title or s.domain} ({s.url}), tier {s.tier}"
        for s in citations.cited
    ]
    return "\n".join(lines).rstrip() + "\n"
