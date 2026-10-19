"""Shared eval helpers: load claims together with their verdict and cited source text."""

from dataclasses import dataclass

from sqlmodel import select

from app.deps import Deps
from app.models import Claim, EvidenceChunk, Verdict


@dataclass
class ClaimRow:
    claim: Claim
    verdict: Verdict
    source_text: str


def load_claims(deps: Deps, run_ids: list[str], labels: tuple[str, ...] | None = None):
    with deps.session_factory() as session:
        q = (
            select(Claim, Verdict)
            .join(Verdict, Verdict.claim_id == Claim.id)
            .where(Claim.run_id.in_(run_ids))
            .order_by(Claim.id)
        )
        if labels:
            q = q.where(Verdict.label.in_(labels))
        rows = session.exec(q).all()
        chunk_ids = {cid for c, _ in rows for cid in c.chunk_ids}
        chunks = {
            ch.id: ch.text
            for ch in session.exec(
                select(EvidenceChunk).where(EvidenceChunk.id.in_(list(chunk_ids)))
            ).all()
        }
    return [
        ClaimRow(c, v, "\n\n".join(chunks[i] for i in c.chunk_ids if i in chunks)) for c, v in rows
    ]
