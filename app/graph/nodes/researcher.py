import asyncio
from urllib.parse import urlparse

from langchain_core.runnables import RunnableConfig
from sqlmodel import select

from app.deps import Deps, SearchResult
from app.events import log_event
from app.graph.nodes import get_deps, node_events
from app.models import EvidenceChunk, SearchRecord, Source, SubQuestion
from app.tools.chunking import chunk_text, clean_text, is_noise
from app.tools.fetch import MIN_TEXT_CHARS
from app.tools.tiers import source_tier


async def _page_text(deps: Deps, result: SearchResult) -> str | None:
    """Tavily's raw content if it's a real page, else fetch it; any error skips the source."""
    if result.content and len(result.content) >= MIN_TEXT_CHARS:
        return result.content
    try:
        return await deps.fetcher.fetch_text(result.url)
    except Exception:
        return None


async def research_node(payload: dict, config: RunnableConfig) -> dict:
    """One researcher per sub-question (fanned out with LangGraph Send).

    A follow-up question uses it too (payload has followup_id): it then skips every URL the run
    already fetched, so the search really adds new evidence, and its events carry followup_id."""
    deps = get_deps(config)
    settings = deps.settings
    run_id, sq_id, round_ = payload["run_id"], payload["sub_question_id"], payload["round"]
    followup = {"followup_id": payload["followup_id"]} if payload.get("followup_id") else {}
    seen_scope = (
        EvidenceChunk.run_id == run_id if followup else EvidenceChunk.sub_question_id == sq_id
    )

    with node_events(deps, run_id, "researcher", sub_question_id=sq_id, round=round_, **followup):
        with deps.session_factory() as session:
            query = session.get(SubQuestion, sq_id).query
            seen_urls = set(
                session.exec(
                    select(Source.url)
                    .join(EvidenceChunk, EvidenceChunk.source_id == Source.id)
                    .where(seen_scope)
                ).all()
            )

        results = await deps.search.search(query, settings.search_results_per_query)
        fresh = [r for r in results if r.url not in seen_urls]
        texts = await asyncio.gather(*(_page_text(deps, r) for r in fresh))

        new_ids: list[int] = []
        new_texts: list[str] = []
        with deps.session_factory() as session:
            for result, raw in zip(fresh, texts, strict=True):
                text = clean_text(raw) if raw else ""
                pieces = [
                    p
                    for p in chunk_text(
                        text,
                        settings.chunk_words,
                        settings.chunk_overlap_words,
                        settings.chunk_max_chars,
                    )
                    if not is_noise(p)
                ]
                if not pieces:
                    reason = "no usable text" if not text else "only tables/markup, no prose"
                    log_event(
                        session,
                        run_id,
                        "researcher",
                        "source_skipped",
                        {"sub_question_id": sq_id, "url": result.url, "reason": reason} | followup,
                    )
                    continue
                source = Source(
                    run_id=run_id,
                    url=result.url,
                    domain=(urlparse(result.url).hostname or "").removeprefix("www."),
                    title=result.title,
                    tier=source_tier(result.url),
                )
                session.add(source)
                session.commit()
                session.refresh(source)
                rows = [
                    EvidenceChunk(
                        source_id=source.id,
                        run_id=run_id,
                        sub_question_id=sq_id,
                        text=piece,
                        position=i,
                    )
                    for i, piece in enumerate(pieces[: settings.max_chunks_per_source])
                ]
                session.add_all(rows)
                session.commit()
                for r in rows:
                    session.refresh(r)
                    new_ids.append(r.id)
                    new_texts.append(r.text)

            session.add(
                SearchRecord(
                    sub_question_id=sq_id, query=query, round=round_, n_results=len(results)
                )
            )
            sq = session.get(SubQuestion, sq_id)
            if not followup:
                sq.status = "researched"
                session.add(sq)
            session.commit()
            log_event(
                session,
                run_id,
                "researcher",
                "search_done",
                {
                    "sub_question_id": sq_id,
                    "round": round_,
                    "query": query,
                    "n_results": len(results),
                    "n_new_chunks": len(new_ids),
                }
                | followup,
            )

        await deps.vectors.add(run_id, sq_id, new_ids, new_texts)
    return {"researched": [sq_id]}
