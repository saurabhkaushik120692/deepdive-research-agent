"""Plain-code loop rules (no LLM): when is a sub-question too thin, and may we loop again."""


def thin_sub_questions(supported_counts: dict[int, int], min_supported: int = 2) -> list[int]:
    return sorted(sq_id for sq_id, n in supported_counts.items() if n < min_supported)


def should_loop(thin_ids: list[int], round_: int, max_extra_rounds: int = 2) -> bool:
    return bool(thin_ids) and round_ < max_extra_rounds
