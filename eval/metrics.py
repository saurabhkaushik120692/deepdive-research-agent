"""Pure metric functions. Rates return None when there is nothing to measure (shown as n/a)."""

from statistics import mean

from app.models import Verdict


def _share(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def catch_rate(mutant_labels: list[str]) -> float | None:
    """Share of planted false claims the verifier rejected."""
    return _share([label == "unsupported" for label in mutant_labels])


def wrongful_rejection_rate(original_labels: list[str]) -> float | None:
    """Share of previously supported claims rejected on re-verification (false alarms)."""
    return _share([label == "unsupported" for label in original_labels])


def quote_validity_rate(verdicts: list[Verdict]) -> float | None:
    """Share of verifier quotes that really appear in the cited source."""
    return _share([v.quote_found for v in verdicts if v.quote])


def agreement(pred: list[bool], gold: list[bool]) -> tuple[float | None, float | None]:
    """Accuracy and Cohen's kappa of the verifier against hand labels."""
    if not gold:
        return None, None
    n = len(gold)
    po = sum(p == g for p, g in zip(pred, gold, strict=True)) / n
    p_pred, p_gold = sum(pred) / n, sum(gold) / n
    pe = p_pred * p_gold + (1 - p_pred) * (1 - p_gold)
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    return po, kappa


def _percentile(sorted_vals: list[float], q: float) -> float:
    pos = (len(sorted_vals) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p50": None, "p95": None}
    s = sorted(values)
    return {"mean": mean(s), "p50": _percentile(s, 0.5), "p95": _percentile(s, 0.95)}
