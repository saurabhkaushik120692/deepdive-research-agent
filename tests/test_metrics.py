from app.models import Verdict
from eval.metrics import (
    agreement,
    catch_rate,
    quote_validity_rate,
    summarize,
    wrongful_rejection_rate,
)


def test_catch_rate():
    assert catch_rate(["unsupported", "unsupported", "supported", "partial"]) == 0.5
    assert catch_rate([]) is None


def test_wrongful_rejection():
    assert wrongful_rejection_rate(["supported", "unsupported", "supported", "supported"]) == 0.25


def test_quote_validity_ignores_empty_quotes():
    vs = [
        Verdict(claim_id=1, label="supported", quote="a", quote_found=True),
        Verdict(claim_id=2, label="unsupported", quote="b", quote_found=False),
        Verdict(claim_id=3, label="unsupported", quote="", quote_found=False),
    ]
    assert quote_validity_rate(vs) == 0.5


def test_agreement_perfect():
    assert agreement([True, False, True], [True, False, True]) == (1.0, 1.0)


def test_agreement_kappa_known_value():
    acc, kappa = agreement([True, True, False, False], [True, False, False, False])
    assert acc == 0.75 and round(kappa, 3) == 0.5


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["p50"] == 2.5 and s["mean"] == 2.5 and 3.8 <= s["p95"] <= 4.0
    assert summarize([]) == {"mean": None, "p50": None, "p95": None}
