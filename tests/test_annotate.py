import json

from eval.annotate import annotate, load_rows, save_rows


def _rows():
    return [
        {
            "claim_id": i,
            "claim": f"claim {i}",
            "source_text": f"source {i}",
            "verifier_label": label,
            "gold_supported": None,
        }
        for i, label in enumerate(["supported", "partial", "unsupported", "supported"])
    ]


def _run(rows, answers):
    shown, saves = [], []
    it = iter(answers)
    annotate(rows, ask=lambda _prompt: next(it), show=shown.append, save=lambda: saves.append(1))
    return shown, saves


def test_yes_no_answers_are_saved_after_each_label():
    rows = _rows()
    shown, saves = _run(rows, ["y", "n", "y", "n"])
    assert sorted(r["gold_supported"] for r in rows) == [False, False, True, True]
    assert len(saves) == 4


def test_verifier_label_is_never_shown():
    rows = _rows()
    shown, _ = _run(rows, ["y", "y", "y", "y"])
    screen = "\n".join(shown)
    assert "claim 0" in screen and "source >>0<<" in screen
    assert (
        "verifier" not in screen.lower() and "partial" not in screen and "unsupported" not in screen
    )


def test_order_is_shuffled_not_the_stratified_label_order():
    rows = [
        {
            "claim_id": i,
            "claim": f"c{i}",
            "source_text": "s",
            "verifier_label": "x",
            "gold_supported": None,
        }
        for i in range(30)
    ]
    order = []
    annotate(rows, ask=lambda p: "y", show=lambda t: order.append(t), save=lambda: None)
    claims = [
        int(line.split("c", 1)[1].split()[0])
        for t in order
        for line in t.splitlines()
        if line.startswith("CLAIM: c")
    ]
    assert sorted(claims) == list(range(30)) and claims != list(range(30))


def test_skip_back_and_quit():
    rows = _rows()
    # label first as yes, go back and change it to no, skip the next, then quit
    shown, _ = _run(rows, ["y", "b", "n", "s", "q"])
    labelled = [r for r in rows if r["gold_supported"] is not None]
    assert len(labelled) == 1 and labelled[0]["gold_supported"] is False


def test_already_labelled_rows_are_not_asked_again():
    rows = _rows()
    for r in rows[:3]:
        r["gold_supported"] = True
    shown, _ = _run(rows, ["n"])
    assert sum("CLAIM:" in s for s in shown) == 1 and rows[3]["gold_supported"] is False


def test_invalid_answer_reprompts():
    rows = _rows()[:1]
    shown, _ = _run(rows, ["maybe", "", "y"])
    assert rows[0]["gold_supported"] is True


def test_load_and_save_roundtrip(tmp_path):
    path = tmp_path / "gold.jsonl"
    rows = _rows()
    rows[0]["gold_supported"] = True
    save_rows(path, rows)
    assert load_rows(path) == rows
    assert all(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def test_claim_numbers_are_marked_in_source():
    rows = [
        {
            "claim_id": 1,
            "claim": "Japan installed 46.1 thousand robots in 2023.",
            "source_text": (
                "In 2023 Japan installed 46,100 robots; 146.1 elsewhere; 46.1 thousand total."
            ),
            "verifier_label": "supported",
            "gold_supported": None,
        }
    ]
    shown, _ = _run(rows, ["y"])
    screen = shown[0]
    assert "In >>2023<< Japan" in screen and ">>46.1<< thousand total" in screen
    assert "146.1" in screen and ">>146.1" not in screen and "1>>46.1" not in screen
