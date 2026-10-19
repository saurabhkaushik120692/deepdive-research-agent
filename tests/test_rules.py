from app.graph.rules import should_loop, thin_sub_questions


def test_thin_detection():
    assert thin_sub_questions({1: 3, 2: 1, 3: 0}, min_supported=2) == [2, 3]


def test_loop_cap():
    assert should_loop([2], round_=0) and should_loop([2], round_=1)
    assert not should_loop([2], round_=2)
    assert not should_loop([], round_=0)
