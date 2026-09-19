import pytest

from svp.metrics import coverage, is_correct, majority_vote_accuracy, pass_at_k, pass_at_k_curve
from svp.models.coconut import CoconutAdapter
from svp.models.codi import CodiAdapter


def test_is_correct():
    assert is_correct(18.0, 18)
    assert not is_correct(18.5, 18)
    assert not is_correct(None, 18)
    assert is_correct(4.3333333, 4.33)      # rounded gold
    assert not is_correct(4.36, 4.3)


def test_pass_at_k():
    flags = [[True, False, False, False], [False] * 4, [True] * 4]
    assert pass_at_k(flags, 1) == pytest.approx((0.25 + 0 + 1) / 3)
    assert pass_at_k(flags, 4) == pytest.approx(2 / 3)
    assert pass_at_k(flags, 8) == pytest.approx(2 / 3)
    assert list(pass_at_k_curve(flags, 4)) == ["pass@1", "pass@2", "pass@4"]
    assert coverage(flags) == pytest.approx(2 / 3)


def test_majority_vote():
    answers = [[7.0, 7.0, 3.0], [None, None, 5.0]]
    assert majority_vote_accuracy(answers, [7.0, 5.0]) == 0.5


def test_answer_extraction():
    assert CoconutAdapter.extract_answer(None, " 16 - 3 - 4 = 9 # 1,234") == 1234.0
    assert CoconutAdapter.extract_answer(None, "no answer") is None
    assert CodiAdapter.extract_answer(None, "The answer is: -12.5.") == -12.5
    assert CodiAdapter.extract_answer(None, "The answer is: 1,000") == 1000.0
    assert CodiAdapter.extract_answer(None, "nothing") is None
