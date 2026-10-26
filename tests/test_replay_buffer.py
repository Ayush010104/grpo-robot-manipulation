"""Unit tests for MM-GRPO's persistent memory (advantages/replay_buffer.py) and the
extended k-NN baseline that consumes it (state_matched_baselines' extra_states/
extra_returns, advantages/group_baselines.py). Novel extension proposed 2026-10-26.
"""
import numpy as np

from advantages.group_baselines import state_matched_baselines
from advantages.replay_buffer import StateReturnBuffer


def test_buffer_add_and_len():
    buf = StateReturnBuffer(capacity=100)
    buf.add(np.zeros((5, 3)), np.arange(5, dtype=float), iteration=0)
    assert len(buf) == 5


def test_buffer_respects_capacity_fifo():
    buf = StateReturnBuffer(capacity=10)
    for it in range(5):
        buf.add(np.zeros((4, 2)), np.full(4, it, dtype=float), iteration=it)
    assert len(buf) == 10
    # the oldest entries (iteration 0) should have been evicted
    _, returns = buf.sample_candidates(current_iteration=4, max_age=None)
    assert 0.0 not in returns


def test_buffer_max_age_filters_old_entries():
    buf = StateReturnBuffer(capacity=1000)
    buf.add(np.zeros((1, 2)), np.array([111.0]), iteration=0)
    buf.add(np.zeros((1, 2)), np.array([222.0]), iteration=10)
    states, returns = buf.sample_candidates(current_iteration=10, max_age=2)
    assert 222.0 in returns
    assert 111.0 not in returns


def test_buffer_empty_returns_empty_arrays():
    buf = StateReturnBuffer()
    states, returns = buf.sample_candidates(current_iteration=0)
    assert states.shape[0] == 0 and returns.shape[0] == 0


def test_state_matched_baselines_uses_extra_candidates_when_group_is_too_small():
    # A single-rollout "group" has no in-group siblings at all -- without extra
    # candidates, this must fall back to the trajectory's own mean (the existing
    # no-neighbours path). With extra candidates, it should use them instead.
    states = [np.array([[0.0, 0.0], [0.0, 0.0]])]
    rtg = [np.array([10.0, 10.0])]

    adv_no_extra = state_matched_baselines(states, rtg, k=1, time_window=0)
    np.testing.assert_allclose(adv_no_extra[0], [0.0, 0.0])  # rtg - own mean = 0

    extra_states = np.array([[0.0, 0.0]])
    extra_returns = np.array([4.0])
    adv_with_extra = state_matched_baselines(
        states, rtg, k=1, time_window=0, extra_states=extra_states, extra_returns=extra_returns
    )
    np.testing.assert_allclose(adv_with_extra[0], [6.0, 6.0])  # 10 - 4


def test_state_matched_baselines_extra_candidates_prefer_nearest():
    states = [np.array([[0.0, 0.0]])]
    rtg = [np.array([5.0])]
    extra_states = np.array([[0.0, 0.0], [50.0, 50.0]])  # one near, one far
    extra_returns = np.array([1.0, 999.0])
    adv = state_matched_baselines(
        states, rtg, k=1, time_window=0, extra_states=extra_states, extra_returns=extra_returns
    )
    np.testing.assert_allclose(adv[0], [4.0])  # 5 - 1 (the near one), not 5 - 999
