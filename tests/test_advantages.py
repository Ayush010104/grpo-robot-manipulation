"""Unit tests for advantages/group_baselines.py -- these are the core-contribution
functions (SM-GRPO, Idea A) so they get correctness tests independent of any env/policy.
"""
import numpy as np
import pytest

from advantages.group_baselines import (
    compute_returns_to_go,
    fuse_trajectory_and_step,
    is_degenerate_group,
    leave_one_out_advantages,
    robust_scale,
    robust_trajectory_advantages,
    state_matched_baselines,
    trajectory_advantages,
)


def test_trajectory_advantages_basic():
    returns = np.array([1.0, 2.0, 3.0, 4.0])
    adv = trajectory_advantages(returns)
    assert np.isclose(adv.mean(), 0.0, atol=1e-6)
    assert np.isclose(adv.std(), 1.0, atol=1e-2)


def test_tiny_std_amplifies_more_under_vanilla_than_robust():
    # Tiny real spread in returns -> vanilla std-normalisation divides by a tiny number
    # and over-amplifies it; robust normalisation floors the denominator (Section 4.2 #2).
    returns = np.array([1.0, 1.0 + 1e-4, 1.0 - 1e-4, 1.0])
    vanilla_adv = trajectory_advantages(returns)
    robust_adv = robust_trajectory_advantages(returns, sigma_min=1e-2)
    assert np.abs(vanilla_adv).max() > 10 * np.abs(robust_adv).max()


def test_robust_trajectory_advantages_is_bounded():
    returns = np.array([1.0, 1.0 + 1e-6, 1.0 - 1e-6, 1.0])
    adv = robust_trajectory_advantages(returns, sigma_min=1e-2)
    assert np.abs(adv).max() < 1.0  # sigma_min floor prevents blow-up


def test_robust_scale_mad_ignores_one_outlier():
    returns = np.array([1.0, 1.0, 1.0, 100.0])
    scale = robust_scale(returns, sigma_min=1e-3, method="mad")
    assert scale < 5.0  # MAD is robust to the single outlier; plain std would not be


def test_is_degenerate_group_all_success_and_all_fail():
    assert is_degenerate_group(np.array([1.0, 1.0, 1.0]))
    assert is_degenerate_group(np.array([0.0, 0.0, 0.0]))
    assert not is_degenerate_group(np.array([0.0, 1.0, 0.0]))


def test_leave_one_out_excludes_self():
    returns = np.array([0.0, 0.0, 0.0, 10.0])
    adv = leave_one_out_advantages(returns)
    # rollout 3's baseline is mean of [0,0,0] = 0, so its advantage should be the largest
    assert adv.argmax() == 3


def test_compute_returns_to_go_undiscounted():
    rewards = np.array([1.0, 1.0, 1.0])
    rtg = compute_returns_to_go(rewards, gamma=1.0)
    np.testing.assert_allclose(rtg, [3.0, 2.0, 1.0])


def test_compute_returns_to_go_discounted():
    rewards = np.array([1.0, 1.0])
    rtg = compute_returns_to_go(rewards, gamma=0.5)
    np.testing.assert_allclose(rtg, [1.5, 1.0])


def test_state_matched_baselines_shapes_and_zero_variance_case():
    # Two rollouts with identical states at every step -> baseline should exactly match
    # the other rollout's return-to-go at that step (k=1).
    states = [np.zeros((3, 2)), np.zeros((3, 2))]
    rtg = [np.array([3.0, 2.0, 1.0]), np.array([30.0, 20.0, 10.0])]
    adv = state_matched_baselines(states, rtg, k=1, time_window=0)
    assert len(adv) == 2
    assert adv[0].shape == (3,)
    # rollout 0's baseline comes from rollout 1 (its only "other" neighbour)
    np.testing.assert_allclose(adv[0], rtg[0] - rtg[1])
    np.testing.assert_allclose(adv[1], rtg[1] - rtg[0])


def test_state_matched_baselines_picks_nearest_state():
    # rollout 0 query state is close to rollout 1's step 0 (state=[0,0]) and far from
    # rollout 2's step 0 (state=[100,100]) -> baseline should come from rollout 1.
    states = [
        np.array([[0.0, 0.0]]),
        np.array([[0.1, 0.1]]),
        np.array([[100.0, 100.0]]),
    ]
    rtg = [np.array([5.0]), np.array([1.0]), np.array([999.0])]
    adv = state_matched_baselines(states, rtg, k=1, time_window=0)
    np.testing.assert_allclose(adv[0], [5.0 - 1.0])


def test_fuse_trajectory_and_step_weighting():
    traj_adv = 2.0
    step_adv = np.array([0.0, 1.0, 2.0])
    fused_full_traj = fuse_trajectory_and_step(traj_adv, step_adv, alpha=1.0)
    np.testing.assert_allclose(fused_full_traj, [2.0, 2.0, 2.0])
    fused_full_step = fuse_trajectory_and_step(traj_adv, step_adv, alpha=0.0)
    np.testing.assert_allclose(fused_full_step, step_adv)
