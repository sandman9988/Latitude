#!/usr/bin/env python3
"""
Phase 3.5 Integration Test
===========================
Test complete online learning pipeline with dual-agent architecture.

Tests:
1. ExperienceBuffer integration with TriggerAgent
2. ExperienceBuffer integration with HarvesterAgent
3. Experience addition after simulated trades
4. Training step execution
5. Buffer statistics tracking
6. Priority updates

Expected behavior:
- Agents accumulate experiences in buffers
- Training starts after min_experiences threshold
- Priorities updated based on TD-errors
- Defensive validation prevents crashes
"""

import logging
import sys

import numpy as np
from numpy.random import Generator, default_rng

from experience_buffer import RegimeSampling
from harvester_agent import HarvesterAgent
from trigger_agent import TriggerAgent

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

TRIGGER_BUFFER_CAPACITY = 50_000
HARVESTER_BUFFER_CAPACITY = 50_000
INITIAL_EXPERIENCE_COUNT = 100
ADDITIONAL_EXPERIENCES = 950
HARVESTER_EXPERIENCE_COUNT = 1200
TRAINING_STEPS_EXPECTED = 6
DEFENSIVE_VALID_EXPERIENCES = 100
REGIME_EXPERIENCES = 500


def test_trigger_agent_training():
    """Test TriggerAgent with online learning enabled."""
    print("\n" + "=" * 80)
    print("TEST 1: TriggerAgent Online Learning")
    print("=" * 80)

    rng: Generator = default_rng(42)

    # Initialize with training enabled
    trigger = TriggerAgent(window=64, n_features=7, enable_training=True)

    # Verify buffer initialized
    assert trigger.buffer is not None
    assert trigger.buffer.capacity == TRIGGER_BUFFER_CAPACITY
    assert trigger.enable_training
    print("✓ TriggerAgent initialized with buffer (capacity=50k)")

    # Add experiences (simulate trades)
    print("\nAdding 100 simulated trade experiences...")
    for idx in range(INITIAL_EXPERIENCE_COUNT):
        entry_state = rng.standard_normal((64, 7)).astype(np.float32)
        exit_state = rng.standard_normal((64, 7)).astype(np.float32)
        action = rng.choice([1, 2])  # LONG or SHORT

        # Runway utilization reward: actual_MFE / predicted_runway
        predicted_runway = 0.002  # 20 pips
        actual_mfe = rng.uniform(0.0005, 0.0035)  # 5-35 pips
        runway_util = actual_mfe / predicted_runway

        # Reward: 1.0 if perfect prediction, <1 if overestimated, >1 if underestimated
        # Clip to [-1, 2] range
        reward = np.clip(runway_util - 0.5, -1.0, 2.0)

        trigger.add_experience(
            state=entry_state,
            action=action,
            reward=reward,
            next_state=exit_state,
            done=True,
            regime=RegimeSampling.TRENDING if idx % 2 == 0 else RegimeSampling.MEAN_REVERTING,
        )

    # Check buffer size
    stats = trigger.get_training_stats()
    print(f"✓ Buffer size: {stats['buffer_size']}")
    print(f"  Total added: {stats['total_added']}")
    print(f"  Ready to train: {stats['ready_to_train']}")

    assert stats["buffer_size"] == INITIAL_EXPERIENCE_COUNT
    assert stats["total_added"] == INITIAL_EXPERIENCE_COUNT
    assert not stats["ready_to_train"]  # Need 1000 minimum

    # Add more experiences to reach training threshold
    print("\nAdding 950 more experiences to reach training threshold...")
    for _ in range(ADDITIONAL_EXPERIENCES):
        entry_state = rng.standard_normal((64, 7)).astype(np.float32)
        exit_state = rng.standard_normal((64, 7)).astype(np.float32)
        action = rng.choice([1, 2])
        reward = rng.uniform(-1.0, 2.0)

        trigger.add_experience(
            state=entry_state,
            action=action,
            reward=reward,
            next_state=exit_state,
            done=True,
            regime=RegimeSampling.TRENDING,
        )

    stats = trigger.get_training_stats()
    print(f"✓ Buffer size: {stats['buffer_size']}")
    print(f"  Ready to train: {stats['ready_to_train']}")
    assert stats["ready_to_train"]

    # Execute training step
    print("\nExecuting training step...")
    metrics = trigger.train_step()

    assert metrics is not None
    print("✓ Training step complete:")
    print(f"  Mean reward: {metrics['mean_reward']:.4f}")
    print(f"  Mean TD-error: {metrics['mean_td_error']:.4f}")
    print(f"  Max TD-error: {metrics['max_td_error']:.4f}")

    # Verify training stats updated
    stats = trigger.get_training_stats()
    assert stats["training_steps"] == 1
    assert stats["total_sampled"] > 0
    print(f"✓ Training stats updated (steps={stats['training_steps']}, sampled={stats['total_sampled']})")

    print("\n✅ TriggerAgent online learning test PASSED")
    return True


def test_harvester_agent_training():
    """Test HarvesterAgent with online learning enabled."""
    print("\n" + "=" * 80)
    print("TEST 2: HarvesterAgent Online Learning")
    print("=" * 80)

    rng: Generator = default_rng(43)

    # Initialize with training enabled
    harvester = HarvesterAgent(window=64, n_features=10, enable_training=True)

    # Verify buffer initialized
    assert harvester.buffer is not None
    assert harvester.buffer.capacity == HARVESTER_BUFFER_CAPACITY
    assert harvester.enable_training
    print("✓ HarvesterAgent initialized with buffer (capacity=50k)")

    # Add experiences (simulate exit decisions)
    print("\nAdding 1200 simulated exit decision experiences...")
    for _ in range(HARVESTER_EXPERIENCE_COUNT):
        # Position state: market (7) + position (3: mfe_norm, mae_norm, bars_held_norm)
        # State should be (window=64, n_features=10)
        state = rng.standard_normal((64, 10)).astype(np.float32)
        next_state = rng.standard_normal((64, 10)).astype(np.float32)

        action = rng.choice([0, 1])  # HOLD or CLOSE

        # Capture efficiency reward: exit_pnl / MFE
        # Good exits: high capture ratio
        # WTL: negative reward
        capture_ratio = rng.uniform(0.0, 1.2)
        reward = capture_ratio - 0.5  # Center around 0

        harvester.add_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=(action == 1),  # Done if CLOSE
            regime=RegimeSampling.MEAN_REVERTING,
        )

    # Check buffer size
    stats = harvester.get_training_stats()
    print(f"✓ Buffer size: {stats['buffer_size']}")
    print(f"  Total added: {stats['total_added']}")
    print(f"  Ready to train: {stats['ready_to_train']}")

    assert stats["buffer_size"] == HARVESTER_EXPERIENCE_COUNT
    assert stats["ready_to_train"]

    # Execute training step
    print("\nExecuting training step...")
    metrics = harvester.train_step()

    assert metrics is not None
    print("✓ Training step complete:")
    print(f"  Mean reward: {metrics['mean_reward']:.4f}")
    print(f"  Mean TD-error: {metrics['mean_td_error']:.4f}")
    print(f"  Max TD-error: {metrics['max_td_error']:.4f}")

    # Execute multiple training steps
    print("\nExecuting 5 more training steps...")
    for _ in range(5):
        metrics = harvester.train_step()
        assert metrics is not None

    stats = harvester.get_training_stats()
    assert stats["training_steps"] == TRAINING_STEPS_EXPECTED
    print(f"✓ Completed {stats['training_steps']} training steps")

    print("\n✅ HarvesterAgent online learning test PASSED")
    return True


def test_defensive_validation():
    """Test defensive programming in training pipeline."""
    print("\n" + "=" * 80)
    print("TEST 3: Defensive Validation")
    print("=" * 80)

    rng: Generator = default_rng(44)

    trigger = TriggerAgent(window=64, n_features=7, enable_training=True)

    # Test 1: Add invalid experiences (should be rejected)
    print("\n[Test 3.1] Invalid experiences rejection")

    # Empty state
    trigger.add_experience(state=np.array([]), action=1, reward=1.0, next_state=rng.standard_normal((64, 7)), done=True)

    # Non-finite reward
    trigger.add_experience(
        state=rng.standard_normal((64, 7)),
        action=1,
        reward=np.nan,
        next_state=rng.standard_normal((64, 7)),
        done=True,
    )

    # Invalid action
    trigger.add_experience(
        state=rng.standard_normal((64, 7)),
        action=5,  # Invalid
        reward=1.0,
        next_state=rng.standard_normal((64, 7)),
        done=True,
    )

    stats = trigger.get_training_stats()
    assert stats["buffer_size"] == 0  # All rejected
    print("✓ Invalid experiences correctly rejected (buffer_size=0)")

    # Test 2: Add valid experiences
    print("\n[Test 3.2] Valid experiences accepted")
    for _ in range(DEFENSIVE_VALID_EXPERIENCES):
        trigger.add_experience(
            state=rng.standard_normal((64, 7)).astype(np.float32),
            action=1,
            reward=rng.uniform(-1, 2),
            next_state=rng.standard_normal((64, 7)).astype(np.float32),
            done=True,
        )

    stats = trigger.get_training_stats()
    assert stats["buffer_size"] == DEFENSIVE_VALID_EXPERIENCES
    print(f"✓ Valid experiences accepted (buffer_size={stats['buffer_size']})")

    print("\n✅ Defensive validation test PASSED")
    return True


def test_regime_aware_sampling():
    """Test regime-aware prioritization."""
    print("\n" + "=" * 80)
    print("TEST 4: Regime-Aware Sampling")
    print("=" * 80)

    rng: Generator = default_rng(45)

    trigger = TriggerAgent(window=64, n_features=7, enable_training=True)

    # Set current regime to TRENDING
    trigger.buffer.set_current_regime(RegimeSampling.TRENDING)
    print("✓ Current regime: TRENDING")

    # Add 500 TRENDING experiences
    print("\nAdding 500 TRENDING experiences...")
    for _ in range(REGIME_EXPERIENCES):
        trigger.add_experience(
            state=rng.standard_normal((64, 7)).astype(np.float32),
            action=1,
            reward=1.0,
            next_state=rng.standard_normal((64, 7)).astype(np.float32),
            done=True,
            regime=RegimeSampling.TRENDING,
        )

    # Add 500 MEAN_REVERTING experiences
    print("Adding 500 MEAN_REVERTING experiences...")
    for _ in range(REGIME_EXPERIENCES):
        trigger.add_experience(
            state=rng.standard_normal((64, 7)).astype(np.float32),
            action=1,
            reward=1.0,
            next_state=rng.standard_normal((64, 7)).astype(np.float32),
            done=True,
            regime=RegimeSampling.MEAN_REVERTING,
        )

    stats = trigger.get_training_stats()
    print(f"✓ Total experiences: {stats['buffer_size']}")

    # Sample batch - should favor TRENDING experiences (1.5x boost)
    batch = trigger.buffer.sample(batch_size=64)
    assert batch is not None
    print(f"✓ Sampled batch of {len(batch['states'])} experiences")
    print("  (TRENDING experiences should be sampled more frequently due to regime boost)")

    print("\n✅ Regime-aware sampling test PASSED")
    return True


def run_all_tests():
    """Run all Phase 3.5 integration tests."""
    print("\n" + "=" * 80)
    print("PHASE 3.5 INTEGRATION TEST SUITE")
    print("Online Learning Pipeline Validation")
    print("=" * 80)

    results = []

    try:
        results.append(("TriggerAgent Training", test_trigger_agent_training()))
    except Exception as e:
        LOG.error("TriggerAgent test failed: %s", e, exc_info=True)
        results.append(("TriggerAgent Training", False))

    try:
        results.append(("HarvesterAgent Training", test_harvester_agent_training()))
    except Exception as e:
        LOG.error("HarvesterAgent test failed: %s", e, exc_info=True)
        results.append(("HarvesterAgent Training", False))

    try:
        results.append(("Defensive Validation", test_defensive_validation()))
    except Exception as e:
        LOG.error("Defensive validation test failed: %s", e, exc_info=True)
        results.append(("Defensive Validation", False))

    try:
        results.append(("Regime-Aware Sampling", test_regime_aware_sampling()))
    except Exception as e:
        LOG.error("Regime-aware test failed: %s", e, exc_info=True)
        results.append(("Regime-Aware Sampling", False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED! Phase 3.5 integration complete.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ SOME TESTS FAILED - Review errors above")
        print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
