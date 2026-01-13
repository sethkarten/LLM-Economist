#!/usr/bin/env python3
"""
Local test for distributed training components.

Tests:
1. SharedFileSystem operations
2. WorkerStatus and CoordinatorState serialization
3. Rollout file writing/reading

Run:
    python experiments/local_tests/test_distributed.py
"""

import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_worker_status():
    """Test WorkerStatus serialization."""
    from llm_economist.training.distributed import WorkerStatus

    status = WorkerStatus(
        worker_id="worker_1",
        status="collecting",
        rollouts_collected=10,
        last_heartbeat=time.time(),
        current_iteration=5,
        error_message=None,
    )

    d = status.to_dict()
    assert d["worker_id"] == "worker_1"
    assert d["status"] == "collecting"

    status2 = WorkerStatus.from_dict(d)
    assert status2.rollouts_collected == 10

    print("✓ WorkerStatus test passed")


def test_coordinator_state():
    """Test CoordinatorState serialization."""
    from llm_economist.training.distributed import CoordinatorState

    state = CoordinatorState(
        iteration=10,
        phase="collecting",
        model_version="abc123",
        num_workers=8,
        workers_ready=["worker_1", "worker_2"],
        rollouts_this_iteration=20,
        target_rollouts=64,
        start_time=time.time(),
        last_update=time.time(),
    )

    d = state.to_dict()
    assert d["iteration"] == 10
    assert d["phase"] == "collecting"
    assert len(d["workers_ready"]) == 2

    state2 = CoordinatorState.from_dict(d)
    assert state2.model_version == "abc123"
    assert state2.target_rollouts == 64

    print("✓ CoordinatorState test passed")


def test_shared_filesystem():
    """Test SharedFileSystem operations."""
    from llm_economist.training.distributed import (
        SharedFileSystem, WorkerStatus, CoordinatorState
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        fs = SharedFileSystem(tmpdir)

        # Test directory structure created
        assert (Path(tmpdir) / "model").exists()
        assert (Path(tmpdir) / "workers").exists()
        assert (Path(tmpdir) / "rollouts").exists()

        # Test coordinator state
        state = CoordinatorState(
            iteration=0,
            phase="waiting",
            model_version="test_v1",
            num_workers=4,
            workers_ready=[],
            rollouts_this_iteration=0,
            target_rollouts=32,
            start_time=time.time(),
            last_update=time.time(),
        )
        fs.write_coordinator_state(state)

        loaded_state = fs.read_coordinator_state()
        assert loaded_state is not None
        assert loaded_state.model_version == "test_v1"

        # Test worker status
        worker_status = WorkerStatus(
            worker_id="worker_test",
            status="idle",
            rollouts_collected=0,
            last_heartbeat=time.time(),
            current_iteration=-1,
        )
        fs.write_worker_status(worker_status)

        loaded_worker = fs.read_worker_status("worker_test")
        assert loaded_worker is not None
        assert loaded_worker.status == "idle"

        # Test all workers
        all_workers = fs.read_all_worker_statuses()
        assert len(all_workers) == 1

        # Test rollouts
        rollouts = [
            {"observation": {"tax_year": 1}, "reward": 0.1},
            {"observation": {"tax_year": 2}, "reward": 0.2},
        ]
        fs.write_rollouts(iteration=0, worker_id="worker_test", rollouts=rollouts)

        loaded_rollouts = fs.read_all_rollouts(iteration=0)
        assert len(loaded_rollouts) == 2
        assert loaded_rollouts[0]["reward"] == 0.1

    print("✓ SharedFileSystem test passed")


def test_model_versioning():
    """Test model version tracking."""
    from llm_economist.training.distributed import SharedFileSystem

    with tempfile.TemporaryDirectory() as tmpdir:
        fs = SharedFileSystem(tmpdir)

        # Initially no version
        assert fs.read_model_version() is None

        # Create a mock model directory
        model_dir = Path(tmpdir) / "mock_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"test": true}')
        (model_dir / "model.safetensors").write_text('fake_weights')

        # Write model with version
        fs.write_model(str(model_dir), "v1.0")

        # Check version
        assert fs.read_model_version() == "v1.0"

        # Check model path exists
        model_path = fs.get_model_path()
        assert Path(model_path).exists()
        assert (Path(model_path) / "config.json").exists()

    print("✓ Model versioning test passed")


def test_concurrent_file_access():
    """Test that file operations are atomic."""
    from llm_economist.training.distributed import (
        SharedFileSystem, WorkerStatus
    )
    import threading

    with tempfile.TemporaryDirectory() as tmpdir:
        fs = SharedFileSystem(tmpdir)

        errors = []
        iterations = 50

        def writer_thread(worker_id):
            try:
                for i in range(iterations):
                    status = WorkerStatus(
                        worker_id=worker_id,
                        status="collecting",
                        rollouts_collected=i,
                        last_heartbeat=time.time(),
                        current_iteration=i,
                    )
                    fs.write_worker_status(status)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(f"{worker_id}: {e}")

        def reader_thread():
            try:
                for _ in range(iterations):
                    statuses = fs.read_all_worker_statuses()
                    # Just check it doesn't crash
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"reader: {e}")

        # Start threads
        threads = []
        for i in range(4):
            t = threading.Thread(target=writer_thread, args=(f"worker_{i}",))
            threads.append(t)
            t.start()

        reader = threading.Thread(target=reader_thread)
        threads.append(reader)
        reader.start()

        # Wait for all
        for t in threads:
            t.join()

        if errors:
            print(f"Errors during concurrent access: {errors}")
            assert False, "Concurrent access test failed"

    print("✓ Concurrent file access test passed")


def run_all_tests():
    """Run all distributed component tests."""
    print("\n" + "="*50)
    print("Running Local Distributed Training Tests")
    print("="*50 + "\n")

    test_worker_status()
    test_coordinator_state()
    test_shared_filesystem()
    test_model_versioning()
    test_concurrent_file_access()

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()
