#!/usr/bin/env python3
"""
Distributed rollout collection for B200 cluster training.

Architecture:
- Coordinator: Manages training loop, aggregates rollouts, updates model
- Workers: Run on B200s, collect rollouts, sync with coordinator

Communication via shared filesystem (NFS) or HTTP (optional).

Usage:
    # Start coordinator (on head node)
    python -m llm_economist.training.distributed \
        --mode coordinator \
        --shared-dir /shared/rl_training \
        --num-workers 8 \
        --output models/planner-rl

    # Start worker (on B200 node)
    python -m llm_economist.training.distributed \
        --mode worker \
        --worker-id 1 \
        --shared-dir /shared/rl_training \
        --planner-model qwen3-4b \
        --worker-model qwen3-30b-a3b
"""

import argparse
import json
import os
import time
import uuid
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

import torch
import numpy as np


@dataclass
class WorkerStatus:
    """Status of a distributed worker."""
    worker_id: str
    status: str  # "idle", "collecting", "done", "error"
    rollouts_collected: int
    last_heartbeat: float
    current_iteration: int
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkerStatus":
        return cls(**d)


@dataclass
class CoordinatorState:
    """State of the coordinator."""
    iteration: int
    phase: str  # "waiting", "collecting", "training", "done"
    model_version: str
    num_workers: int
    workers_ready: List[str]
    rollouts_this_iteration: int
    target_rollouts: int
    start_time: float
    last_update: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoordinatorState":
        return cls(**d)


class SharedFileSystem:
    """
    Handles communication via shared filesystem (NFS).

    Directory structure:
        shared_dir/
        ├── coordinator_state.json    # Current coordinator state
        ├── model/                    # Current model weights
        │   └── checkpoint/
        ├── workers/                  # Worker status files
        │   ├── worker_1.json
        │   └── worker_2.json
        ├── rollouts/                 # Collected rollouts per iteration
        │   ├── iter_001/
        │   │   ├── worker_1_rollouts.json
        │   │   └── worker_2_rollouts.json
        │   └── iter_002/
        └── logs/                     # Training logs
    """

    def __init__(self, shared_dir: str):
        self.shared_dir = Path(shared_dir)
        self._setup_directories()

    def _setup_directories(self):
        """Create directory structure."""
        (self.shared_dir / "model").mkdir(parents=True, exist_ok=True)
        (self.shared_dir / "workers").mkdir(exist_ok=True)
        (self.shared_dir / "rollouts").mkdir(exist_ok=True)
        (self.shared_dir / "logs").mkdir(exist_ok=True)

    # Coordinator state
    def write_coordinator_state(self, state: CoordinatorState):
        """Write coordinator state atomically."""
        path = self.shared_dir / "coordinator_state.json"
        tmp_path = path.with_suffix(".tmp")

        with open(tmp_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        shutil.move(tmp_path, path)

    def read_coordinator_state(self) -> Optional[CoordinatorState]:
        """Read coordinator state."""
        path = self.shared_dir / "coordinator_state.json"
        if not path.exists():
            return None

        try:
            with open(path) as f:
                return CoordinatorState.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            return None

    # Worker status
    def write_worker_status(self, status: WorkerStatus):
        """Write worker status."""
        path = self.shared_dir / "workers" / f"{status.worker_id}.json"
        tmp_path = path.with_suffix(".tmp")

        with open(tmp_path, "w") as f:
            json.dump(status.to_dict(), f, indent=2)

        shutil.move(tmp_path, path)

    def read_worker_status(self, worker_id: str) -> Optional[WorkerStatus]:
        """Read worker status."""
        path = self.shared_dir / "workers" / f"{worker_id}.json"
        if not path.exists():
            return None

        try:
            with open(path) as f:
                return WorkerStatus.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            return None

    def read_all_worker_statuses(self) -> List[WorkerStatus]:
        """Read all worker statuses."""
        workers_dir = self.shared_dir / "workers"
        statuses = []

        for path in workers_dir.glob("*.json"):
            if path.suffix == ".tmp":
                continue
            try:
                with open(path) as f:
                    statuses.append(WorkerStatus.from_dict(json.load(f)))
            except (json.JSONDecodeError, KeyError):
                continue

        return statuses

    # Rollouts
    def write_rollouts(self, iteration: int, worker_id: str, rollouts: List[Dict]):
        """Write rollouts for an iteration."""
        iter_dir = self.shared_dir / "rollouts" / f"iter_{iteration:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        path = iter_dir / f"{worker_id}_rollouts.json"
        with open(path, "w") as f:
            json.dump(rollouts, f)

    def read_all_rollouts(self, iteration: int) -> List[Dict]:
        """Read all rollouts for an iteration."""
        iter_dir = self.shared_dir / "rollouts" / f"iter_{iteration:04d}"
        if not iter_dir.exists():
            return []

        all_rollouts = []
        for path in iter_dir.glob("*_rollouts.json"):
            try:
                with open(path) as f:
                    rollouts = json.load(f)
                    all_rollouts.extend(rollouts)
            except (json.JSONDecodeError, KeyError):
                continue

        return all_rollouts

    # Model weights
    def write_model(self, model_dir: str, version: str):
        """Copy model to shared directory."""
        dest = self.shared_dir / "model" / "checkpoint"

        # Remove old checkpoint
        if dest.exists():
            shutil.rmtree(dest)

        # Copy new checkpoint
        shutil.copytree(model_dir, dest)

        # Write version marker
        with open(self.shared_dir / "model" / "version.txt", "w") as f:
            f.write(version)

    def read_model_version(self) -> Optional[str]:
        """Read current model version."""
        path = self.shared_dir / "model" / "version.txt"
        if not path.exists():
            return None
        return path.read_text().strip()

    def get_model_path(self) -> str:
        """Get path to current model checkpoint."""
        return str(self.shared_dir / "model" / "checkpoint")


class DistributedCoordinator:
    """
    Coordinator for distributed RL training.

    Responsibilities:
    - Manage training iterations
    - Signal workers to collect rollouts
    - Aggregate rollouts and perform training updates
    - Save checkpoints
    """

    def __init__(
        self,
        shared_fs: SharedFileSystem,
        num_workers: int,
        output_dir: str,
        config: Dict[str, Any],
    ):
        self.fs = shared_fs
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.config = config

        self.iteration = 0
        self.model_version = self._generate_version()

        # Training components (lazy loaded)
        self.policy = None
        self.trainer = None

    def _generate_version(self) -> str:
        """Generate unique model version."""
        return hashlib.md5(
            f"{time.time()}_{uuid.uuid4()}".encode()
        ).hexdigest()[:12]

    def initialize(self):
        """Initialize coordinator and publish initial state."""
        print(f"Initializing coordinator...")
        print(f"  Shared directory: {self.fs.shared_dir}")
        print(f"  Expected workers: {self.num_workers}")
        print(f"  Output directory: {self.output_dir}")

        # Load or create policy
        from .rl_trainer import PlannerPolicy, RLConfig, REINFORCETrainer

        rl_config = RLConfig(**self.config)

        self.policy = PlannerPolicy(
            self.config.get("planner_model", "Qwen/Qwen3-4B-Instruct"),
            rl_config
        )
        self.policy.load_model()

        # Save initial model to shared location
        initial_model_dir = os.path.join(self.output_dir, "initial")
        self.policy.save(initial_model_dir)
        self.fs.write_model(initial_model_dir, self.model_version)

        # Create trainer
        self.trainer = REINFORCETrainer(self.policy, rl_config, self.output_dir)
        self.trainer.setup_optimizer()

        # Publish initial state
        state = CoordinatorState(
            iteration=0,
            phase="waiting",
            model_version=self.model_version,
            num_workers=self.num_workers,
            workers_ready=[],
            rollouts_this_iteration=0,
            target_rollouts=self.config.get("batch_size", 64),
            start_time=time.time(),
            last_update=time.time(),
        )
        self.fs.write_coordinator_state(state)

        print("Coordinator initialized. Waiting for workers...")

    def wait_for_workers(self, timeout: float = 300):
        """Wait for all workers to connect."""
        start = time.time()

        while time.time() - start < timeout:
            statuses = self.fs.read_all_worker_statuses()
            active = [s for s in statuses if s.status in ["idle", "collecting"]]

            if len(active) >= self.num_workers:
                print(f"All {self.num_workers} workers connected!")
                return True

            print(f"Waiting for workers: {len(active)}/{self.num_workers}")
            time.sleep(5)

        raise TimeoutError(f"Only {len(active)} workers connected within {timeout}s")

    def run_training_loop(self, num_iterations: int):
        """Main training loop."""
        from .rl_trainer import RolloutBatch, Rollout

        print(f"\n{'='*60}")
        print("Starting distributed training loop")
        print(f"{'='*60}\n")

        for iteration in range(num_iterations):
            self.iteration = iteration
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

            # Phase 1: Signal workers to collect rollouts
            print("Phase 1: Signaling workers to collect rollouts...")
            state = CoordinatorState(
                iteration=iteration,
                phase="collecting",
                model_version=self.model_version,
                num_workers=self.num_workers,
                workers_ready=[],
                rollouts_this_iteration=0,
                target_rollouts=self.config.get("batch_size", 64),
                start_time=time.time(),
                last_update=time.time(),
            )
            self.fs.write_coordinator_state(state)

            # Phase 2: Wait for rollouts
            print("Phase 2: Waiting for rollouts...")
            rollouts = self._wait_for_rollouts(iteration)
            print(f"  Collected {len(rollouts)} rollouts")

            # Phase 3: Training update
            print("Phase 3: Performing training update...")
            rollout_objs = [Rollout.from_dict(r) for r in rollouts]
            batch = RolloutBatch.from_rollouts(
                rollout_objs,
                use_group_baseline=self.config.get("use_group_baseline", True),
            )

            metrics = self.trainer.train_step(batch)
            self._log_metrics(iteration, metrics)

            # Phase 4: Update model and signal ready
            print("Phase 4: Updating shared model...")
            self.model_version = self._generate_version()
            checkpoint_dir = os.path.join(self.output_dir, f"iter_{iteration:04d}")
            self.policy.save(checkpoint_dir)
            self.fs.write_model(checkpoint_dir, self.model_version)

            # Update state to waiting
            state.phase = "waiting"
            state.model_version = self.model_version
            state.last_update = time.time()
            self.fs.write_coordinator_state(state)

            print(f"  Metrics: loss={metrics['loss/total']:.4f}, reward={metrics['reward/mean']:.4f}")

        # Final state
        state.phase = "done"
        self.fs.write_coordinator_state(state)
        print("\nTraining complete!")

    def _wait_for_rollouts(
        self,
        iteration: int,
        timeout: float = 3600,  # 1 hour
        poll_interval: float = 10,
    ) -> List[Dict]:
        """Wait for workers to collect rollouts."""
        target = self.config.get("batch_size", 64)
        start = time.time()

        while time.time() - start < timeout:
            rollouts = self.fs.read_all_rollouts(iteration)

            if len(rollouts) >= target:
                return rollouts[:target]  # Return exactly target amount

            # Check worker statuses
            statuses = self.fs.read_all_worker_statuses()
            done_workers = [s for s in statuses if s.status == "done" and s.current_iteration == iteration]

            print(f"  Rollouts: {len(rollouts)}/{target}, Workers done: {len(done_workers)}/{self.num_workers}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Rollout collection timed out after {timeout}s")

    def _log_metrics(self, iteration: int, metrics: Dict[str, float]):
        """Log metrics to file."""
        log_path = self.fs.shared_dir / "logs" / f"metrics.jsonl"

        entry = {
            "iteration": iteration,
            "timestamp": time.time(),
            **metrics
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class DistributedWorker:
    """
    Worker for distributed rollout collection.

    Runs on a B200 node:
    - Loads planner model from shared location
    - Loads worker model (for agent simulation)
    - Collects rollouts when signaled by coordinator
    - Writes rollouts to shared storage
    """

    def __init__(
        self,
        worker_id: str,
        shared_fs: SharedFileSystem,
        planner_model: str,
        worker_model: str,
        config: Dict[str, Any],
    ):
        self.worker_id = worker_id
        self.fs = shared_fs
        self.planner_model = planner_model
        self.worker_model = worker_model
        self.config = config

        self.policy = None
        self.collector = None
        self.current_model_version = None

    def initialize(self):
        """Initialize worker and register with coordinator."""
        print(f"Initializing worker {self.worker_id}...")

        # Register with coordinator
        status = WorkerStatus(
            worker_id=self.worker_id,
            status="idle",
            rollouts_collected=0,
            last_heartbeat=time.time(),
            current_iteration=-1,
        )
        self.fs.write_worker_status(status)

        # Wait for coordinator to be ready
        print("Waiting for coordinator...")
        self._wait_for_coordinator()

        # Load models
        print("Loading models...")
        self._load_models()

        print(f"Worker {self.worker_id} initialized!")

    def _wait_for_coordinator(self, timeout: float = 300):
        """Wait for coordinator to publish initial state."""
        start = time.time()

        while time.time() - start < timeout:
            state = self.fs.read_coordinator_state()
            if state is not None:
                return

            time.sleep(5)

        raise TimeoutError("Coordinator not found")

    def _load_models(self):
        """Load planner policy from shared location."""
        from .rl_trainer import PlannerPolicy, RLConfig, RolloutCollector

        # Load policy from shared model
        model_path = self.fs.get_model_path()
        rl_config = RLConfig(**self.config)

        self.policy = PlannerPolicy(model_path, rl_config)
        self.policy.load_model()
        self.current_model_version = self.fs.read_model_version()

        # Create rollout collector
        self.collector = RolloutCollector(
            self.worker_model,
            rl_config,
            seed=hash(self.worker_id) % (2**32),
        )

    def _maybe_reload_model(self):
        """Reload model if coordinator has updated it."""
        current_version = self.fs.read_model_version()

        if current_version != self.current_model_version:
            print(f"  Model updated: {self.current_model_version} -> {current_version}")
            self._load_models()

    def run(self):
        """Main worker loop."""
        print(f"\nWorker {self.worker_id} starting main loop...")

        while True:
            # Check coordinator state
            state = self.fs.read_coordinator_state()

            if state is None:
                time.sleep(5)
                continue

            if state.phase == "done":
                print("Coordinator signaled done. Exiting.")
                break

            if state.phase == "collecting" and state.iteration > self._get_last_iteration():
                # Time to collect rollouts
                self._collect_rollouts_for_iteration(state)

            # Update heartbeat
            self._update_status("idle")

            time.sleep(5)

    def _get_last_iteration(self) -> int:
        """Get last iteration we processed."""
        status = self.fs.read_worker_status(self.worker_id)
        return status.current_iteration if status else -1

    def _collect_rollouts_for_iteration(self, state: CoordinatorState):
        """Collect rollouts for a training iteration."""
        iteration = state.iteration
        rollouts_per_worker = max(1, state.target_rollouts // state.num_workers)

        print(f"\n  Collecting {rollouts_per_worker} rollouts for iteration {iteration}...")

        # Maybe reload model
        self._maybe_reload_model()

        # Update status
        self._update_status("collecting", iteration=iteration)

        # Collect rollouts
        rollouts = self.collector.collect(
            policy=self.policy,
            num_rollouts=rollouts_per_worker,
        )

        # Write rollouts
        rollout_dicts = [r.to_dict() for r in rollouts]
        self.fs.write_rollouts(iteration, self.worker_id, rollout_dicts)

        # Update status
        self._update_status("done", iteration=iteration, rollouts=len(rollouts))
        print(f"  Done! Collected {len(rollouts)} rollouts.")

    def _update_status(
        self,
        status: str,
        iteration: int = -1,
        rollouts: int = 0,
        error: str = None,
    ):
        """Update worker status."""
        current = self.fs.read_worker_status(self.worker_id)

        new_status = WorkerStatus(
            worker_id=self.worker_id,
            status=status,
            rollouts_collected=rollouts if rollouts > 0 else (current.rollouts_collected if current else 0),
            last_heartbeat=time.time(),
            current_iteration=iteration if iteration >= 0 else (current.current_iteration if current else -1),
            error_message=error,
        )

        self.fs.write_worker_status(new_status)


def run_coordinator(args):
    """Run coordinator mode."""
    fs = SharedFileSystem(args.shared_dir)

    config = {
        "planner_model": args.planner_model,
        "worker_model": args.worker_model,
        "num_agents": args.num_agents,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "num_iterations": args.num_iterations,
        "use_group_baseline": True,
    }

    coordinator = DistributedCoordinator(
        fs,
        num_workers=args.num_workers,
        output_dir=args.output,
        config=config,
    )

    coordinator.initialize()
    coordinator.wait_for_workers()
    coordinator.run_training_loop(args.num_iterations)


def run_worker(args):
    """Run worker mode."""
    fs = SharedFileSystem(args.shared_dir)

    config = {
        "planner_model": args.planner_model,
        "worker_model": args.worker_model,
        "num_agents": args.num_agents,
    }

    worker = DistributedWorker(
        worker_id=args.worker_id,
        shared_fs=fs,
        planner_model=args.planner_model,
        worker_model=args.worker_model,
        config=config,
    )

    worker.initialize()
    worker.run()


def main():
    parser = argparse.ArgumentParser(description="Distributed RL Training")

    # Mode
    parser.add_argument("--mode", "-m", choices=["coordinator", "worker"], required=True,
                       help="Run as coordinator or worker")

    # Shared args
    parser.add_argument("--shared-dir", "-s", type=str, required=True,
                       help="Shared directory for communication (NFS)")
    parser.add_argument("--planner-model", type=str, default="Qwen/Qwen3-4B-Instruct",
                       help="Planner model name")
    parser.add_argument("--worker-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct",
                       help="Worker model name")
    parser.add_argument("--num-agents", type=int, default=100,
                       help="Number of agents in simulation")

    # Coordinator args
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of expected workers (coordinator only)")
    parser.add_argument("--output", "-o", type=str, default="models/planner-rl",
                       help="Output directory (coordinator only)")
    parser.add_argument("--num-iterations", type=int, default=100,
                       help="Training iterations (coordinator only)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Rollouts per iteration (coordinator only)")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate (coordinator only)")

    # Worker args
    parser.add_argument("--worker-id", type=str, default=None,
                       help="Unique worker ID (worker only)")

    args = parser.parse_args()

    # Validate args
    if args.mode == "worker" and args.worker_id is None:
        args.worker_id = f"worker_{uuid.uuid4().hex[:8]}"
        print(f"Auto-generated worker ID: {args.worker_id}")

    # Run
    if args.mode == "coordinator":
        run_coordinator(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
