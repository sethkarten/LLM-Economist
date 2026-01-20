# AI Agent Instructions

> **📋 Complete Guide Index:** See [FILE_MAP.md](FILE_MAP.md) for all available documentation
> - GPU & Training: `training/` directory
> - Writing & Research: `writing/` directory

---

## GPU Job Submission

This project uses a GPU manager system at `http://localhost:8080` to run experiments across multiple GPU resources.

### To submit a training job:

```bash
curl -X POST http://localhost:8080/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "REPO_URL",
    "branch": "BRANCH_NAME",
    "script": "SCRIPT.py",
    "args": ["--arg1", "value1"],
    "gpu_count": 1,
    "gpu_memory_min_gb": 40,
    "time_limit_hours": 24
  }'
```

### Key parameters:
- `script`: Python file (`.py`) or bash script (`.sh`) - auto-detected by extension
- `gpu_count`: Number of GPUs (1, 2, 4, or 8)
- `gpu_memory_min_gb`: Minimum GPU memory (32=5090, 48=A6000, 80=H100/A100, 141=B200)
- `time_limit_hours`: SLURM QoS priority (shorter = higher priority):
  - **≤1h**: `gpu-test` QoS (highest priority) - debugging/interactive
  - **≤24h**: `gpu-short` QoS (medium priority) - short training runs
  - **≤72h**: `gpu-medium` QoS (lower priority) - long training runs

### Resource allocation (SLURM clusters):
- **12 CPUs per GPU** (automatically allocated)
- **100GB RAM per GPU** (automatically allocated)
- Example: 4 GPUs = 48 CPUs + 400GB RAM

### Check job status:
```bash
curl http://localhost:8080/api/jobs/{job_id}
```

### Available resources (auto-selected by scheduler):
- **local-5090**: 2x 5090 (32GB), CUDA 12.8 - instant start, no VPN required
- **Cynthia**: 8x A6000 Ada (48GB), CUDA 12.8 - instant start, via Mac-VPN
- **Pikachu**: 8x A6000 (48GB), CUDA 12.8 - instant start, via Mac-VPN
- **della-ailab**: 8x B200 (141GB), CUDA 12.8 - SLURM queue, via Mac-VPN
- **della-pli**: 15x H100 (80GB), CUDA 12.8 - SLURM queue, via Mac-VPN
- **della-a100**: 4x A100 (80GB), CUDA 12.8 - SLURM queue, via Mac-VPN

**Network Requirements**: All remote resources (Cynthia, Pikachu, della-*) require SSH access through Mac-VPN proxy (`milkkartenVPN.local`). If Mac is offline, only local-5090 will work.

### Before submitting:
1. Push code changes to GitHub (use `git@github.com:user/repo.git` format)
2. Ensure `requirements.txt` or `pyproject.toml` exists (for Python scripts)
3. Script should accept CLI arguments for hyperparameters
4. For remote servers:
   - SSH keys must be set up on target servers for GitHub access
   - Mac-VPN proxy (`milkkartenVPN.local`) must be online and reachable
   - Test connectivity: `ssh Cynthia hostname` or `ssh della-gpu hostname`
5. **For SLURM (della-*) clusters**: Compute nodes lack internet - pre-install dependencies on login node:
   ```bash
   ssh della-gpu "cd /scratch/gpfs/CHIJ/milkkarten/your-project && uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt"
   ```
   Or use a bash script (`.sh`) instead of Python

### Full documentation:

**GPU & Training:**
- Training Best Practices: `training/TRAINING_AGENT_GUIDE.md`
- LLM Optimization: `training/LLM_OPTIMIZATION_GUIDE.md`
- Quick Reference: `training/GPU_MANAGER_REFERENCE.md`

**Writing & Research:**
- Style Guide: `writing/STYLE_GUIDE.md` - NeurIPS academic writing
- Writing Assistant: `writing/WRITING_ASSISTANT.md` - Drafting papers
- Review Loop: `writing/WRITING_LOOP.md` - Polishing drafts
- See [FILE_MAP.md](FILE_MAP.md) for complete index (19 writing guides)

**Training Workflow:**
- Training Loop: `training/TRAINING_LOOP.md` - Systematic experimentation workflow
