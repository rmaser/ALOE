import os
import time
import logging
from pathlib import Path
from omegaconf import DictConfig
from src.utils.env import get_cache_path

log = logging.getLogger(__name__)

def get_rank():
    """Get the rank of the current process."""
    # Try to get rank from environment variables
    rank = os.getenv("RANK")
    if rank is not None:
        return int(rank)
    
    # Try SLURM_PROCID
    rank = os.getenv("SLURM_PROCID")
    if rank is not None:
        return int(rank)
        
    # Default to 0 if not found (e.g. local run)
    return 0

def get_slurm_job_identifier():
    """Get the SLURM job identifier (handling array jobs)."""
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    if not slurm_job_id:
        return None
        
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    
    if array_job_id and array_task_id:
        return f"{array_job_id}_{array_task_id}"
    return slurm_job_id

def get_nugget_path(job_identifier):
    """Get the path to the nugget file for a given job identifier."""
    cache_path = get_cache_path()
    nugget_dir = cache_path / "slurm_nuggets"
    nugget_dir.mkdir(exist_ok=True)
    return nugget_dir / f"{job_identifier}.txt"

def check_slurm_resume(cfg: DictConfig):
    """
    Check if we should resume a SLURM job.
    If a nugget exists, update cfg.logger.id with the stored run ID.
    All ranks check for the nugget. If it exists, they load it.
    If it doesn't exist, they proceed (assuming a new run).
    """
    resume_slurm = cfg.get("resume_slurm", True)
    if not resume_slurm:
        return

    job_identifier = get_slurm_job_identifier()
    if not job_identifier:
        return

    nugget_path = get_nugget_path(job_identifier)
    
    # Check if nugget exists (safe for all ranks to read)
    if nugget_path.exists():
        # Check if the nugget is old enough (e.g. > 2 minutes)
        # This prevents reading a nugget that was just created by Rank 0 in the current run
        try:
            mtime = nugget_path.stat().st_mtime
            age = time.time() - mtime
            if age < 120:
                log.info(f"Found SLURM nugget for job {job_identifier}, but it is too new ({age:.1f}s). Ignoring.")
                return

            with open(nugget_path, "r") as f:
                resumed_run_id = f.read().strip()
            
            log.info(f"Found SLURM nugget for job {job_identifier}. Resuming WandB run: {resumed_run_id}")
            print(f"Found SLURM nugget for job {job_identifier}. Resuming WandB run: {resumed_run_id}")
            
            if "logger" in cfg and cfg.logger is not None:
                cfg.logger.id = resumed_run_id
        except Exception as e:
            log.warning(f"Failed to read nugget file: {e}")

def create_slurm_nugget(cfg: DictConfig, logger):
    """
    Create a nugget file with the current run ID if it doesn't exist.
    Only Rank 0 should do this.
    """
    resume_slurm = cfg.get("resume_slurm", True)
    if not resume_slurm:
        return

    job_identifier = get_slurm_job_identifier()
    if not job_identifier:
        return

    # Only Rank 0 creates the nugget
    if get_rank() != 0:
        return

    nugget_path = get_nugget_path(job_identifier)
    if nugget_path.exists():
        return

    if logger and hasattr(logger, "experiment") and hasattr(logger.experiment, "id"):
        try:
            run_id = logger.experiment.id
            with open(nugget_path, "w") as f:
                f.write(run_id)
            log.info(f"[Rank 0] Created SLURM nugget for job {job_identifier} with run ID: {run_id}")
            print(f"[Rank 0] Created SLURM nugget for job {job_identifier} with run ID: {run_id}")
        except Exception as e:
            log.error(f"Could not create SLURM nugget: {e}")
            print(f"Could not create SLURM nugget: {e}")


def find_latest_checkpoint_for_run_id(run_id: str, cache_path: Path | None = None) -> Path | None:
    """Find the newest ``last*.ckpt`` for a WandB/Lightning run ID.

    We support both layouts used in this repo:
    - ``<CACHE_PATH>/wandb/run-*-<run_id>/.../last*.ckpt``
    - ``<CACHE_PATH>/**/<run_id>/checkpoints/last*.ckpt``
    """
    if cache_path is None:
        cache_path = get_cache_path()

    candidate_checkpoints: list[Path] = []
    seen: set[Path] = set()

    wandb_root = cache_path / "wandb"
    if wandb_root.exists():
        for run_dir in wandb_root.glob(f"*run-*-{run_id}"):
            for ckpt in run_dir.rglob("last*.ckpt"):
                resolved = ckpt.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidate_checkpoints.append(ckpt)

    for ckpt in cache_path.rglob(f"{run_id}/checkpoints/last*.ckpt"):
        resolved = ckpt.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidate_checkpoints.append(ckpt)

    if not candidate_checkpoints:
        return None

    candidate_checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidate_checkpoints[0]
