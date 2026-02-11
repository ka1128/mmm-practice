# src/bootstrap.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.runlog import new_run_id, make_runlog_payload, write_runlog


@dataclass(frozen=True)
class Ctx:
    run_id: str
    seed: int
    root_dir: Path
    out_dir: Path


def bootstrap(
    *,
    seed: int,
    root_dir: Path,
    out_dir: Path,
    make_runlog: bool = False,
    config_snapshot: Optional[Dict[str, Any]] = None,
    extra_runlog: Optional[Dict[str, Any]] = None,
) -> Ctx:
    """
    Notebook bootstrap:
    - create run_id
    - set random seed
    - optionally write runlog
    - return context
    """
    run_id = new_run_id()

    # seed (numpy only; add others only if you actually use them)
    np.random.seed(seed)

    if make_runlog:
        payload = make_runlog_payload(
            run_id=run_id,
            seed=seed,
            config_snapshot=config_snapshot,
            repo_root=root_dir,
            extra=extra_runlog,
        )
        runlog_path = out_dir / "runlogs" / f"{run_id}.json"
        write_runlog(runlog_path, payload)

        latest_path = out_dir / "runlogs" / "latest_run_id.txt"
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(run_id, encoding="utf-8")

    return Ctx(run_id=run_id, seed=seed, root_dir=root_dir, out_dir=out_dir)
