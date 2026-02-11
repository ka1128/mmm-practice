# src/runlog.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
import uuid
from typing import Any, Dict, Optional


def new_run_id(prefix: str = "run") -> str:
    """
    Example: run_20260208_235959_ab12cd
    UTC timestamp + short random suffix
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suf = uuid.uuid4().hex[:6]
    return f"{prefix}_{ts}_{suf}"


def _safe_import_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def collect_env() -> Dict[str, Any]:
    """Collect minimal environment metadata for reproducibility."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "libs": {
            "pandas": _safe_import_version("pandas"),
            "numpy": _safe_import_version("numpy"),
            "statsmodels": _safe_import_version("statsmodels"),
            "matplotlib": _safe_import_version("matplotlib"),
        },
    }


def collect_git(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Returns commit hash / dirty flag if git is available and repo exists.
    If unavailable, commit/is_dirty will be None.
    """
    def _run(cmd: list[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(
                cmd,
                cwd=str(repo_root) if repo_root else None,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    is_dirty = None if status is None else (len(status) > 0)

    return {"commit": commit, "is_dirty": is_dirty}


def make_runlog_payload(
    run_id: str,
    seed: Optional[int] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config_snapshot": config_snapshot,
        "git": collect_git(repo_root=repo_root),
        "env": collect_env(),
    }
    if extra:
        payload["extra"] = extra
    return payload


def write_runlog(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path
