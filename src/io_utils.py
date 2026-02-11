# src/io_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Union
import pickle

PathLike = Union[str, Path]

from src.artifacts import ArtifactPaths, build_artifact_paths

def project_root(marker: str = ".git") -> Path:
    """
    Return repository root by walking up from this file until `marker` is found.
    Typical marker: ".git" (recommended) or "pyproject.toml".
    """
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / marker).exists():
            return p
    # Fallback: src/ の1階層上をルート扱い
    return here.parent.parent


def outputs_dir(subdir: str = "outputs", create: bool = True, marker: str = ".git") -> Path:
    """
    Return outputs directory path (repo_root / subdir).
    """
    out = project_root(marker=marker) / subdir
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def load_pickle(path: PathLike) -> Any:
    """
    Load python object from pickle.
    Accepts absolute path or path relative to project root.
    """
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with p.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: PathLike, *, protocol: int = pickle.HIGHEST_PROTOCOL) -> Path:
    """
    Save python object to pickle.
    Accepts absolute path or path relative to project root.
    Creates parent dirs automatically.
    Returns the resolved path.
    """
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    return p

def get_artifact_paths(
    *,
    subdir: str = "outputs",
    create: bool = True,
    marker: str = ".git",
) -> ArtifactPaths:
    """
    Resolve all contracted artifact paths under outputs/ and return them as a bundle.

    Notebook should NOT hardcode file names. Use:
        paths = get_artifact_paths()
        df = load_pickle(paths.df_w)
    """
    out_dir = outputs_dir(subdir=subdir, create=create, marker=marker)
    return build_artifact_paths(out_dir)
