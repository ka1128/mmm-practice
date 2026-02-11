# src/artifacts.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


# =========================================================
# Outputs contract (single source of truth)
# =========================================================

DF_W = "df_w.pkl"
DF_W_FEAT = "df_w_feat.pkl"
BEST_DECAY = "best_decay.pkl"
FINAL_MODEL = "final_model.pkl"
WEEKLY_SCORE = "weekly_score.pkl"
DF_ALLOC_CURVE = "df_alloc_curve.csv"
SCENARIO_TABLE = "scenario_table.csv"


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolved artifact paths under outputs/ directory."""
    df_w: Path
    df_w_feat: Path
    best_decay: Path
    final_model: Path
    weekly_score: Path
    df_alloc_curve: Path
    scenario_table: Path

    def as_dict(self) -> Dict[str, Path]:
        return {
            "df_w": self.df_w,
            "df_w_feat": self.df_w_feat,
            "best_decay": self.best_decay,
            "final_model": self.final_model,
            "weekly_score": self.weekly_score,
            "df_alloc_curve": self.df_alloc_curve,
            "scenario_table": self.scenario_table,
        }


def build_artifact_paths(out_dir: Path) -> ArtifactPaths:
    """Create absolute paths for all artifacts (contracted names)."""
    out_dir = Path(out_dir)
    return ArtifactPaths(
        df_w=out_dir / DF_W,
        df_w_feat=out_dir / DF_W_FEAT,
        best_decay=out_dir / BEST_DECAY,
        final_model=out_dir / FINAL_MODEL,
        weekly_score=out_dir / WEEKLY_SCORE,
        df_alloc_curve=out_dir / DF_ALLOC_CURVE,
        scenario_table=out_dir / SCENARIO_TABLE,
    )
