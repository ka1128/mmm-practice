# Marketing Mix Modeling & ROI Optimization Pipeline

## Overview

This project builds a reproducible Marketing Mix Modeling (MMM) pipeline
to estimate media impact and optimize budget allocation based on ROI.

The repository is structured to separate:
- Core logic (`src/`)
- Execution notebooks (`notebooks/pipeline`)
- Exploratory analysis (`notebooks/exploratory`)
- Reproducible artifacts (`outputs/`)

The goal is to provide a practical, reusable structure suitable for real-world analytics workflows.

---

## Pipeline Flow (00 → 08)

Execute notebooks in the following order:

00_runlog  
01_data_preparation  
02_feature_engineering  
03_lag_decay_exploration  
04_decay_optimization  
05_model_fit  
06_roi_timing_optimization  
07_roi_allocation_optimization  
08_roi_alwayson_scenario_comparison  

Exploratory notebook:
- 01_5_exploratory_analysis.ipynb (not part of the main pipeline)

---

## Outputs (Contract Artifacts)

The pipeline generates the following reproducible artifacts:

- df_w.pkl
- df_w_feat.pkl
- best_decay.pkl
- final_model.pkl
- weekly_score.pkl
- df_alloc_curve.csv
- scenario_table.csv

Intermediate artifacts are intentionally not persisted.

---

## Data

The dataset is based on publicly available weekly sales and media spend data from Kaggle.
Raw data is not included in this repository.
The focus is on pipeline design and reproducibility.

---

## Key Design Principles

- Single Source of Truth: logic centralized in `src/`
- Notebook minimalism: notebooks only orchestrate execution
- Reproducibility: artifacts defined and controlled
- Scenario-driven optimization for practical decision support

---

## How to Run

1. Clone the repository
2. Install dependencies (pandas, numpy, statsmodels, matplotlib, seaborn)
3. Execute notebooks in `notebooks/pipeline` from 00 to 08

---

## What This Project Demonstrates

- Adstock modeling and decay optimization
- Model selection via AIC
- ROI-based timing optimization
- Budget allocation simulation
- Always-on vs Pulse scenario comparison
- Clean analytical architecture for real-world use
