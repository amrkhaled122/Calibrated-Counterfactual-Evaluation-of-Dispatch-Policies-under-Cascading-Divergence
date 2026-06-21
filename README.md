# Calibrated Counterfactual Evaluation of Dispatch Policies under Cascading Divergence

This repository contains the data-processing code, agents, simulator, and result-analysis utilities needed to reproduce the ECML PKDD 2026 paper results for counterfactual dispatch-policy evaluation.

## Citation

If you use this repository, source code, experimental setup, or derived artifacts, please cite the final Springer proceedings version of the paper when it becomes available. The official BibTeX entry will be added here after Springer publishes it.

This work uses the public Meituan INFORMS TSL Research Challenge dataset. Please also follow the dataset license and acknowledgment requirements:

```bibtex
@misc{meituan2024informs_tsl_challenge,
  title        = {Meituan INFORMS TSL Research Challenge Dataset},
  author       = {{Meituan}},
  year         = {2024},
  url          = {https://github.com/meituan/Meituan-INFORMS-TSL-Research-Challenge},
  note         = {Available under the Creative Commons Attribution-NonCommercial 4.0 International license}
}
```

Dataset acknowledgment:

> This research was supported by data provided by Meituan.

## Layout

- `data/`: expected raw/processed data layout plus preprocessing code.
- `agents/`: BC and DDQN agent code plus local output directories for generated model artifacts.
- `simulation/`: simulator, integrated DDQN training, and result-analysis utilities.
- `results/`: generated simulation outputs for baseline, BC, and DDQN.

## Reproduce Results

Run commands from the repository root.

## Data And Model Artifacts

Licensed data files and trained model weights are not committed to git. The repository keeps placeholder directories so regenerated or locally provided files land at the expected paths.

Place licensed data under `data/` as described in `data/README.md`. Run the training commands below to regenerate model artifacts under `agents/outputs/`, or place locally trained/downloaded artifacts at the same paths.

## Runtime Notes

Full reproduction is computationally expensive. DDQN training for 20 epochs took up to 3 days on the RTX 4080 machine with 6 GB VRAM available at the time. Full-window simulations can also take a long time to finish, especially when running baseline, BC, and DDQN sequentially. Use shorter `--hours` windows first to validate the setup before launching full experiments.

### 1. Build Processed Features

```bash
python -m data.preprocessing_code.features.cli \
  --main data/all_waybill_info_meituan_0322_edited.csv \
  --cycles data/dispatch_cycles_with_scaled_recipient_coords.cleaned.csv \
  --out_dir data/features
```

The simulator also consumes:

- `data/actual_eta_by_order.csv`
- `data/courier_working_blocks.csv`
- `data/travel_time_model.json`

### 2. Train BC

```bash
python -m agents.training.bc.train_bc \
  --parquet data/features/offers_observations.parquet \
  --manifest data/features/manifest.json \
  --out_dir agents/outputs/bc_model
```

### 3. Train DDQN

```bash
python -m simulation.train_ddqn_integrated \
  --parquet data/features/offers_observations.parquet \
  --manifest data/features/manifest.json \
  --out_dir agents/outputs/ddqn_model_integrated \
  --travel_time_model data/travel_time_model.json \
  --episodes 10 \
  --train_hours 48 \
  --eval_hours 48 \
  --reward_type paper
```

### 4. Run Evaluation

```bash
python -m simulation.run_simulation --baseline --start_hour 48 --hours 48
python -m simulation.run_simulation --agent_type bc --start_hour 48 --hours 48
python -m simulation.run_simulation --agent_type ddqn --start_hour 48 --hours 48
```

Outputs are written to `results/`:

- `system_simulation_metrics_baseline.json`
- `system_simulation_metrics_bc_agent.json`
- `system_simulation_metrics_ddqn_agent.json`
- `log_*_underutilization.txt`
- `log_*_idle_blocks.txt`
- `profiling_*.json`

## Useful Checks

```bash
python -m simulation.validate
python -m simulation.run_all_simulations --hours 48
python -m simulation.run_ddqn_action_analysis
```
