# Research experiment matrix: M0–M4 × (FP32, AMP) = 10 cells

## M4 definition (All applied)

**M4 All applied** means NHWC layout where applicable, GPU preprocessing offload (`preprocess_gpu`, `compile_voxelizer`), memory layout knobs, and `torch.compile` on the model for **both** M4_FP32 and M4_AMP (compile + `autocast` FP16 for AMP); use a large `--warmup` (dynamo + compiled voxelizer).

Source: `research_experiment_matrix.M4_ALL_APPLIED_DEFINITION` (alias: `M5_ALL_APPLIED_DEFINITION`).

---

## Master table (status per cell)

| Cell    | Model                  | Precision | Status   | Notes |
|---------|------------------------|-----------|----------|-------|
| M0_FP32 | OpenPCDet Baseline     | FP32      | runnable | |
| M0_AMP  | OpenPCDet Baseline     | AMP       | runnable | |
| M1_FP32 | Compiled               | FP32      | runnable | `--compile` |
| M1_AMP  | Compiled               | AMP       | runnable | `--compile` + `--amp`; use large `--warmup` |
| M2_*    | NHWC                   | *         | future   | Not in harness |
| M3_*    | Preprocessing offload  | *         | future   | Unify with `inference.py` |
| M4_*    | All applied            | *         | future   | Full stack wiring |

Machine-readable definitions: `research_experiment_matrix.EXPERIMENT_MATRIX_FP32_AMP`.

Export CSV: `python collect_research_metrics.py matrix --output_csv path/to/experiment_matrix_fp32_amp.csv`.

---

## M1_AMP (compile + AMP)

Enabled in the harness: `profile_suite.py` / `energy_monitor.py` wrap forward with `torch.autocast` and use the same `batch_dict` tensor-only path as compile-only runs. For fair latency and energy, use **large `--warmup`** (e.g. 100–200+); dynamo may still recompile on some batches, which can inflate **p99** and energy totals on short runs.

---

## collect_research_metrics integration

- `--matrix legacy` (default): three runs (`baseline_fp32`, `torch_compile_fp32`, `fp16_amp`).
- `--matrix fp32_amp` (alias: `--matrix 15`): runnable cells from `EXPERIMENT_MATRIX_FP32_AMP` by default (currently M0_FP32, M0_AMP, M1_FP32, M1_AMP).
- `matrix` subcommand: writes the full 10-row status table to CSV.

---

## Harness notes (NHWC / preprocess)

- **Today:** `profile_suite.py` uses `build_dataloader` + `load_data_to_gpu` + model forward + `generate_prediction_dicts`.
- **Target:** Match `inference.py` flags `--preprocess_gpu` / `--compile_voxelizer` with identical `warmup`/`steps` and the same CSV outputs.
- **NHWC:** Apply channels-last where applicable; extend `profile_suite` when integrated.

INT8 is not part of this matrix; any prior INT8 column design is out of scope for the fp32_amp table.
