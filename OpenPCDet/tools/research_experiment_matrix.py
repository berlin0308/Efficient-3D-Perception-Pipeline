"""
Research design: 5 pipeline variants (M0–M4) × 2 precisions (FP32, AMP).

- M0: OpenPCDet baseline (no compile)
- M1: torch.compile
- M2: memory layout knobs (scatter HWC / conv2d channels_last); three FP32 and three AMP variants
- M3: GPU preprocessing offload (--preprocess_gpu, optional --compile_voxelizer; profile_suite / inference.py path)
- M4: All applied (harness TBD)

Use experiment_matrix_fp32_amp() with collect_research_metrics --matrix fp32_amp.
"""

from __future__ import annotations

from typing import Any, Literal

# One-sentence definition for report text (M4 "All applied")
M4_ALL_APPLIED_DEFINITION = (
    'M4 All applied means NHWC layout where applicable, GPU preprocessing offload '
    '(preprocess_gpu, compile_voxelizer), memory_opt_scatter/conv2d, and torch.compile on the model '
    'for both M4_FP32 and M4_AMP (compile + autocast(fp16)); use large warmup (dynamo + compiled voxelizer).'
)

# Backward compatibility for docs / old imports
M5_ALL_APPLIED_DEFINITION = M4_ALL_APPLIED_DEFINITION

CellStatus = Literal['runnable', 'blocked', 'future']


def _cell(
    cell_id: str,
    model_id: str,
    precision: str,
    variant_name: str,
    status: CellStatus,
    *,
    compile_: bool = False,
    amp: bool = False,
    int8: bool = False,
    nhwc: bool = False,
    preprocess_gpu: bool = False,
    compile_voxelizer: bool = False,
    skip_reason: str = '',
    memory_opt_scatter: bool = False,
    memory_opt_conv2d: bool = False,
) -> dict[str, Any]:
    return {
        'cell_id': cell_id,
        'model_variant': model_id,
        'precision_mode': precision,
        'variant_name': variant_name,
        'status': status,
        'compile': compile_,
        'amp': amp,
        'int8': int8,
        'nhwc': nhwc,
        'preprocess_gpu': preprocess_gpu,
        'compile_voxelizer': compile_voxelizer,
        'skip_reason': skip_reason,
        'memory_opt_scatter': memory_opt_scatter,
        'memory_opt_conv2d': memory_opt_conv2d,
    }


# Design matrix: M0–M4 × (FP32, AMP), with M2 split into six memory-layout cells (FP32×3 + AMP×3) plus other futures.
EXPERIMENT_MATRIX_FP32_AMP: list[dict[str, Any]] = [
    # M0 Baseline
    _cell('M0_FP32', 'M0', 'FP32', 'M0_FP32', 'runnable'),
    _cell('M0_AMP', 'M0', 'AMP', 'M0_AMP', 'runnable', amp=True),
    # M1 Compiled
    _cell('M1_FP32', 'M1', 'FP32', 'M1_FP32', 'runnable', compile_=True),
    # torch.compile + autocast(fp16); use generous warmup — dynamo may still recompile on some batches.
    _cell('M1_AMP', 'M1', 'AMP', 'M1_AMP', 'runnable', compile_=True, amp=True),
    # M2 memory layout (FP32): three runnable harness variants (see --memory_opt_* in profile_suite)
    _cell(
        'M2_FP32_mem_scatter', 'M2', 'FP32', 'M2_FP32_mem_scatter', 'runnable', nhwc=True,
        memory_opt_scatter=True, memory_opt_conv2d=False,
    ),
    _cell(
        'M2_FP32_mem_conv2d', 'M2', 'FP32', 'M2_FP32_mem_conv2d', 'runnable', nhwc=True,
        memory_opt_scatter=False, memory_opt_conv2d=True,
    ),
    _cell(
        'M2_FP32_mem_both', 'M2', 'FP32', 'M2_FP32_mem_both', 'runnable', nhwc=True,
        memory_opt_scatter=True, memory_opt_conv2d=True,
    ),
    # M2 memory layout (AMP / autocast fp16): same three harness variants as M2_FP32
    _cell(
        'M2_AMP_mem_scatter', 'M2', 'AMP', 'M2_AMP_mem_scatter', 'runnable', nhwc=True, amp=True,
        memory_opt_scatter=True, memory_opt_conv2d=False,
    ),
    _cell(
        'M2_AMP_mem_conv2d', 'M2', 'AMP', 'M2_AMP_mem_conv2d', 'runnable', nhwc=True, amp=True,
        memory_opt_scatter=False, memory_opt_conv2d=True,
    ),
    _cell(
        'M2_AMP_mem_both', 'M2', 'AMP', 'M2_AMP_mem_both', 'runnable', nhwc=True, amp=True,
        memory_opt_scatter=True, memory_opt_conv2d=True,
    ),
    # M3 Preprocessing offload (GPU voxelization; no torch.compile on model; optional voxelizer compile off here)
    _cell(
        'M3_FP32', 'M3', 'FP32', 'M3_FP32', 'runnable',
        preprocess_gpu=True, compile_voxelizer=False,
    ),
    _cell(
        'M3_AMP', 'M3', 'AMP', 'M3_AMP', 'runnable',
        preprocess_gpu=True, compile_voxelizer=False, amp=True,
    ),
    # M4 All applied: M1 torch.compile(model) + M2 mem_both + M3 preprocess_gpu + compile_voxelizer (see M4_ALL_APPLIED_DEFINITION)
    _cell(
        'M4_FP32', 'M4', 'FP32', 'M4_FP32', 'runnable',
        compile_=True, nhwc=True, preprocess_gpu=True, compile_voxelizer=True,
        memory_opt_scatter=True, memory_opt_conv2d=True,
    ),
    _cell(
        'M4_AMP', 'M4', 'AMP', 'M4_AMP', 'runnable',
        compile_=True, amp=True, nhwc=True, preprocess_gpu=True, compile_voxelizer=True,
        memory_opt_scatter=True, memory_opt_conv2d=True,
    ),
]


def experiment_matrix_fp32_amp() -> list[dict[str, Any]]:
    """Return a copy of the M0–M4 × (FP32, AMP) design."""
    return [dict(c) for c in EXPERIMENT_MATRIX_FP32_AMP]


def experiment_matrix_15() -> list[dict[str, Any]]:
    """Deprecated alias; use experiment_matrix_fp32_amp()."""
    return experiment_matrix_fp32_amp()


def runnable_cells() -> list[dict[str, Any]]:
    return [c for c in EXPERIMENT_MATRIX_FP32_AMP if c['status'] == 'runnable']


def cell_by_variant_name(name: str) -> dict[str, Any] | None:
    for c in EXPERIMENT_MATRIX_FP32_AMP:
        if c['variant_name'] == name:
            return dict(c)
    return None


def default_variant_matrix_legacy():
    """Backward-compatible 3-run matrix (same as research_metrics_schema.default_variant_matrix)."""
    return [
        {'variant_name': 'baseline_fp32', 'compile': False, 'amp': False},
        {'variant_name': 'torch_compile_fp32', 'compile': True, 'amp': False},
        {'variant_name': 'fp16_amp', 'compile': False, 'amp': True},
    ]
