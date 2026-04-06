"""
Frozen CSV schema for PointPillars research metrics (runs.csv + related files).

Use RUNS_CSV_COLUMNS as the single source of truth for column order.
ENERGY_BREAKDOWN_METHODOLOGY documents how to interpret DRAM vs on-chip memory energy claims.
"""

# -----------------------------------------------------------------------------
# Energy / memory hierarchy (for report text — NVML does not split DRAM vs SRAM)
# -----------------------------------------------------------------------------

ENERGY_BREAKDOWN_METHODOLOGY = """
Energy numbers from energy_monitor.py use NVML GPU board power sampled over time (integrated
to Joules). This is a single rail for the whole GPU package, not a breakdown into DRAM vs
on-chip SRAM energy.

Warmup vs measurement (parity with profile_suite.py):
  Warmup runs before the NVML integration window starts, so warmup energy is never included.
  Optional --measurement_burnin_steps runs additional forwards with the sampler on, but the
  reported Joules integrate only the window after burn-in (same idea as profile_suite's
  measurement burn-in before the timed profiler loop).
  With --compile / --compile_voxelizer / --energy_exclude_spikes_always, forward steps slower
  than --energy_exclude_compile_over_ms can be excluded from energy and samples/J; pick the
  threshold from k× steady forward p99 on a non-compile baseline (typical k=2–4).

To argue that data movement dominates (e.g., DRAM energy > SRAM), use a complementary method:
  (1) NVIDIA Nsight Compute (NCU): DRAM/L2 traffic, hit rates, memory throughput vs compute.
  (2) An analytical or published energy-per-byte model applied to measured bytes moved.
  (3) Platform-specific power counters if available (uncommon on consumer laptop GPUs).

Always set column energy_method to 'nvml_integrated' for monitor runs, or describe the proxy
(e.g. 'ncu_bytes_times_energy_model') when combining NCU with a model.
""".strip()


# -----------------------------------------------------------------------------
# runs.csv — one row per (variant, run_id); empty string for unknown / not measured
# -----------------------------------------------------------------------------

RUNS_CSV_COLUMNS = (
    # --- experiment metadata ---
    'run_id',
    'timestamp_iso',
    'variant_name',
    'experiment_cell_id',
    'model_variant',
    'precision_mode',
    'experiment_status',
    'model_name',
    'config_path',
    'checkpoint_path',
    'gpu_name',
    'driver_version',
    'cuda_version',
    'pytorch_version',
    'commit_hash',
    'batch_size',
    'warmup_steps',
    'measured_steps',
    'num_workers',
    'cuda_id',
    'dataset_note',
    'split_note',
    'frame_source_note',
    'flag_compile',
    'flag_amp',
    'flag_preprocess_gpu',
    'flag_compile_voxelizer',
    'flag_nhwc',
    'flag_memory_opt_scatter',
    'flag_int8',
    'flag_fp16_full',
    'notes_amp_compile_mutually_exclusive',
    'energy_method',
    # --- profile_suite (end-to-end stages) ---
    'prof_dataloader_mean_ms',
    'prof_dataloader_p50_ms',
    'prof_dataloader_p95_ms',
    'prof_dataloader_p99_ms',
    'prof_dataloader_std_ms',
    'prof_h2d_mean_ms',
    'prof_h2d_p50_ms',
    'prof_h2d_p95_ms',
    'prof_h2d_p99_ms',
    'prof_h2d_std_ms',
    # --- profile_suite (KITTI / inference.py NVTX stages; absent in pure DataLoader RT path) ---
    'prof_read_points_mean_ms',
    'prof_read_points_p50_ms',
    'prof_read_points_p95_ms',
    'prof_read_points_p99_ms',
    'prof_read_points_std_ms',
    'prof_cpu_prepare_mean_ms',
    'prof_cpu_prepare_p50_ms',
    'prof_cpu_prepare_p95_ms',
    'prof_cpu_prepare_p99_ms',
    'prof_cpu_prepare_std_ms',
    'prof_data_to_gpu_pts_mean_ms',
    'prof_data_to_gpu_pts_p50_ms',
    'prof_data_to_gpu_pts_p95_ms',
    'prof_data_to_gpu_pts_p99_ms',
    'prof_data_to_gpu_pts_std_ms',
    'prof_pre_processing_mean_ms',
    'prof_pre_processing_p50_ms',
    'prof_pre_processing_p95_ms',
    'prof_pre_processing_p99_ms',
    'prof_pre_processing_std_ms',
    'prof_h2d_voxel_tail_mean_ms',
    'prof_h2d_voxel_tail_p50_ms',
    'prof_h2d_voxel_tail_p95_ms',
    'prof_h2d_voxel_tail_p99_ms',
    'prof_h2d_voxel_tail_std_ms',
    'prof_forward_mean_ms',
    'prof_forward_p50_ms',
    'prof_forward_p95_ms',
    'prof_forward_p99_ms',
    'prof_forward_std_ms',
    'prof_postprocess_mean_ms',
    'prof_postprocess_p50_ms',
    'prof_postprocess_p95_ms',
    'prof_postprocess_p99_ms',
    'prof_postprocess_std_ms',
    'prof_full_frame_mean_ms',
    'prof_full_frame_p50_ms',
    'prof_full_frame_p95_ms',
    'prof_full_frame_p99_ms',
    'prof_full_frame_std_ms',
    'prof_throughput_sps',
    'prof_peak_gpu_memory_mb',
    'prof_peak_gpu_memory_steady_mb',
    'prof_t_rt_mean_ms',
    'prof_mean_peak_gpu_memory_mb',
    'prof_cuda_kernel_events',
    'profile_output_dir',
    'profile_latency_per_step_csv',
    # --- energy_monitor (forward-only wall time + NVML) ---
    'energy_forward_mean_ms',
    'energy_forward_p50_ms',
    'energy_forward_p95_ms',
    'energy_forward_p99_ms',
    'energy_throughput_sps',
    'energy_wall_time_s',
    'energy_mean_power_W',
    'energy_peak_power_W',
    'energy_total_J',
    'energy_samples_per_J',
    'energy_samples_per_s_per_W',
    'energy_output_dir',
    'energy_samples_csv',
    'energy_latency_per_step_csv',
    # --- accuracy (KITTI official Python eval R40; map_car_r11 = Car 3D moderate, legacy name) ---
    'map_car_r11',
    'kitti_car_3d_easy_r40',
    'kitti_car_3d_moderate_r40',
    'kitti_car_3d_hard_r40',
    'eval_protocol_notes',
    # --- NCU summary (optional; from manual report or --ncu-csv) ---
    'ncu_report_path',
    'ncu_top_kernel_name',
    'ncu_roofline_bound',
    'ncu_compute_intensity_flop_per_byte',
    'ncu_dram_throughput_gbps',
    'ncu_mem_throughput_pct_of_peak',
)


# -----------------------------------------------------------------------------
# ncu_kernels.csv — one row per kernel per run (optional detailed roofline input)
# -----------------------------------------------------------------------------

NCU_KERNELS_CSV_COLUMNS = (
    'run_id',
    'variant_name',
    'kernel_name',
    'section_name',
    'duration_us',
    'dram_throughput_gbps',
    'sm_throughput_pct',
    'mem_throughput_pct',
    'compute_throughput_pct',
    'l2_hit_rate_pct',
    'dram_bytes_read',
    'dram_bytes_write',
    'roofline_bound_note',
)


# -----------------------------------------------------------------------------
# Default variant matrix (torch.compile and --amp must not be combined in current stack)
# -----------------------------------------------------------------------------

def default_variant_matrix():
    """
    Legacy 3-run matrix. Full M0–M4 × (FP32, AMP) design is in research_experiment_matrix.EXPERIMENT_MATRIX_FP32_AMP.
    """
    from research_experiment_matrix import default_variant_matrix_legacy
    return default_variant_matrix_legacy()
