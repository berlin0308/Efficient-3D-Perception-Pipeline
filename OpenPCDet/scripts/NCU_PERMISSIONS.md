# Nsight Compute (ncu) Permission and Driver Notes

If `ncu` runs but reports the following, use the steps below.

## 1. ERR_NVGPUCTRPERM – no permission to GPU performance counters

**Error:** `The user does not have permission to access NVIDIA GPU Performance Counters on the target device.`

**Fix on Linux (needs root once):** Allow all users to use GPU profiling by setting the NVIDIA kernel module option:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u
```

Then **reboot** (or unload/reload the nvidia module if you can do that safely). After reboot, run `ncu` again as your normal user.

Reference: https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters

## 2. Cuda driver is not compatible with Nsight Compute

**Error:** `Cuda driver is not compatible with Nsight Compute` or `Failed to load Nsight Compute CUDA modules.`

This usually means the **driver version** on the machine is older than what your Nsight Compute version supports. Nsight Compute 2025.4 typically needs a driver that supports CUDA 13 (e.g. 590.x or newer on Linux).

**Options:**

- **Upgrade the NVIDIA driver** on the machine to a version supported by your Nsight Compute release (see [Nsight Compute release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html) for “System Requirements” / “Recommended Drivers”).
- **Install an older Nsight Compute** that matches your current driver (e.g. from [Get Started – Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started), use an older version from the release list).

Check your driver version:

```bash
nvidia-smi
```

Then compare with the “Recommended Drivers” section in the Nsight Compute documentation for your installed version.
