"""
AMD ROCm-Specific Constants for Trading Bot Optimization
========================================================

Optimized parameters for AMD Radeon RX 7600/7600 XT (gfx1102, Navi 33)
with 8GB VRAM and ROCm 7.2.2+.

Key characteristics:
- 32 Compute Units (2048 stream processors)
- 8GB VRAM (memory-constrained)
- RDNA 3 architecture with native BF16 support
- Dual-issue FP16 (2x FP16 throughput via FP32)
- 128-bit memory bus

Usage:
    from src.constants_amd import AMD_CONFIG, get_optimal_batch_size

Author: AI Trading System
Version: 1.0.0
"""

import os
from typing import Any

# =============================================================================
# GPU Hardware Constants
# =============================================================================

# Navi 33 (gfx1102) specifications
NAVI33_COMPUTE_UNITS: int = 32
NAVI33_STREAM_PROCESSORS: int = 2048
NAVI33_VRAM_GB: float = 8.0  # 16GB on XT variant
NAVI33_MEMORY_BUS_BITS: int = 128
NAVI33_BASE_CLOCK_MHZ: int = 1720
NAVI33_BOOST_CLOCK_MHZ: int = 2650

# RDNA 3 capabilities
RDNA3_NATIVE_BF16: bool = True  # Native BF16 support
RDNA3_DUAL_ISSUE_FP16: bool = True  # 2x FP16 throughput via FP32
RDNA3_WMMA_INT8: bool = True  # INT8 matrix multiply support


# =============================================================================
# Memory-Optimized Buffer Sizes
# =============================================================================

# For 8GB VRAM: reduced buffer sizes to prevent OOM
# Original: TRIGGER_BUFFER_CAPACITY = 100_000, HARVESTER_BUFFER_CAPACITY = 100_000
# Memory savings: ~50% reduction

AMD_TRIGGER_BUFFER_CAPACITY: int = 50_000  # Reduced from 100K (~1.2GB saved)
AMD_HARVESTER_BUFFER_CAPACITY: int = 50_000  # Reduced from 100K (~1.2GB saved)

# State dimension memory estimate:
# - float32: 4 bytes per value
# - state_dim ~128 (window * features)
# - Each experience: 2 * state_dim * 4 bytes (state + next_state) = ~1KB
# - 50K experiences: ~50MB (manageable)


# =============================================================================
# Training Batch Sizes
# =============================================================================

# Optimal batch sizes for Navi 33 (8GB VRAM)
# Larger batches improve GPU utilization but use more memory

AMD_BATCH_SIZE_SMALL: int = 32  # Safe for all models
AMD_BATCH_SIZE_MEDIUM: int = 64  # Optimal for DDQN (state_dim ~128)
AMD_BATCH_SIZE_LARGE: int = 128  # Only for small state_dim (<64)

# Gradient accumulation to simulate larger batches
AMD_GRADIENT_ACCUMULATION_STEPS: int = 2  # Effective batch = 128 with batch_size=64


# =============================================================================
# Precision Settings
# =============================================================================

# BF16 is preferred on RDNA 3 for numerical stability (native support)
AMD_USE_BF16: bool = True  # Enable BF16 training

# FP16 fallback (dual-issue, less stable than BF16)
AMD_USE_FP16: bool = False  # Disabled in favor of BF16

# Mixed precision policy
AMD_AMP_ENABLED: bool = True  # Automatic Mixed Precision


# =============================================================================
# Inference Optimization
# =============================================================================

# JIT scripting for faster inference (torch.jit.script)
AMD_USE_JIT_INFERENCE: bool = True  # Script model for inference

# Note: torch.compile has limited ROCm support, prefer eager mode
AMD_USE_COMPILE: bool = False  # Disabled for ROCm compatibility


# =============================================================================
# MIOpen Tuning Parameters
# =============================================================================

# Convolution algorithm selection
MIOPEN_FIND_MODE_NORMAL: int = 1  # Fast kernel selection (training)
MIOPEN_FIND_MODE_EXHAUSTIVE: int = 2  # Exhaustive search (benchmarking)

# Default: Normal mode for training
AMD_MIOPEN_FIND_MODE: int = MIOPEN_FIND_MODE_NORMAL

# Enable direct convolutions (optimal for small kernels like DDQN)
AMD_MIOPEN_CONV_DIRECT: bool = True

# Disable FFT convolutions (not optimal for small kernels)
AMD_MIOPEN_CONV_FFT: bool = False


# =============================================================================
# Multi-threading
# =============================================================================

# Optimal thread count for Navi 33 (32 CUs)
# 4 threads provides good balance for inference/training
AMD_NUM_THREADS: int = 4

# MKL threads (if using Intel MKL for CPU operations)
AMD_MKL_NUM_THREADS: int = 4


# =============================================================================
# Memory Pool Configuration
# =============================================================================

# Reserve memory for system (2GB buffer on 8GB card)
AMD_VRAM_SYSTEM_RESERVE_MB: int = 2048

# PyTorch memory pool size (6GB = 8GB - 2GB reserve)
AMD_MEMORY_POOL_SIZE_MB: int = 6144


# =============================================================================
# Environment Variables for ROCm
# =============================================================================

AMD_ROCM_ENV_VARS: dict[str, str] = {
    # Architecture
    "HSA_OVERRIDE_GFX_VERSION": "11.0.2",  # gfx1102 mapping
    # Memory optimization
    "HSA_ENABLE_SDMA": "1",
    "HSA_ENABLE_FINE_GRAINED_MEMORY": "1",
    "HIP_FORCE_DEV_KERNARG": "1",
    # MIOpen
    "MIOPEN_LOG_LEVEL": "3",
    "MIOPEN_FIND_MODE": "1",
    "MIOPEN_DEBUG_CONV_DIRECT": "1",
    "MIOPEN_DEBUG_CONV_FFT": "0",
    # PyTorch
    "USE_MIOPEN": "1",
    "TORCH_ROCM_AOT": "0",
    "PYTORCH_HIP_MEMORY_POOL_SIZE": "6144",
    # Threading
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    # Debugging (disabled by default)
    "HSA_ENABLE_DEBUG": "0",
    "AMD_LOG_LEVEL": "0",
}


# =============================================================================
# Utility Functions
# =============================================================================


def get_optimal_batch_size(state_dim: int, window_size: int = 1) -> int:
    """Calculate optimal batch size based on state dimension and VRAM.

    Args:
        state_dim: Dimension of state vector (e.g., 128 for window=20, features=7)
        window_size: Number of time steps in state (for Conv1d)

    Returns:
        Optimal batch size for Navi 33 with 8GB VRAM
    """
    # Estimate memory per sample (conservative)
    # state + next_state + actions + rewards + dones + weights
    bytes_per_sample = (
        state_dim * 4 * 2  # state + next_state (float32)
        + 4  # actions (int32)
        + 4  # rewards (float32)
        + 1  # dones (bool)
        + 4  # weights (float32)
    )

    # Add Conv1d overhead if window_size > 1
    if window_size > 1:
        bytes_per_sample += state_dim * window_size * 4 * 2  # Conv1d input

    # Available VRAM for training (leave 2GB for system)
    available_vram_bytes = (NAVI33_VRAM_GB - 2.0) * 1024**3

    # Safe batch size (use 50% of available VRAM)
    safe_batch = int((available_vram_bytes * 0.5) / bytes_per_sample)

    # Clamp to reasonable range
    return max(16, min(safe_batch, AMD_BATCH_SIZE_MEDIUM))


def get_buffer_capacity_for_vram(vram_gb: float = NAVI33_VRAM_GB) -> int:
    """Calculate optimal buffer capacity based on available VRAM.

    Args:
        vram_gb: Available VRAM in GB

    Returns:
        Recommended buffer capacity
    """
    # Base capacity for 8GB: 50K experiences
    # Scale linearly with VRAM
    base_capacity = 50_000
    base_vram = 8.0

    return int(base_capacity * (vram_gb / base_vram))


def is_amd_gpu() -> bool:
    """Check if current GPU is AMD (ROCm).

    Returns:
        True if AMD GPU is available via ROCm
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        name = torch.cuda.get_device_name(0).upper()
        return any(x in name for x in ["AMD", "RADEON", "RX", "NAVI", "GFX"])
    except (ImportError, RuntimeError):
        return False


def get_gpu_info() -> dict[str, Any]:
    """Get detailed GPU information.

    Returns:
        Dictionary with GPU details
    """
    info: dict[str, Any] = {
        "available": False,
        "name": "N/A",
        "vram_gb": 0.0,
        "is_amd": False,
        "gfx_arch": "unknown",
        "supports_bf16": False,
        "supports_fp16": True,  # All modern GPUs support FP16
    }

    try:
        import torch

        info["available"] = torch.cuda.is_available()

        if info["available"]:
            info["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = props.total_memory / 1024**3
            info["compute_capability"] = (props.major, props.minor)

            # Detect AMD architecture
            name_upper = info["name"].upper()
            info["is_amd"] = any(x in name_upper for x in ["AMD", "RADEON", "RX", "NAVI", "GFX"])

            if info["is_amd"]:
                # Detect specific architecture
                if "GFX110" in name_upper or "7600" in name_upper or "7900" in name_upper:
                    info["gfx_arch"] = "gfx1100_series"
                    info["supports_bf16"] = True  # RDNA 3
                elif "GFX103" in name_upper or "6800" in name_upper or "6900" in name_upper:
                    info["gfx_arch"] = "gfx1030_series"
                    info["supports_bf16"] = False  # RDNA 2
                elif "GFX10" in name_upper:
                    info["gfx_arch"] = "gfx10_series"
                    info["supports_bf16"] = False  # RDNA 1/2

    except (ImportError, RuntimeError) as e:
        info["error"] = str(e)

    return info


def configure_rocm_environment() -> None:
    """Set ROCm environment variables for optimal performance."""
    for key, value in AMD_ROCM_ENV_VARS.items():
        if key not in os.environ:
            os.environ[key] = value


# =============================================================================
# Initialization
# =============================================================================

# Auto-configure environment when module is imported
_AMD_CONFIGURED = False


def get_amd_config() -> dict[str, Any]:
    """Get complete AMD configuration dictionary.

    Returns:
        Dictionary with all AMD-specific settings
    """
    return {
        "buffer_capacity": {
            "trigger": AMD_TRIGGER_BUFFER_CAPACITY,
            "harvester": AMD_HARVESTER_BUFFER_CAPACITY,
        },
        "batch_size": {
            "small": AMD_BATCH_SIZE_SMALL,
            "medium": AMD_BATCH_SIZE_MEDIUM,
            "large": AMD_BATCH_SIZE_LARGE,
            "gradient_accumulation": AMD_GRADIENT_ACCUMULATION_STEPS,
        },
        "precision": {
            "bf16": AMD_USE_BF16,
            "fp16": AMD_USE_FP16,
            "amp": AMD_AMP_ENABLED,
            "jit": AMD_USE_JIT_INFERENCE,
            "compile": AMD_USE_COMPILE,
        },
        "miopen": {
            "find_mode": AMD_MIOPEN_FIND_MODE,
            "conv_direct": AMD_MIOPEN_CONV_DIRECT,
            "conv_fft": AMD_MIOPEN_CONV_FFT,
        },
        "threading": {
            "omp_threads": AMD_NUM_THREADS,
            "mkl_threads": AMD_MKL_NUM_THREADS,
        },
        "memory": {
            "vram_gb": NAVI33_VRAM_GB,
            "system_reserve_mb": AMD_VRAM_SYSTEM_RESERVE_MB,
            "pool_size_mb": AMD_MEMORY_POOL_SIZE_MB,
        },
        "hardware": {
            "compute_units": NAVI33_COMPUTE_UNITS,
            "stream_processors": NAVI33_STREAM_PROCESSORS,
            "memory_bus_bits": NAVI33_MEMORY_BUS_BITS,
            "native_bf16": RDNA3_NATIVE_BF16,
            "dual_issue_fp16": RDNA3_DUAL_ISSUE_FP16,
        },
    }
