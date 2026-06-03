#!/bin/bash
# =============================================================================
# ROCm 7.2.2 Environment Configuration for AMD gfx1102 (Navi 33)
# =============================================================================
# Usage: source config/rocm_env.sh
#
# This script configures optimal settings for AMD Radeon RX 7600/7600 XT
# (gfx1102, Navi 33) with ROCm 7.2.2 on Ubuntu 24.04 (Noble)
#
# Key optimizations:
# - Native gfx1102 support (no fallback needed with ROCm 7.2+)
# - Memory optimization for 8GB VRAM
# - MIOpen kernel tuning for RDNA 3
# - HIP performance flags
# =============================================================================

set -euo pipefail

# ── ROCm Installation Paths ──────────────────────────────────────────────────
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# ── gfx1102 (Navi 33) Architecture Settings ───────────────────────────────────
# ROCm 7.2.2 has native gfx1102 support - override is optional but helps fallback
# For Navi 33 (RX 7600/7600 XT), gfx1102 maps to VIRTINST=11.0.2
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.2}"

# ── Memory Optimization for 8GB VRAM ─────────────────────────────────────────
# Enable SDMA (System DMA) for faster memory transfers
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-1}"

# Fine-grained memory for better memory management on limited VRAM
export HSA_ENABLE_FINE_GRAINED_MEMORY="${HSA_ENABLE_FINE_GRAINED_MEMORY:-1}"

# Force device kernel arguments (reduces host memory pressure)
export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"

# Memory allocator settings for ROCm 7.2+
export PYTORCH_HIP_ALLOC_CONFLICT_TEST="${PYTORCH_HIP_ALLOC_CONFLICT_TEST:-1}"

# ── MIOpen (AMD's cuDNN equivalent) Tuning ────────────────────────────────────
# Reduce logging overhead
export MIOPEN_LOG_LEVEL="${MIOPEN_LOG_LEVEL:-3}"

# Fast kernel selection (1 = Normal, 2 = Exhaustive search)
# Use 1 for training (faster startup), 2 for inference benchmarking
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-1}"

# Convolution algorithm selection
# Direct convolutions are faster for small kernels (like DDQN 5x5)
export MIOPEN_DEBUG_CONV_DIRECT="${MIOPEN_DEBUG_CONV_DIRECT:-1}"

# Disable FFT convolutions (not optimal for small kernels)
export MIOPEN_DEBUG_CONV_FFT="${MIOPEN_DEBUG_CONV_FFT:-0}"

# Enable MIOpen implicit GEMM for better performance
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM="${MIOPEN_DEBUG_CONV_IMPLICIT_GEMM:-1}"

# ── HIP Performance Flags ────────────────────────────────────────────────────
# Use MIOpen backend for PyTorch
export USE_MIOPEN="${USE_MIOPEN:-1}"

# Disable Ahead-Of-Time compilation (faster startup, slightly slower first run)
export TORCH_ROCM_AOT="${TORCH_ROCM_AOT:-0}"

# HIP synchronization mode (0 = default, 1 = precise but slower)
export HIP_VISIBILITY_TIMEOUT="${HIP_VISIBILITY_TIMEOUT:-0}"

# ── Multi-threading (RDNA 3 has 32 CUs) ─────────────────────────────────────
# Optimal thread count for Navi 33 (32 CUs, good for inference)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# ── PyTorch ROCm Specific ───────────────────────────────────────────────────
# Enable ROCm profiler (useful for debugging performance)
export HIP_PROFILE="${HIP_PROFILE:-0}"  # Set to 1 for profiling

# Disable NCCL for single-GPU (reduces overhead)
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

# ── Debugging (optional, disabled by default) ───────────────────────────────
# Set to 1 to enable debugging output
export HSA_ENABLE_DEBUG="${HSA_ENABLE_DEBUG:-0}"
export AMD_LOG_LEVEL="${AMD_LOG_LEVEL:-0}"  # 0=off, 1=error, 2=warn, 3=info

# ── GPU Memory Pool Settings ────────────────────────────────────────────────
# For 8GB VRAM: reserve 6GB for PyTorch, leave 2GB for system
# PyTorch will use memory pool by default in ROCm 7.2+
export PYTORCH_HIP_MEMORY_POOL_SIZE="${PYTORCH_HIP_MEMORY_POOL_SIZE:-6144}"  # MB

# ── BF16/FP16 Support Detection ─────────────────────────────────────────────
# RDNA 3 (Navi 33) supports native BF16 and dual-issue FP16
# These are informational flags that Python code can check
export ROCM_NATIVE_BF16="${ROCM_NATIVE_BF16:-1}"
export ROCM_DUAL_ISSUE_FP16="${ROCM_DUAL_ISSUE_FP16:-1}"

# ── Verification Function ───────────────────────────────────────────────────
rocm_verify() {
    echo "========================================"
    echo "ROCm Environment Verification"
    echo "========================================"

    # Check ROCm installation
    if [ -d "${ROCM_PATH}" ]; then
        echo "✓ ROCm path: ${ROCM_PATH}"
    else
        echo "✗ ROCm not found at ${ROCM_PATH}"
        return 1
    fi

    # Check rocm-smi
    if command -v rocm-smi &>/dev/null; then
        echo "✓ rocm-smi available"
        echo ""
        echo "GPU Status:"
        rocm-smi --showid --showproductname --showmeminfo vram 2>/dev/null || true
    else
        echo "⚠ rocm-smi not in PATH"
    fi

    # Check Python/PyTorch
    if command -v python3 &>/dev/null; then
        echo ""
        echo "PyTorch ROCm Status:"
        python3 -c "
import sys
try:
    import torch
    print(f'  PyTorch: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  Device: {torch.cuda.get_device_name(0)}')
        props = torch.cuda.get_device_properties(0)
        print(f'  VRAM: {props.total_memory / 1024**3:.1f} GB')
        print(f'  Compute: {props.major}.{props.minor}')
except ImportError:
    print('  PyTorch not installed')
except Exception as e:
    print(f'  Error: {e}')
" 2>/dev/null || echo "  Unable to check PyTorch"
    fi

    echo ""
    echo "========================================"
    echo "Environment Variables:"
    echo "========================================"
    echo "  ROCM_PATH: ${ROCM_PATH:-not set}"
    echo "  GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-default}"
    echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS:-default}"
    echo "  MIOPEN_FIND_MODE: ${MIOPEN_FIND_MODE:-default}"
    echo "  USE_MIOPEN: ${USE_MIOPEN:-default}"
    echo ""
}

# ── Print status if sourced directly ────────────────────────────────────────
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    echo "Usage: source config/rocm_env.sh"
    echo ""
    echo "Options:"
    echo "  source config/rocm_env.sh      # Configure environment"
    echo "  source config/rocm_env.sh && rocm_verify  # Verify setup"
    echo ""
    rocm_verify
else
    # Sourced - print brief status
    echo "ROCm 7.2.2 environment configured for gfx1102 (Navi 33)"
    echo "  GFX: ${HSA_OVERRIDE_GFX_VERSION}, Threads: ${OMP_NUM_THREADS}, MIOpen: enabled"
fi
