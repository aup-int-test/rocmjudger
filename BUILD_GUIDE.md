# ROCm Judger Build Guide

## Prerequisites

Before building, ensure your system has the required ROCm environment set up.

### Check Your Environment

**1. Verify ROCm installation:**
```bash
hipcc --version
# Expected output: HIP version with ROCm information
```

**2. Check available GPUs:**
```bash
rocminfo | grep "Name:"
# Should list your AMD GPU(s)
```

**3. Verify rocm-smi works:**
```bash
rocm-smi
# Should display GPU information and status
```

**4. Check user permissions:**
```bash
groups | grep -E 'video|render'
# Should show you're in video and/or render groups
```

**5. Verify Python (for test generation):**
```bash
python3 --version
# Python 3.x required for geninput.py scripts
```

**6. Check GPU architecture:**
```bash
rocminfo | grep "gfx"
# Shows your GPU's architecture (e.g., gfx90a, gfx908, gfx1100)
```

### Fix Common Issues

**If hipcc not found:**
```bash
export PATH=/opt/rocm/bin:$PATH
# Add to ~/.bashrc for persistence
```

**If permission denied:**
```bash
sudo usermod -a -G video,render $USER
# Log out and log back in
```

## GPU Architecture Configuration

The build system supports centralized GPU architecture selection from the top-level Makefile.

### Supported GPU Architectures

| Architecture | GPU Model | Flag |
|-------------|-----------|------|
| `gfx90a` | AMD MI210 | `--offload-arch=gfx90a` |
| `gfx908` | AMD MI100 | `--offload-arch=gfx908` |
| `gfx1100` | AMD Radeon W7900 | `--offload-arch=gfx1100` |

## Build Methods

### Method 1: Edit Top-Level Makefile (Recommended)

Edit `easy/Makefile` or `meduim/Makefile` and uncomment the desired GPU architecture:

```makefile
# GPU Architecture Selection
# Uncomment ONE of the following lines:
GPU_ARCH = gfx90a   # AMD MI210
# GPU_ARCH = gfx908   # AMD MI100
# GPU_ARCH = gfx1100  # AMD Radeon W7900
```

Then build:
```bash
cd easy
make
```

### Method 2: Command Line Override

Set GPU architecture from the command line without editing files:

```bash
# Build for MI210
cd easy
make GPU_ARCH=gfx90a

# Build for MI100
make GPU_ARCH=gfx908

# Build for Radeon W7900
make GPU_ARCH=gfx1100
```

### Method 3: Auto-Detection (Default)

If `GPU_ARCH` is not set, ROCm will auto-detect your GPU:

```bash
cd easy
make
```

## Build Commands

### Build All Challenges

```bash
cd easy
make           # Stop on first error

# OR

make force     # Continue even if errors occur
```

### Build Specific Challenge

```bash
cd easy
make vector_addition

# With specific GPU
make vector_addition GPU_ARCH=gfx90a
```

### Build Individual Directory

```bash
cd easy/vector_addition
make

# With specific GPU
make GPU_ARCH=gfx90a
```

### Clean

```bash
cd easy
make clean     # Clean all subdirectories
```

## Complete Examples

### Example 1: Build All Easy Challenges for MI210
```bash
cd easy
# Edit Makefile, uncomment: GPU_ARCH = gfx90a
make
```

Or without editing:
```bash
cd easy
make GPU_ARCH=gfx90a
```

### Example 2: Build Specific Challenge for Radeon W7900
```bash
cd easy
make monte_carlo_integration GPU_ARCH=gfx1100
```

### Example 3: Build Medium Challenges with Force Mode
```bash
cd meduim
make force GPU_ARCH=gfx90a
```

## Architecture Priority

The GPU architecture is determined in this order:

1. **Command line**: `make GPU_ARCH=gfx90a` (highest priority)
2. **Top-level Makefile**: `GPU_ARCH = gfx90a` in `easy/Makefile`
3. **Auto-detection**: ROCm detects installed GPU (lowest priority)

## Troubleshooting

### Check Compilation Flags
To see what flags are being used:
```bash
cd easy/vector_addition
make -n
```

### Verify GPU Architecture
Check available GPUs:
```bash
rocminfo | grep "Name:"
```

### Wrong Architecture Compiled
Make sure to clean before rebuilding:
```bash
make clean
make GPU_ARCH=gfx90a
```

## Directory Structure

```
rocmjudger/
├── easy/
│   ├── Makefile              # Top-level with GPU_ARCH
│   ├── vector_addition/
│   │   └── Makefile          # Uses GPU_ARCH from parent
│   └── ...
└── meduim/
    ├── Makefile              # Top-level with GPU_ARCH
    ├── gemm/
    │   └── Makefile          # Uses GPU_ARCH from parent
    └── ...
```

## Notes

- **Consistent builds**: Using the top-level Makefile ensures all challenges compile for the same GPU
- **No warnings**: All Makefiles include `-Wno-unused-result` to suppress nodiscard warnings
- **Testcase cleanup**: `make clean` removes both executables and generated testcases
- **Mixed CPU/GPU code**: Medium challenges with `.cpp` files use `CXXFLAGS` (no GPU flags) and `.cu` files use `HIPFLAGS` (with GPU flags)
