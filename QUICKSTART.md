# Quickstart

This guide gets you from a fresh machine to a trained TTT model. Pick whichever path matches your setup.

## Option A: Docker (easiest)

No Rust or GPU SDK install needed  - everything is in the image.

### AMD GPU (ROCm)

```bash
docker build -f Dockerfile.rocm -t ttt-rocm .
docker run --rm -it --device /dev/kfd --device /dev/dri --group-add video \
  -v $(pwd)/data:/workspace/dataset \
  ttt-rocm
```

### NVIDIA GPU (CUDA)

```bash
docker build -f Dockerfile.cuda -t ttt-cuda .
docker run --rm -it --gpus all \
  -v $(pwd)/data:/workspace/dataset \
  ttt-cuda
```

Inside the container, binaries are already on `$PATH`:

```bash
ttt train --size 60m --layer-type linear --epochs 5 --batch 32
ttt generate ./artifacts "Once upon a time"
```

## Option B: Manual install

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### 2. Install system dependencies

**Ubuntu/Debian:**

```bash
sudo apt-get install build-essential pkg-config cmake libssl-dev libsqlite3-dev
```

**Fedora:**

```bash
sudo dnf install gcc gcc-c++ pkg-config cmake openssl-devel sqlite-devel
```

**Arch:**

```bash
sudo pacman -S base-devel pkg-config cmake openssl sqlite
```

**macOS (CPU only):**

```bash
brew install cmake openssl sqlite pkg-config
```

### 3. Install GPU SDK

Pick one. Skip this step if you only want CPU mode.

**ROCm (AMD):** Follow the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for your distro. You need ROCm 6.x or 7.x.

**CUDA (NVIDIA):** Follow the [CUDA toolkit install guide](https://developer.nvidia.com/cuda-downloads). You need CUDA 12.x.

### 4. Build

```bash
cd rust

# Pick your backend:
cargo build --release --features rocm    # AMD
cargo build --release --features cuda    # NVIDIA
cargo build --release --features cpu     # No GPU
```

The binaries land in `rust/target/release/`:
- `ttt`  - main CLI
- `ttt-bench`  - benchmarking tool
- `ttt-harness`  - multi-run coordinator

### 5. Train

```bash
# Minimal training run on TinyStories (downloads automatically)
./target/release/ttt train --size 60m --layer-type linear --epochs 5 --batch 32

# The model and checkpoints are saved to ./artifacts/ by default
```

### 6. Generate text

```bash
./target/release/ttt generate ./artifacts "Once upon a time"

# Or start an interactive session
./target/release/ttt interactive ./artifacts
```

### 7. Evaluate

```bash
./target/release/ttt eval ./artifacts --samples 1000
```

## Option C: Nix

If you have Nix with flakes enabled, this is the fastest path:

```bash
nix develop            # drops you into a shell with everything
cd rust
cargo build --release --features rocm
```

## Common issues

**"Can't use tensor maps on HIP"**  - You passed `--features rocm,bf16` in debug mode. Burn has an erroneous debug assertion that fails when trying to use `bf16` on AMD. When building in release mode, this check is skipped.

**"linker cc not found"**  - Install `build-essential` (Ubuntu) or `base-devel` (Arch) for the C toolchain.

**"failed to run custom build command for openssl-sys"**  - Install `libssl-dev` (Ubuntu) or `openssl-devel` (Fedora).

**GPU not detected**  - Make sure your GPU SDK is installed and `rocminfo` (AMD) or `nvidia-smi` (NVIDIA) works before building.
