# Triton kernels reference implementation benchmark environment
{
  pkgs,
  patchesDir,
  rocmShellHook,
}: let
  # ============================================================
  # Source repository
  # ============================================================

  ttt-lm-kernels-src = pkgs.fetchFromGitHub {
    owner = "test-time-training";
    repo = "ttt-lm-kernels";
    rev = "99851e6dcc44060952a5618f0131a5ca6d7f6519";
    hash = "sha256-+SqsWfQfOBHZVzfKugpZiFvAAU0IlpHQFpQxS04j9Ew=";
  };

  # ============================================================
  # Patched source & Python environment
  # ============================================================

  ttt-lm-kernels-patched = pkgs.stdenv.mkDerivation {
    pname = "ttt-lm-kernels-patched";
    version = "0.0.1";
    src = ttt-lm-kernels-src;
    patches = [
      "${patchesDir}/kernels-triton-rocm.patch"
      "${patchesDir}/kernels-all-sizes.patch"
    ];
    installPhase = ''
      mkdir -p $out
      cp -r . $out/
      cp ${patchesDir}/kernels-benchmark.py $out/benchmark.py
    '';
  };

  kernelsPythonEnv = pkgs.pkgsRocm.python313.withPackages (ps:
    with ps; [
      torch
      triton
      transformers
      numpy
      tqdm
      einops
      causal-conv1d
    ]);
  envSetup =
    rocmShellHook
    + ''
      export PYTHONPATH="${ttt-lm-kernels-patched}:''${PYTHONPATH:-}"
      cd ${ttt-lm-kernels-patched}
    '';
in {
  app = {
    type = "app";
    program = "${pkgs.writeShellScript "bench-kernels" ''
      ${envSetup}
      exec ${kernelsPythonEnv}/bin/python benchmark.py "$@"
    ''}";
  };

  devShell = pkgs.mkShell {
    buildInputs = [
      kernelsPythonEnv
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocm-smi
      pkgs.llvmPackages.lld
    ];
    shellHook =
      envSetup
      + ''
        if [[ $- == *i* ]]; then
          echo ""
          echo "TTT-LM Kernels (ROCm) Benchmark Shell"
          echo "======================================"
          echo ""
          echo "NOTE: Triton kernels patched for ROCm (fp32 casts for sqrt/sigmoid)"
          echo ""
          echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
          echo "Triton: $(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'N/A')"
          echo "ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())')"
          echo ""
          echo "Run benchmark:"
          echo "  python benchmark.py --model-size 1b --ttt-type linear"
          echo "  python benchmark.py --model-size 1b --ttt-type linear --fast"
        fi
      '';
  };
}
