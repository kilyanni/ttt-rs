# PyTorch reference implementation benchmark environment
{
  pkgs,
  patchesDir,
  rocmShellHook,
}: let
  # ============================================================
  # Source repository
  # ============================================================

  ttt-lm-pytorch-src = pkgs.fetchFromGitHub {
    owner = "test-time-training";
    repo = "ttt-lm-pytorch";
    rev = "cd831db10c8c9a0f6340f02da5613316a8a92b67";
    hash = "sha256-g+MR4TQHptcfaWlFanBBzY1k0ip2aCHP9AL0a1qGj8k=";
  };

  # ============================================================
  # Patched source & Python environment
  # ============================================================

  ttt-lm-pytorch-patched = pkgs.stdenv.mkDerivation {
    pname = "ttt-lm-pytorch-patched";
    version = "0.0.1";
    src = ttt-lm-pytorch-src;
    patches = [
      "${patchesDir}/pytorch-h32-config.patch"
    ];
    installPhase = ''
      mkdir -p $out
      cp -r . $out/
      cp ${patchesDir}/pytorch-benchmark.py $out/benchmark.py
    '';
  };

  pytorchPythonEnv = pkgs.pkgsRocm.python313.withPackages (ps:
    with ps; [
      torch
      transformers
      numpy
      tqdm
      einops
      causal-conv1d
    ]);
  envSetup =
    rocmShellHook
    + ''
      export PYTHONPATH="${ttt-lm-pytorch-patched}:''${PYTHONPATH:-}"
      export TORCHINDUCTOR_FX_GRAPH_CACHE=1
      export TORCHINDUCTOR_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/torch-inductor-cache"
      cd ${ttt-lm-pytorch-patched}
    '';
in {
  app = {
    type = "app";
    program = "${pkgs.writeShellScript "bench-pytorch" ''
      ${envSetup}
      exec ${pytorchPythonEnv}/bin/python benchmark.py "$@"
    ''}";
  };

  devShell = pkgs.mkShell {
    buildInputs = [
      pytorchPythonEnv
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocm-smi
    ];
    shellHook =
      envSetup
      + ''
        if [[ $- == *i* ]]; then
          echo ""
          echo "TTT-LM PyTorch (ROCm) Benchmark Shell"
          echo "======================================"
          echo ""
          echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
          echo "ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())')"
          echo ""
          echo "Run benchmark:"
          echo "  python benchmark.py --model-size 125m --ttt-type linear"
        fi
      '';
  };
}
