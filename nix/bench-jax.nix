# JAX reference implementation benchmark environment
{
  pkgs,
  patchesDir,
  rocmShellHook,
}: let
  # ============================================================
  # Source repository
  # ============================================================

  ttt-lm-jax-src = pkgs.fetchFromGitHub {
    owner = "test-time-training";
    repo = "ttt-lm-jax";
    rev = "6f529b124c7fb5879b33c06926408b15add1d82f";
    hash = "sha256-jBGbvBL8nlFpPFAWI3do9hhxiUTkuX9tQO64GdoiwVc=";
  };

  # ============================================================
  # Python packages
  #
  # nixpkgs JAX package currently does not support ROCm,
  # so we package the pre-built wheels.
  # We don't need full JAX, so we hack around some dependencies
  # that aren't readily available in nixpkgs.
  # We use JAX 7.1
  # ============================================================

  jaxPython = pkgs.python312.override {
    self = jaxPython;
    packageOverrides = pyfinal: pyprev: {
      jax = pyfinal.buildPythonPackage rec {
        pname = "jax";
        version = "0.7.1";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/83/81/793d78c91b0546b3b1f08e55fdd97437174171cd7d70e46098f1a4d94b7b/jax-0.7.1-py3-none-any.whl";
          hash = "sha256:19nlvwcv4hjcf0sg1sy9wk69325c260z96an2835aijq1rp5fvh5";
        };
        dependencies = with pyfinal; [jaxlib ml-dtypes numpy opt-einsum scipy];
        pythonImportsCheck = ["jax"];
      };

      jaxlib = pyfinal.buildPythonPackage rec {
        pname = "jaxlib";
        version = "0.7.1";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/0d/50/e37d02e250f5feb755112ec95b1c012a36d48a99209277267037d100f630/jaxlib-0.7.1-cp312-cp312-manylinux_2_27_x86_64.whl";
          hash = "sha256:1saj79rl23mpbfw78m56pg8k1i0np9fa649pvm029y4paw9x7avl";
        };
        nativeBuildInputs = [pkgs.autoPatchelfHook];
        buildInputs = [(pkgs.lib.getLib pkgs.stdenv.cc.cc)];
        dependencies = with pyfinal; [absl-py flatbuffers ml-dtypes scipy];
        pythonImportsCheck = ["jaxlib"];
      };

      flax = pyprev.flax.overridePythonAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pyfinal.pythonRelaxDepsHook];
        pythonRelaxDeps = ["jax"];
        doCheck = false;
      });

      optax = pyprev.optax.overridePythonAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pyfinal.pythonRelaxDepsHook];
        pythonRelaxDeps = ["jax"];
      });

      chex = pyprev.chex.overridePythonAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pyfinal.pythonRelaxDepsHook];
        pythonRelaxDeps = ["jax"];
      });

      mlxu = pyfinal.buildPythonPackage rec {
        pname = "mlxu";
        version = "0.2.0";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          hash = "sha256-O85yWb36jjciRxlx5zB43L4uHfVK9HWDkjasBJQhOI8=";
        };
        build-system = with pyfinal; [setuptools];
        dependencies = with pyfinal; [
          absl-py
          ml-collections
          pyyaml
          requests
          gcsfs
          wandb
          cloudpickle
          numpy
        ];
        pythonImportsCheck = ["mlxu"];
      };
    };
  };

  # ROCm 7.1.1 JAX plugin wheels
  jax-rocm7-pjrt = jaxPython.pkgs.buildPythonPackage rec {
    pname = "jax-rocm7-pjrt";
    version = "0.7.1";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/jax_rocm7_pjrt-0.7.1-py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256:0jz15c4379v6r2vp0byx2bz547g7am5336x1acmddgks5vi3437p";
    };
    nativeBuildInputs = [pkgs.autoPatchelfHook];
    buildInputs = [
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocm-runtime
      pkgs.rocmPackages.rccl
      pkgs.rocmPackages.hipsolver
      pkgs.rocmPackages.rocsolver
      pkgs.rocmPackages.hipfft
      pkgs.rocmPackages.miopen
      pkgs.rocmPackages.rocprofiler-register
      pkgs.numactl
      (pkgs.lib.getLib pkgs.stdenv.cc.cc)
    ];
    autoPatchelfIgnoreMissingDeps = ["librocprofiler-sdk*.so*" "libhipfftw.so*" "libhipsolver_fortran.so*"];
    dontCheckPythonImports = true;
  };

  jax-rocm7-plugin = jaxPython.pkgs.buildPythonPackage rec {
    pname = "jax-rocm7-plugin";
    version = "0.7.1";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/jax_rocm7_plugin-0.7.1-cp312-cp312-manylinux_2_28_x86_64.whl";
      hash = "sha256:11zcfrqjx48aajb6dqqzgs96s4v1xrs587flkllm5kl2b9kzgwaz";
    };
    nativeBuildInputs = [pkgs.autoPatchelfHook];
    buildInputs = [
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocm-runtime
      pkgs.rocmPackages.rccl
      pkgs.rocmPackages.hipblas
      pkgs.rocmPackages.hipsparse
      pkgs.rocmPackages.hipsolver
      pkgs.rocmPackages.rocsolver
      pkgs.rocmPackages.hipfft
      pkgs.rocmPackages.miopen
      pkgs.rocmPackages.rocprofiler-register
      pkgs.numactl
      (pkgs.lib.getLib pkgs.stdenv.cc.cc)
    ];
    autoPatchelfIgnoreMissingDeps = ["librocprofiler-sdk*.so*" "libhipsparselt.so*" "libhipfftw.so*" "libhipsolver_fortran.so*"];
    dependencies = [jax-rocm7-pjrt];
    dontCheckPythonImports = true;
  };

  # ============================================================
  # Patched source & Python environment
  # ============================================================

  ttt-lm-jax-patched = pkgs.stdenv.mkDerivation {
    pname = "ttt-lm-jax-patched";
    version = "0.0.1";
    src = ttt-lm-jax-src;
    patches = [
      "${patchesDir}/jax-utils-api.patch"
      "${patchesDir}/jax-bfloat16-scan-carry.patch"
      "${patchesDir}/jax-h32-config.patch"
    ];
    installPhase = ''
      mkdir -p $out
      cp -r . $out/
      cp ${patchesDir}/jax-benchmark.py $out/benchmark.py
    '';
  };

  jaxPythonEnv = jaxPython.withPackages (ps:
    with ps; [
      jax
      jaxlib
      jax-rocm7-plugin
      numpy
      matplotlib
      tqdm
      transformers
      datasets
      einops
      scipy
      ml-collections
      mlxu
      flax
      optax
      torch
    ]);
  envSetup =
    rocmShellHook
    + ''
      export PYTHONPATH="${ttt-lm-jax-patched}:''${PYTHONPATH:-}"
      export JAX_COMPILATION_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/jax-compilation-cache"
      cd ${ttt-lm-jax-patched}
    '';
in {
  app = {
    type = "app";
    program = "${pkgs.writeShellScript "bench-jax" ''
      ${envSetup}
      exec ${jaxPythonEnv}/bin/python benchmark.py "$@"
    ''}";
  };

  devShell = pkgs.mkShell {
    buildInputs = [
      jaxPythonEnv
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocm-smi
      pkgs.llvmPackages.lld
    ];
    shellHook =
      envSetup
      + ''
        if [[ $- == *i* ]]; then
          echo ""
          echo "TTT-LM JAX (ROCm) Benchmark Shell"
          echo "=================================="
          echo ""
          echo "JAX version: $(python -c 'import jax; print(jax.__version__)')"
          echo "JAX devices: $(python -c 'import jax; print(jax.devices())')"
          echo ""
          echo "Run benchmark:"
          echo "  python benchmark.py --model-size 125m --ttt-type linear"
        fi
      '';
  };
}
