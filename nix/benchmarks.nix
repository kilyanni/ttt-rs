# Reference implementation benchmark environments (JAX, PyTorch, Triton Kernels)
# Used for comparing against the Burn/CubeCL implementation.
{
  pkgs,
}: let
  # ============================================================
  # Per-implementation modules
  # ============================================================

  patchesDir = ./patches;

  jax = import ./bench-jax.nix {inherit pkgs patchesDir rocmShellHook;};
  pytorch = import ./bench-pytorch.nix {inherit pkgs patchesDir rocmShellHook;};
  kernels = import ./bench-kernels.nix {inherit pkgs patchesDir rocmShellHook;};

  # ============================================================
  # Shared: ROCm stub libraries & shell hook
  # ============================================================

  # We stub some ROCm libraries that aren't directly used,
  # but cause linker issues when absent.
  rocm-stubs = pkgs.stdenv.mkDerivation {
    pname = "rocm-stubs";
    version = "0.0.1";
    dontUnpack = true;
    buildPhase = ''
      mkdir -p $out/lib
      cat > stub.c << 'EOF'
      #include <stdint.h>
      int rocprofiler_assign_callback_thread(void* a, void* b) { return -1; }
      int rocprofiler_configure_buffer_tracing_service(void* a, void* b, void* c) { return -1; }
      int rocprofiler_configure_callback_tracing_service(void* a, void* b, void* c, void* d) { return -1; }
      int rocprofiler_context_is_valid(void* a) { return 0; }
      int rocprofiler_create_buffer(void* a, void* b, void* c, void* d, void* e, void* f) { return -1; }
      int rocprofiler_create_callback_thread(void* a) { return -1; }
      int rocprofiler_create_context(void* a) { return -1; }
      int rocprofiler_force_configure(void* a) { return -1; }
      const char* rocprofiler_get_status_string(int s) { return "stub"; }
      int rocprofiler_get_timestamp(uint64_t* ts) { if(ts) *ts = 0; return 0; }
      int rocprofiler_iterate_callback_tracing_kind_operations(void* a, void* b, void* c) { return 0; }
      int rocprofiler_iterate_callback_tracing_kinds(void* a, void* b) { return 0; }
      int rocprofiler_query_available_agents(void* a, void* b, void* c, void* d) { return 0; }
      int rocprofiler_query_callback_tracing_kind_name(void* a, void* b, void* c) { return -1; }
      int rocprofiler_query_callback_tracing_kind_operation_name(void* a, void* b, void* c, void* d) { return -1; }
      int rocprofiler_start_context(void* a) { return -1; }
      int rocprofiler_stop_context(void* a) { return -1; }
      void roctxRangePushA(const char* s) {}
      int roctxRangePop() { return 0; }
      void roctxMarkA(const char* s) {}
      EOF
      $CC -shared -fPIC stub.c -o $out/lib/librocprofiler-sdk-attach.so.1
      $CC -shared -fPIC stub.c -o $out/lib/librocprofiler-sdk.so.1
      $CC -shared -fPIC stub.c -o $out/lib/librocprofiler-sdk-roctx.so.1
      $CC -shared -fPIC stub.c -o $out/lib/librocprofiler-sdk-rocpd.so.1
      $CC -shared -fPIC stub.c -o $out/lib/libhipfftw.so
      $CC -shared -fPIC stub.c -o $out/lib/libhipsolver_fortran.so.1
      $CC -shared -fPIC stub.c -o $out/lib/libhipsparselt.so
      ln -s librocprofiler-sdk-attach.so.1 $out/lib/librocprofiler-sdk-attach.so
      ln -s librocprofiler-sdk.so.1 $out/lib/librocprofiler-sdk.so
      ln -s librocprofiler-sdk-rocpd.so.1 $out/lib/librocprofiler-sdk-rocpd.so
      ln -s libhipsolver_fortran.so.1 $out/lib/libhipsolver_fortran.so
    '';
  };

  rocmShellHook = ''
    # RX 6800 is gfx1030 (RDNA2) - set to 10.3.0
    export HSA_OVERRIDE_GFX_VERSION=10.3.0

    # JAX ROCm needs ROCM_PATH and lld
    export ROCM_PATH="${pkgs.rocmPackages.clr}"
    export PATH="${pkgs.llvmPackages.lld}/bin:$PATH"
    export TRITON_HIP_LLD_PATH="${pkgs.llvmPackages.lld}/bin/ld.lld"

    # ROCm library paths + stubs
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      rocm-stubs
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rccl
      pkgs.rocmPackages.miopen
      pkgs.rocmPackages.hipblas
      pkgs.rocmPackages.hipfft
      pkgs.rocmPackages.hipsolver
      pkgs.rocmPackages.hipsparse
      pkgs.rocmPackages.rocsolver
    ]}:''${LD_LIBRARY_PATH:-}"
  '';
in {
  apps = {
    bench-jax = jax.app;
    bench-pytorch = pytorch.app;
    bench-kernels = kernels.app;
  };

  devShells = {
    bench-jax = jax.devShell;
    bench-pytorch = pytorch.devShell;
    bench-kernels = kernels.devShell;
  };
}
