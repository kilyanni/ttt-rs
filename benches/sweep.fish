#!/usr/bin/env fish

# Benchmark sweep script for TTT implementations (JAX, PyTorch, Burn, kernels)

set -q _sweep_dir; or set -g _sweep_dir (status dirname 2>/dev/null)
if not test -d "$_sweep_dir"
    # status dirname fails inside fish -c; fall back to finding sweep.fish relative to cwd
    if test -f (pwd)/sweep.fish
        set -g _sweep_dir (pwd)
    else if test -f (pwd)/benches/sweep.fish
        set -g _sweep_dir (pwd)/benches
    end
end
set -q BENCH_DIR; or set BENCH_DIR $_sweep_dir/results
set -q BENCH_TIMEOUT; or set BENCH_TIMEOUT 300

if test "$argv[1]" = "--help"; or test "$argv[1]" = "-h"
    echo "Usage: ./sweep.fish [OPTION]"
    echo ""
    echo "Run benchmark sweeps across TTT implementations."
    echo ""
    echo "Options:"
    echo "  --help, -h         Show this help message"
    echo "  --status           Show benchmark progress (successful/failed/invalid counts)"
    echo "  --csv              Export results to CSV (redirect to file)"
    echo "  --clear-failures   Remove .failed tombstones to retry failed benchmarks"
    echo "  --clean-invalid    Remove invalid JSON files (<100 bytes) to retry them"
    echo ""
    echo "With no options, runs the full progressive sweep."
    echo ""
    echo "Implementations: jax, pytorch, pytorch-compile, burn, burn-local, kernels"
    echo "Model sizes:     125m, 350m, 760m, 1b (kernels: 1b only)"
    echo "TTT types:       linear, mlp (burn also: fused-naive, fused-tile, fused-tile-multi)"
    echo "Dtypes:          float32, bfloat16 (burn only for bfloat16)"
    echo ""
    echo "Source this script to access run_bench for individual benchmarks:"
    echo "  source ./sweep.fish"
    echo "  run_bench burn 125m linear 2048 16 1 fwd float32          # nix-built (reproducible)"
    echo "  run_bench burn-local 125m linear 2048 16 1 fwd float32    # local ./target/release build"
    echo "  run_bench burn 125m linear 2048 16 1 fwd float32 \"\" inner # inner TTT layer only"
    exit 0
end

if test "$argv[1]" = "--clear-failures"
    echo "Clearing failure tombstones..."
    find $BENCH_DIR -maxdepth 1 -name "*.failed" -delete
    exit 0
end

if test "$argv[1]" = "--status"
    set -l success (find $BENCH_DIR -maxdepth 1 -name "*.json" -size +100c 2>/dev/null | wc -l)
    set -l failed (find $BENCH_DIR -maxdepth 1 -name "*.failed" 2>/dev/null | wc -l)
    set -l invalid (find $BENCH_DIR -maxdepth 1 -name "*.json" -size -100c 2>/dev/null | wc -l)
    echo "Successful: $success"
    echo "Failed (tombstones): $failed"
    echo "Invalid (<100 bytes): $invalid"
    if test "$invalid" -gt 0
        echo ""
        echo "Invalid files (will be retried):"
        find $BENCH_DIR -maxdepth 1 -name "*.json" -size -100c | head -20
        if test "$invalid" -gt 20
            echo "... and "(math $invalid - 20)" more"
        end
    end
    exit 0
end

if test "$argv[1]" = "--clean-invalid"
    echo "Removing invalid JSON files (<100 bytes)..."
    find $BENCH_DIR -maxdepth 1 -name "*.json" -size -100c -delete
    exit 0
end

if test "$argv[1]" = "--csv"
    echo "implementation,model_size,ttt_type,backward,batch,seq_len,mini_batch,dtype,time_ms,throughput,checkpoint_interval,failed,inner_only"
    for f in $BENCH_DIR/*.json
        test (stat -c%s $f) -lt 100; and continue
        set -l is_inner false
        if string match -q '*_inner.json' $f
            set is_inner true
        end
        jq -r --arg inner $is_inner '[.implementation, .model_size, (.ttt_type | gsub("-(?:linear|mlp)$"; "")), .backward, .batch, .seq_len, .mini_batch, .dtype, .time_ms, .throughput, (.checkpoint_interval // ""), false, $inner] | @csv' $f
    end
    # Parse failed tombstones from filename: {impl}_{size}_{model}_len{len}_mb{mb}_b{b}_{dtype}[_bwd][_ckpt{n}][_inner].json.failed
    for f in $BENCH_DIR/*.json.failed
        set base (basename $f .json.failed)
        set parts (string split _ $base)
        set impl $parts[1]
        set size $parts[2]
        set model $parts[3]
        # Find len, mb, b by prefix
        set len ""; set mb ""; set b ""; set dtype ""; set bwd false; set ckpt ""; set is_inner false
        for p in $parts[4..]
            if string match -qr '^len(.+)' $p
                set len (string replace 'len' '' $p)
            else if string match -qr '^mb(.+)' $p
                set mb (string replace 'mb' '' $p)
            else if string match -qr '^b([0-9]+)$' $p
                set b (string replace 'b' '' $p)
            else if string match -qr '^ckpt(.+)' $p
                set ckpt (string replace 'ckpt' '' $p)
            else if test "$p" = bwd
                set bwd true
            else if test "$p" = inner
                set is_inner true
            else
                set dtype $p
            end
        end
        echo "\"$impl\",\"$size\",\"$model\",$bwd,$b,$len,$mb,\"$dtype\",0,0,\"$ckpt\",true,$is_inner"
    end
    exit 0
end

function run_bench
    set impl $argv[1]
    set size $argv[2]
    set model $argv[3]
    set len $argv[4]
    set mb $argv[5]
    set b $argv[6]
    set bwd $argv[7]  # "fwd" or "bwd"
    set dtype $argv[8]  # "float32" or "bfloat16"
    set ckpt $argv[9]  # optional: checkpoint interval (omit or "" to skip)
    set inner $argv[10]  # optional: "inner" for --inner-only

    mkdir -p $BENCH_DIR

    # Skip if seq-len not divisible by mini-batch
    if test (math $len % $mb) -ne 0
        return
    end

    # Build filename and args
    if test "$bwd" = bwd
        set NAME {$impl}_{$size}_{$model}_len{$len}_mb{$mb}_b{$b}_{$dtype}_bwd.json
        set ARGS --model-size $size --ttt-type $model --seq-len $len --mini-batch $mb --batch $b --dtype $dtype --backward --json
    else
        set NAME {$impl}_{$size}_{$model}_len{$len}_mb{$mb}_b{$b}_{$dtype}.json
        set ARGS --model-size $size --ttt-type $model --seq-len $len --mini-batch $mb --batch $b --dtype $dtype --json
    end

    # Append checkpoint interval if specified
    if test -n "$ckpt"
        set NAME (string replace '.json' "_ckpt"$ckpt".json" $NAME)
        set ARGS $ARGS --checkpoint-interval $ckpt
    end

    # Append inner-only if specified
    if test "$inner" = inner
        set NAME (string replace '.json' "_inner.json" $NAME)
        set ARGS $ARGS --inner-only
    end

    # Skip likely OOM configs (conservative guesses for ~16GB VRAM)
    # These are heuristics - actual failures create tombstones and won't retry
    # Comment out to discover actual limits on your hardware
    if test $b -ge 32 -a $len -ge 4096
        echo "SKIP $NAME (likely OOM: batch=$b, len=$len)"
        return
    end
    if test $b -ge 8 -a \( "$size" = "760m" -o "$size" = "1.3b" \) -a $len -ge 4096
        echo "SKIP $NAME (likely OOM: large model + batch + len)"
        return
    end
    if test $b -ge 32 -a \( "$size" = "350m" -o "$size" = "760m" -o "$size" = "1.3b" \)
        echo "SKIP $NAME (likely OOM: batch=32 + large model)"
        return
    end

    # Skip if already done (successful)
    if test -f $BENCH_DIR/$NAME
        set -l filesize (stat -c%s $BENCH_DIR/$NAME 2>/dev/null || echo 0)
        if test $filesize -gt 100
            echo "SKIP $NAME (already done)"
            return
        end
    end

    # Skip if previously failed (tombstone exists)
    if test -f $BENCH_DIR/$NAME.failed
        echo "SKIP $NAME (previously failed)"
        return
    end

    # Skip backward for kernels
    if test "$impl" = kernels -a "$bwd" = bwd
        return
    end

    if test "$impl" = pytorch-compile
        return
    end

    # Skip fused-* models for non-burn implementations
    if string match -q 'fused-*' $model; and test "$impl" != burn -a "$impl" != burn-local
        return
    end

    echo "== $NAME =="

    # Run the appropriate benchmark command, capturing both stdout and stderr
    set -l tmpfile (mktemp)
    echo "output: $tmpfile"
    # All implementations use the project root flake
    set -l project_root (realpath $_sweep_dir/..)
    # --comment is first arg so it's visible in process monitors (nvtop/btm)
    set -q _current_pass; or set -g _current_pass manual
    set -l comment_arg --comment "$_current_pass/$impl"
    if test "$impl" = burn
        # Burn via nix build (reproducible)
        nix build "$project_root#rocm7.ttt" --out-link "$project_root/.bench-result" 2>>$tmpfile
        and timeout $BENCH_TIMEOUT nix develop "$project_root" --command "$project_root/.bench-result/bin/ttt-bench" $comment_arg $ARGS >$tmpfile 2>&1
    else if test "$impl" = burn-local
        # Burn via local cargo build (fast iteration)
        timeout $BENCH_TIMEOUT nix develop "$project_root" --command "$project_root/rust/target/release/ttt-bench" $comment_arg $ARGS >$tmpfile 2>&1
    else if test "$impl" = pytorch-compile
        # PyTorch with torch.compile() via bench-pytorch app
        timeout $BENCH_TIMEOUT nix run "$project_root#bench-pytorch" -- $comment_arg --compile $ARGS >$tmpfile 2>&1
    else
        # JAX, PyTorch, kernels use Python benchmarks via bench-* apps
        timeout $BENCH_TIMEOUT nix run "$project_root#bench-$impl" -- $comment_arg $ARGS >$tmpfile 2>&1
    end

    set output (grep '^{' $tmpfile)

    if test -n "$output"
        echo $output | tee $BENCH_DIR/$NAME
    else
        echo "FAILED"
        # Write tombstone with last 50 lines of output for debugging
        echo "=== FAILED: $impl $size $model len=$len mb=$mb b=$b $bwd $dtype ===" > $BENCH_DIR/$NAME.failed
        tail -50 $tmpfile >> $BENCH_DIR/$NAME.failed
    end
    rm -f $tmpfile
end

function sweep_pass
    set pass $argv[1]
    set impls $argv[2]
    set sizes $argv[3]
    set models $argv[4]
    set lens $argv[5]
    set mbs $argv[6]
    set batches $argv[7]
    set dirs $argv[8]   # "fwd", "bwd", or "fwd,bwd"
    set dtypes $argv[9]
    set ckpts $argv[10]  # optional: checkpoint intervals (omit for none)
    set inner $argv[11]  # optional: "inner" for --inner-only

    echo "=== PASS $pass ==="
    set -g _current_pass $pass

    for impl in (string split , $impls)
        for size in (string split , $sizes)
            for model in (string split , $models)
                for len in (string split , $lens)
                    for mb in (string split , $mbs)
                        for b in (string split , $batches)
                            for dir in (string split , $dirs)
                                for dtype in (string split , $dtypes)
                                    if test -n "$ckpts"
                                        for ckpt in (string split , $ckpts)
                                            run_bench $impl $size $model $len $mb $b $dir $dtype $ckpt $inner
                                        end
                                    else
                                        run_bench $impl $size $model $len $mb $b $dir $dtype "" $inner
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function _sweep_all_figures --argument-names dtype
# Runs all thesis figure sweeps for the given dtype.
# Called twice by run_sweep: first f32 (primary), then bf16 (fallback for OOMs).

# --- Ch. 7: Baseline implementation ---

# @fig-naive-bench: naive burn vs references, linear, all model sizes
sweep_pass "fig-naive-bench-$dtype" \
    "jax,pytorch,pytorch-compile,burn,kernels" \
    "125m,125m-h32,350m,760m,1b" \
    "linear" \
    "2048" \
    "16" \
    "1" \
    "fwd,bwd" \
    "$dtype"

# @fig-naive-bench-mlp: naive burn vs references, MLP, all model sizes
sweep_pass "fig-naive-bench-mlp-$dtype" \
    "jax,pytorch,pytorch-compile,burn,kernels" \
    "125m,125m-h32,350m,760m,1b" \
    "mlp" \
    "2048" \
    "16" \
    "1" \
    "fwd,bwd" \
    "$dtype"

# --- Ch. 8: Optimization ---

# @fig-fusion: kernel fusion speedup (naive vs fused)
sweep_pass "fig-fusion-$dtype" \
    "burn" \
    "125m,125m-h32" \
    "linear,fused-naive" \
    "512" \
    "16" \
    "1" \
    "fwd,bwd" \
    "$dtype"

# @fig-mixed-precision: bf16 vs f32 across model sizes
sweep_pass "fig-mixed-precision-$dtype" \
    "burn" \
    "125m,125m-h32,350m,760m,1b" \
    "fused-naive,fused-naive-multi,fused-tile,fused-tile-multi" \
    "512" \
    "16" \
    "1" \
    "fwd" \
    "float32,bfloat16"

# @fig-bf16-saturation: bf16 vs f32 batch scaling (occupancy benefit)
sweep_pass "fig-bf16-saturation-$dtype" \
    "burn" \
    "125m,125m-h32" \
    "fused-naive,fused-naive-multi,fused-tile,fused-tile-multi" \
    "512" \
    "16" \
    "1,2,4,8,16" \
    "fwd" \
    "float32,bfloat16"

# @fig-fused-tiled: fused vs tiled comparison
sweep_pass "fig-fused-tiled-$dtype" \
    "burn" \
    "125m,125m-h32" \
    "fused-naive,fused-naive-multi,fused-tile,fused-tile-multi" \
    "512" \
    "16" \
    "1" \
    "fwd,bwd" \
    "$dtype"

# @fig-saturation: batch scaling for fused vs tiled
sweep_pass "fig-saturation-1-$dtype" \
    "burn" \
    "125m,125m-h32" \
    "fused-naive,fused-naive-multi,fused-tile,fused-tile-multi" \
    "2048" \
    "16" \
    "1" \
    "fwd" \
    "$dtype"
sweep_pass "fig-saturation-2-$dtype" \
    "burn" \
    "125m,125m-h32" \
    "fused-naive-linear,fused-tile-linear" \
    "2048" \
    "16" \
    "2,4,8,16" \
    "fwd" \
    "$dtype"

# @fig-staging: staged vs unstaged across sequence lengths
sweep_pass "fig-staging-$dtype" \
    "burn" \
    "125m,125m-h32" \
    "fused-naive,fused-naive-multi,fused-tile,fused-tile-multi" \
    "512,2048,4096,8192" \
    "16" \
    "1" \
    "fwd" \
    "$dtype"

# @fig-checkpointing: gradient checkpointing interval scaling
sweep_pass "fig-checkpointing-$dtype" \
    "burn" \
    "125m,125m-h32" \
    "fused-naive-multi,fused-tile-multi" \
    "2048" \
    "16" \
    "1" \
    "bwd" \
    "$dtype" \
    "1,2,4,8,16"

# @fig-final-benchmarks: final forward comparison, all impls
sweep_pass "fig-final-fwd-ref-$dtype" \
    "jax,pytorch,pytorch-compile,kernels" \
    "125m,125m-h32,350m,760m,1b" \
    "linear" \
    "2048" \
    "16" \
    "1" \
    "fwd" \
    "$dtype"
sweep_pass "fig-final-fwd-burn-$dtype" \
    "burn" \
    "125m,125m-h32,350m,760m,1b" \
    "fused-naive,fused-naive-multi,fused-tile,fused-tile-multi" \
    "2048" \
    "16" \
    "1" \
    "fwd" \
    "$dtype"

# @fig-final-benchmarks-bwd: final backward comparison
sweep_pass "fig-final-bwd-ref-$dtype" \
    "jax,pytorch,pytorch-compile" \
    "125m,125m-h32,350m,760m,1b" \
    "linear" \
    "2048" \
    "16" \
    "1" \
    "bwd" \
    "$dtype"
sweep_pass "fig-final-bwd-burn-$dtype" \
    "burn" \
    "125m,125m-h32,350m,760m,1b" \
    "fused-naive,fused-naive-multi,fused-tile,fused-tile-multi" \
    "2048" \
    "16" \
    "1" \
    "bwd" \
    "$dtype"

# @fig-mini-batch-scaling: throughput vs mini-batch size
sweep_pass "fig-mini-batch-scaling-$dtype" \
    "jax,pytorch,pytorch-compile,burn,kernels" \
    "125m,125m-h32" \
    "linear,fused-naive,fused-naive-multi,fused-tile,fused-tile-multi" \
    "2048" \
    "8,16,32,64" \
    "1" \
    "fwd" \
    "$dtype"

# @fig-inner-only: inner TTT layer across all backends, batch scaling
sweep_pass "fig-inner-$dtype" \
    "jax,pytorch,pytorch-compile,burn,kernels" \
    "125m,125m-h32" \
    "linear,fused-naive,fused-naive-multi,fused-tile,fused-tile-multi" \
    "2048" \
    "16" \
    "1,2,4,8,16" \
    "fwd,bwd" \
    "float32,bfloat16" \
    "" \
    "inner"

end

function run_sweep
# Sweeps for all thesis figures.
# Phase 1: f32 (primary data for all figures)
# Phase 2: bf16 (fallback for configs that OOM at f32)

echo "=== PHASE 1: float32 ==="
_sweep_all_figures float32

echo "=== PHASE 2: bfloat16 (fallback) ==="
_sweep_all_figures bfloat16

echo "=== SWEEP COMPLETE ==="
end

# Run sweep only if executed directly (not sourced)
# "status" outputs "sourcing" when the script is being sourced
if not status | grep -q sourcing
    run_sweep
end
