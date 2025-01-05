use cubecl::prelude::*;
use thundercube::{
    LINE_SIZE,
    streaming::{AsyncStream, GpuPtr},
    test_utils::{TestRuntime, client},
    tiles::{D64, Dim, DimOrOne},
    util::wait_for_sync,
};

// Control flag indices
const READY: usize = 0;
const DONE: usize = 1;
const SHUTDOWN: usize = 2;

/// Persistent add kernel (same as in persistent_streaming example)
#[cube(launch)]
fn persistent_add<F: Float, N: Dim>(
    a: &Tensor<Line<F>>,
    b: &Tensor<Line<F>>,
    out: &mut Tensor<Line<F>>,
    ctrl: &mut Array<u32>,
) {
    let tid = UNIT_POS as usize;
    let num_threads = CUBE_DIM as usize;
    let num_lines = N::VALUE / LINE_SIZE;

    loop {
        if UNIT_POS == 0 {
            loop {
                if ctrl[SHUTDOWN] != 0 {
                    ctrl[READY] = u32::MAX;
                    break;
                }
                if ctrl[READY] != 0 {
                    break;
                }
            }
        }

        sync_cube();

        if ctrl[READY] == u32::MAX {
            break;
        }

        for i in range_stepped(tid, num_lines, num_threads) {
            out[i] = a[i] + b[i];
        }

        sync_cube();
        if UNIT_POS == 0 {
            ctrl[READY] = 0;
            ctrl[DONE] = 1;
        }
        sync_cube();
    }
}

fn main() {
    let client = client();
    type N = D64;
    let n = N::VALUE;

    let handle_a = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let handle_b = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let handle_c = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let handle_ctrl = client.create_from_slice(u32::as_bytes(&[0u32, 0, 0]));

    let stream = AsyncStream::new();
    let ptr_a: GpuPtr<f32> = stream.ptr(&client, &handle_a);
    let ptr_b: GpuPtr<f32> = stream.ptr(&client, &handle_b);
    let ptr_c: GpuPtr<f32> = stream.ptr(&client, &handle_c);
    let ptr_ctrl: GpuPtr<u32> = stream.ptr(&client, &handle_ctrl);

    let shape = vec![n];
    let strides = vec![1usize];
    let in_a =
        unsafe { TensorArg::from_raw_parts::<Line<f32>>(&handle_a, &strides, &shape, LINE_SIZE) };
    let in_b =
        unsafe { TensorArg::from_raw_parts::<Line<f32>>(&handle_b, &strides, &shape, LINE_SIZE) };
    let output =
        unsafe { TensorArg::from_raw_parts::<Line<f32>>(&handle_c, &strides, &shape, LINE_SIZE) };
    let ctrl_arg = unsafe { ArrayArg::from_raw_parts::<u32>(&handle_ctrl, 3, 1) };

    persistent_add::launch::<f32, N, TestRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(32),
        in_a,
        in_b,
        output,
        ctrl_arg,
    )
    .expect("launch failed");

    println!("=== Fresh Data Verification Test ===\n");

    let run_iteration = |stream: &AsyncStream,
                         ptr_a: GpuPtr<f32>,
                         ptr_b: GpuPtr<f32>,
                         ptr_c: GpuPtr<f32>,
                         ptr_ctrl: GpuPtr<u32>,
                         a: &[f32],
                         b: &[f32]|
     -> Vec<f32> {
        stream.write(ptr_a, 0, a);
        stream.write(ptr_b, 0, b);
        stream.write(ptr_ctrl, 0, &[1u32, 0, 0]);

        loop {
            let flags = stream.read(ptr_ctrl, 0, 3);
            if flags[DONE] != 0 {
                break;
            }
        }

        let result = stream.read(ptr_c, 0, n);
        stream.write(ptr_ctrl, 0, &[0u32, 0, 0]);
        result
    };

    println!("Test 1: [1,1,1,...] + [2,2,2,...] = [3,3,3,...]");
    let result = run_iteration(
        &stream,
        ptr_a,
        ptr_b,
        ptr_c,
        ptr_ctrl,
        &vec![1.0f32; n],
        &vec![2.0f32; n],
    );
    assert!(
        result.iter().all(|&x| x == 3.0),
        "Expected all 3.0, got {:?}",
        &result[..5]
    );
    println!("  PASS: all elements are 3.0\n");

    println!("Test 2: [10,10,10,...] + [20,20,20,...] = [30,30,30,...]");
    let result = run_iteration(
        &stream,
        ptr_a,
        ptr_b,
        ptr_c,
        ptr_ctrl,
        &vec![10.0f32; n],
        &vec![20.0f32; n],
    );
    assert!(
        result.iter().all(|&x| x == 30.0),
        "Expected all 30.0, got {:?}",
        &result[..5]
    );
    println!("  PASS: all elements are 30.0\n");

    println!("Test 3: [100,200,300,...] + [1,2,3,...] = [101,202,303,...]");
    let a3: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 100.0).collect();
    let b3: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
    let result = run_iteration(&stream, ptr_a, ptr_b, ptr_c, ptr_ctrl, &a3, &b3);
    let expected: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 101.0).collect();
    assert_eq!(result, expected, "Mismatch in unique pattern test!");
    println!(
        "  PASS: result[0]={}, result[31]={}, result[63]={}",
        result[0], result[31], result[63]
    );

    println!("\nTest 4: [0,0,0,...] + [0,0,0,...] = [0,0,0,...]");
    let result = run_iteration(
        &stream,
        ptr_a,
        ptr_b,
        ptr_c,
        ptr_ctrl,
        &vec![0.0f32; n],
        &vec![0.0f32; n],
    );
    assert!(
        result.iter().all(|&x| x == 0.0),
        "Expected all 0.0, got {:?}",
        &result[..5]
    );
    println!("  PASS: all elements are 0.0\n");

    // Shutdown
    stream.write(ptr_ctrl, 0, &[0u32, 0, 1]);
    std::thread::sleep(std::time::Duration::from_millis(50));
    wait_for_sync(&client).unwrap();

    println!("=== All tests passed! Fresh data is correctly read each iteration. ===");
}
