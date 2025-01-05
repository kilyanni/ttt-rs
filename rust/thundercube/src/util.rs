use cubecl::{prelude::*, std::ReinterpretSliceMut};

use crate::LINE_SIZE;

/// Synchronizes at least within a plane.
/// Uses `sync_plane()` on CUDA, falls back to `sync_cube()` on other backends.
#[cfg(feature = "cuda")]
#[cube]
pub fn sync_planes() {
    sync_plane();
}

/// Synchronizes at least within a plane.
/// Uses `sync_plane()` on CUDA, falls back to `sync_cube()` on other backends.
#[cfg(not(feature = "cuda"))]
#[cube]
pub fn sync_planes() {
    sync_cube();
}

// This is ugly, but it works
// CubeCL doesn't like [Line<F>; LINE_SIZE]
// so we have to do this monstrosity
#[cube]
pub fn transpose_4<F: Float>(
    r0: Line<F>,
    r1: Line<F>,
    r2: Line<F>,
    r3: Line<F>,
) -> (Line<F>, Line<F>, Line<F>, Line<F>) {
    let mut c0 = Line::empty(4usize);
    let mut c1 = Line::empty(4usize);
    let mut c2 = Line::empty(4usize);
    let mut c3 = Line::empty(4usize);

    c0[0] = r0[0];
    c0[1] = r1[0];
    c0[2] = r2[0];
    c0[3] = r3[0];

    c1[0] = r0[1];
    c1[1] = r1[1];
    c1[2] = r2[1];
    c1[3] = r3[1];

    c2[0] = r0[2];
    c2[1] = r1[2];
    c2[2] = r2[2];
    c2[3] = r3[2];

    c3[0] = r0[3];
    c3[1] = r1[3];
    c3[2] = r2[3];
    c3[3] = r3[3];

    (c0, c1, c2, c3)
}

/// Square block of lines.
/// 4x4
#[derive(CubeType, Clone, Copy)]
pub struct LineBlock<F: Float> {
    pub l0: Line<F>,
    pub l1: Line<F>,
    pub l2: Line<F>,
    pub l3: Line<F>,
}

#[cube]
impl<F: Float> LineBlock<F> {
    pub fn empty() -> Self {
        LineBlock::<F> {
            l0: Line::empty(LINE_SIZE),
            l1: Line::empty(LINE_SIZE),
            l2: Line::empty(LINE_SIZE),
            l3: Line::empty(LINE_SIZE),
        }
    }

    pub fn load_new<S: SliceVisibility>(
        data: Slice<Line<F>, S>,
        base_row: usize,
        col_line: usize,
        stride: usize,
    ) -> Self {
        let mut block = Self::empty();
        block.load(data, base_row, col_line, stride);
        block
    }

    pub fn load<S: SliceVisibility>(
        &mut self,
        data: Slice<Line<F>, S>,
        base_row: usize,
        col_line: usize,
        stride: usize,
    ) {
        self.l0 = data[base_row * stride + col_line];
        self.l1 = data[(base_row + 1) * stride + col_line];
        self.l2 = data[(base_row + 2) * stride + col_line];
        self.l3 = data[(base_row + 3) * stride + col_line];
    }

    pub fn store(
        self,
        mut data: SliceMut<Line<F>>,
        base_row: usize,
        col_line: usize,
        stride: usize,
    ) {
        data[base_row * stride + col_line] = self.l0;
        data[(base_row + 1) * stride + col_line] = self.l1;
        data[(base_row + 2) * stride + col_line] = self.l2;
        data[(base_row + 3) * stride + col_line] = self.l3;
    }

    #[must_use]
    pub fn transpose(self) -> Self {
        let (t0, t1, t2, t3) = transpose_4(self.l0, self.l1, self.l2, self.l3);
        LineBlock::<F> {
            l0: t0,
            l1: t1,
            l2: t2,
            l3: t3,
        }
    }

    pub fn transpose_mut(&mut self) {
        let (t0, t1, t2, t3) = transpose_4(self.l0, self.l1, self.l2, self.l3);
        self.l0 = t0;
        self.l1 = t1;
        self.l2 = t2;
        self.l3 = t3;
    }
}

// CubeCL doesn't let us do lined_array[2][1] = 3
#[cube]
pub fn write_into_line<F: Float>(one_slice: SliceMut<Line<F>>, idx: usize, val: F) {
    ReinterpretSliceMut::<F, F>::new(one_slice, LINE_SIZE).write(idx, val);
    // let mut l = one_slice[0];
    // l[idx] = val;
    // one_slice[0] = l;
}

#[cube]
pub fn index_1d<T: CubePrimitive>(t: &Tensor<T>, index: usize) -> usize {
    t.stride(0) * index
}

#[cube]
pub fn slice_1d<T: CubePrimitive>(t: &Tensor<T>, index: usize) -> Slice<T> {
    let start = index_1d(t, index);
    let end = index_1d(t, index + 1) - 1;
    t.slice(start, end)
}

#[cube]
pub fn index_2d<T: CubePrimitive>(t: &Tensor<T>, x: usize, y: usize) -> usize {
    t.stride(0) * x + t.stride(1) * y
}

#[cube]
pub fn slice_2d<T: CubePrimitive>(t: &Tensor<T>, x: usize, y: usize) -> Slice<T> {
    let start = index_2d(t, x, y);
    let end = index_2d(t, x + 1, y + 1) - 1;
    t.slice(start, end)
}

#[cube]
pub fn index_3d<T: CubePrimitive>(t: &Tensor<T>, x: usize, y: usize, z: usize) -> usize {
    t.stride(0) * x + t.stride(1) * y + t.stride(2) * z
}

#[cube]
pub fn slice_3d<T: CubePrimitive>(t: &Tensor<T>, x: usize, y: usize, z: usize) -> Slice<T> {
    let start = index_3d(t, x, y, z);
    let end = index_3d(t, x + 1, y + 1, z + 1) - 1;
    t.slice(start, end)
}

/// Sleep for approximately `n * 63` clock cycles. (AMD only)
///
/// Uses the AMD `s_sleep` instruction which puts the wavefront to sleep.
///
/// # Arguments
///
/// * `n` - Sleep duration multiplier (comptime). Actual sleep is ~n*63 clock cycles.
#[cfg(feature = "rocm")]
#[cube]
#[allow(unused_variables, reason = "False positive from the cube macro")]
pub fn gpu_sleep(#[comptime] n: u32) {
    // This is a truly horrible hack to emit `__builtin_amdgcn_s_sleep(n)`
    // since CubeCL doesn't have native intrinsic support.
    // The comment impl automatically uses /* */ when there's a newline,
    // so we trigger that off purpose, then escape out from the comment
    // by closing it ourselves.
    use cubecl::intrinsic;
    intrinsic!(|scope| {
        scope.register(cubecl::ir::NonSemantic::Comment {
            content: format!("*/\n__builtin_amdgcn_s_sleep({});\n/*", n),
        });
    });
}

pub fn wait_for_sync<R: Runtime>(
    client: &ComputeClient<R>,
) -> Result<(), cubecl::server::ExecutionError> {
    pollster::block_on(client.sync())
}

/// Casts a Line<FIn> to Line<FOut> element-wise.
#[cube]
pub fn cast_line<FIn: Float, FOut: Float>(input: Line<FIn>) -> Line<FOut> {
    let mut result = Line::<FOut>::empty(LINE_SIZE);
    result[0] = FOut::cast_from(input[0]);
    result[1] = FOut::cast_from(input[1]);
    result[2] = FOut::cast_from(input[2]);
    result[3] = FOut::cast_from(input[3]);
    result
}
