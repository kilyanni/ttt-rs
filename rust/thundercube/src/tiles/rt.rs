use std::marker::PhantomData;

use cubecl::prelude::*;

use super::dim::{Dim, DimOrOne};
use crate::{binary_ops::*, prelude::*, reduction_ops::*, unary_ops::*};

#[derive(CubeType)]
pub struct Rt<F: Float, R: Dim, C: DimOrOne> {
    pub data: Array<Line<F>>,
    #[cube(comptime)]
    _phantom: PhantomData<(R, C)>,
    // This is just for ergonomics,
    // as we can't access Self::LEN due to CubeCL limitations
    #[cube(comptime)]
    len: usize,
}

pub type Rv<F, L> = Rt<F, L, D1>;

impl<F: Float, R: Dim, C: DimOrOne> Rt<F, R, C> {
    pub const ROWS: usize = R::VALUE;
    pub const COLS: usize = C::VALUE;
    pub const SIZE: usize = R::VALUE * C::VALUE;
    pub const LEN: usize = R::VALUE * C::VALUE / LINE_SIZE;

    pub fn len() -> usize {
        Self::LEN
    }

    pub fn size() -> usize {
        Self::SIZE
    }
}

/// General methods for all register tiles (including vectors).
#[cube]
impl<F: Float, R: Dim, C: DimOrOne> Rt<F, R, C> {
    pub fn new() -> Rt<F, R, C> {
        Rt::<F, R, C> {
            data: Array::lined(comptime!(Rt::<F, R, C>::size()), LINE_SIZE),
            _phantom: PhantomData,
            len: Self::LEN,
        }
    }

    /// Create an Rt from existing data array, reinterpreting the dimensions.
    /// Used for transpose where we want to change the type from Rt<F, R, C> to Rt<F, C, R>.
    pub fn from_data(data: Array<Line<F>>) -> Rt<F, R, C> {
        Rt::<F, R, C> {
            data,
            _phantom: PhantomData,
            len: Self::LEN,
        }
    }

    pub fn apply_unary_op<O: UnaryOp<F>>(&mut self, op: O) {
        #[unroll(self.len <= UNROLL_LIMIT)]
        for i in 0..self.len {
            self.data[i] = op.apply(self.data[i]);
        }
    }

    pub fn apply_binary_op<O: BinaryOp<F>>(&mut self, op: O, other: &Rt<F, R, C>) {
        #[unroll(self.len <= UNROLL_LIMIT)]
        for i in 0..self.len {
            self.data[i] = op.apply(self.data[i], other.data[i]);
        }
    }

    pub fn copy_from(&mut self, other: &Rt<F, R, C>) {
        #[unroll(self.len <= UNROLL_LIMIT)]
        for i in 0..self.len {
            self.data[i] = other.data[i];
        }
    }

    pub fn copy_from_array(&mut self, array: &Array<Line<F>>) {
        #[unroll(self.len <= UNROLL_LIMIT)]
        for i in 0..self.len {
            self.data[i] = array[i];
        }
    }

    pub fn copy_to_array(&self, array: &mut Array<Line<F>>) {
        #[unroll(self.len <= UNROLL_LIMIT)]
        for i in 0..self.len {
            array[i] = self.data[i];
        }
    }

    /// Cast to a different floating-point type.
    pub fn cast<FOut: Float>(&self) -> Rt<FOut, R, C> {
        let mut result = Rt::<FOut, R, C>::new();
        #[unroll(self.len <= UNROLL_LIMIT)]
        for i in 0..self.len {
            result.data[i] = Line::cast_from(self.data[i]);
        }
        result
    }
}

/// Matrix-specific methods (require C: Dim, i.e., C != D1).
#[cube]
impl<F: Float, R: Dim, C: Dim> Rt<F, R, C> {
    pub fn apply_row_broadcast<O: BinaryOp<F>>(&mut self, op: O, row: &Rv<F, C>) {
        #[unroll(R::VALUE <= UNROLL_LIMIT)]
        for r in 0..R::VALUE {
            #[unroll(C::LINES <= UNROLL_LIMIT)]
            for c in 0..C::LINES {
                let idx = r * comptime!(C::LINES) + c;
                self.data[idx] = op.apply(self.data[idx], row.data[c]);
            }
        }
    }

    pub fn apply_col_broadcast<O: BinaryOp<F>>(&mut self, op: O, col: &Rv<F, R>) {
        #[unroll(R::LINES <= UNROLL_LIMIT)]
        for r_line in 0..R::LINES {
            let col_gathered = col.data[r_line];

            #[unroll]
            for i in 0..LINE_SIZE {
                let row = r_line * LINE_SIZE + i;
                let broadcast = Line::<F>::empty(LINE_SIZE).fill(col_gathered[i]);

                #[unroll(C::LINES <= UNROLL_LIMIT)]
                for c in 0..C::LINES {
                    let idx = row * comptime!(C::LINES) + c;
                    self.data[idx] = op.apply(self.data[idx], broadcast);
                }
            }
        }
    }

    pub fn reduce_rows<O: ReductionOp<F>>(&self, result: &mut Rv<F, R>) {
        #[unroll(R::LINES <= UNROLL_LIMIT)]
        for r_line in 0..R::LINES {
            let mut out_line = Line::<F>::empty(LINE_SIZE);

            #[unroll]
            for i in 0..LINE_SIZE {
                let r = r_line * LINE_SIZE + i;

                let mut acc = O::identity();
                #[unroll(C::LINES <= UNROLL_LIMIT)]
                for c_line in 0..C::LINES {
                    acc = O::combine(acc, self.data[r * comptime!(C::LINES) + c_line]);
                }
                out_line[i] = O::finalize(acc);
            }
            result.data[r_line] = out_line;
        }
    }

    pub fn reduce_cols<O: ReductionOp<F>>(&self, result: &mut Rv<F, C>) {
        #[unroll(C::LINES <= UNROLL_LIMIT)]
        for c_line in 0..C::LINES {
            let mut acc = O::identity();

            #[unroll(R::VALUE <= UNROLL_LIMIT)]
            for r in 0..R::VALUE {
                acc = O::combine(acc, self.data[r * comptime!(C::LINES) + c_line]);
            }
            result.data[c_line] = acc;
        }
    }
}

impl<F: Float, R: Dim, C: DimOrOne> Default for Rt<F, R, C> {
    fn default() -> Self {
        Self::new()
    }
}
