use crate::{
    matrix::{operations::identity::identity, Matrix},
    number::c64,
    types::{Square, Standard},
};
use lapack::zgesv;
use std::intrinsics::transmute;

impl Matrix<Square, c64> {
    /// # Inverse
    /// for Square Matrix
    pub fn inv(mut self) -> Result<Matrix<Standard, c64>, String> {
        let mut solution_matrix = identity::<c64>(self.rows).transmute::<Standard>();

        let mut ipiv = vec![0; self.rows];
        let mut info = 0;

        unsafe {
            zgesv(
                self.rows as i32,
                self.rows as i32,
                transmute::<&mut [c64], &mut [blas::c64]>(&mut self.elements),
                self.rows as i32,
                &mut ipiv,
                transmute::<&mut [c64], &mut [blas::c64]>(&mut solution_matrix.elements),
                self.rows as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok(solution_matrix),
            i => Err(i.to_string()),
        }
    }
}
