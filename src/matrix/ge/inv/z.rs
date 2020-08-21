use crate::{matrix::Matrix, number::c64};
use lapack::zgesv;
use std::mem::transmute;

impl Matrix<c64> {
    /// # Inverse
    /// for square matrix
    pub fn geinv(self) -> Result<Matrix<c64>, String> {
        let mut slf = self;
        let mut ipiv = vec![0; slf.rows];
        let mut solution_matrix = Matrix::<c64>::identity(slf.rows);
        let mut info = 0;

        unsafe {
            zgesv(
                slf.rows as i32,
                slf.rows as i32,
                transmute::<&mut [c64], &mut [blas::c64]>(&mut slf.elements),
                slf.rows as i32,
                &mut ipiv,
                transmute::<&mut [c64], &mut [blas::c64]>(&mut solution_matrix.elements),
                slf.rows as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok(solution_matrix),
            i => Err(i.to_string()),
        }
    }
}
