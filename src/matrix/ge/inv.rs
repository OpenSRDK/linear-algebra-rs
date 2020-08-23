use crate::{matrix::Matrix, number::c64};
use lapack::dgesv;
use lapack::zgesv;

impl Matrix {
    /// # Inverse
    /// for square matrix
    pub fn geinv(self) -> Result<Matrix, String> {
        let mut slf = self;
        let mut ipiv = vec![0; slf.rows];
        let mut solution_matrix = Matrix::identity(slf.rows);
        let mut info = 0;

        unsafe {
            dgesv(
                slf.rows as i32,
                slf.rows as i32,
                &mut slf.elems,
                slf.rows as i32,
                &mut ipiv,
                &mut solution_matrix.elems,
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
                &mut slf.elems,
                slf.rows as i32,
                &mut ipiv,
                &mut solution_matrix.elems,
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
