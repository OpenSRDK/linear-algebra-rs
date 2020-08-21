use crate::matrix::Matrix;
use lapack::dgesv;

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
                &mut slf.elements,
                slf.rows as i32,
                &mut ipiv,
                &mut solution_matrix.elements,
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
