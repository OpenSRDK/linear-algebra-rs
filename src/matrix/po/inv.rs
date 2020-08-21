use crate::matrix::Matrix;
use lapack::dposv;

impl Matrix {
    /// # Inverse
    /// for positive definite matrix
    pub fn poinv(self) -> Result<Matrix, String> {
        if self.rows != self.columns {
            return Err("dimension mismatch".to_owned());
        }

        let mut slf = self;
        let mut solution_matrix = Matrix::identity(slf.rows);
        let mut info = 0;

        unsafe {
            dposv(
                'U' as u8,
                slf.rows as i32,
                slf.rows as i32,
                &mut slf.elements,
                slf.rows as i32,
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
