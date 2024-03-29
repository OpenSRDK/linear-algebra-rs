use crate::matrix::ge::Matrix;
use crate::matrix::MatrixError;
use lapack::dgesvd;

impl Matrix {
    /// # Singular Value Decomposition
    ///
    /// https://en.wikipedia.org/wiki/Singular_value_decomposition
    ///
    /// `M = U * Sigma * V^T`
    /// `(u, sigma, vt)`
    pub fn gesvd(mut self) -> Result<(Matrix, Matrix, Matrix), MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut info = 0;
        let mut u = Matrix::new(self.rows, self.rows);
        let mut sigma = Matrix::new(self.rows, self.cols);
        let mut vt = Matrix::new(self.cols, self.cols);
        let lwork = 1usize.max(5usize * self.rows.min(self.cols));

        unsafe {
            dgesvd(
                'A' as u8,
                'A' as u8,
                self.rows as i32,
                self.cols as i32,
                &mut self.elems,
                self.rows as i32,
                &mut sigma.elems,
                &mut u.elems,
                self.rows as i32,
                &mut vt.elems,
                self.cols as i32,
                &mut vec![0.0; lwork],
                lwork as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok((u, sigma, vt)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dgesvd".to_owned(),
                info,
            }),
        }
    }
}
