use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use lapack::dgesvd;
use std::error::Error;

impl Matrix {
    /// # Singular Value Decomposition
    ///
    /// https://en.wikipedia.org/wiki/Singular_value_decomposition
    ///
    /// `M = U * Sigma * V^T`
    /// `(u, sigma, vt)`
    pub fn gesvd(&self) -> Result<(Matrix, Matrix, Matrix), Box<dyn Error>> {
        if self.rows != self.cols {
            return Err(Box::new(MatrixError::DimensionMismatch));
        }

        let mut info = 0;
        let mut u = Matrix::new(self.rows, self.rows);
        let mut sigma = Matrix::new(self.rows, self.cols);
        let mut vt = Matrix::new(self.cols, self.cols);
        let lwork = 2 * self.rows;

        unsafe {
            dgesvd(
                'A' as u8,
                'A' as u8,
                self.rows as i32,
                self.cols as i32,
                &mut self.t().elems,
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
            _ => Err(Box::new(MatrixError::LapackRoutineError {
                routine: "dpotrf".to_owned(),
                info,
            })),
        }
    }
}
