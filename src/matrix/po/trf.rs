use crate::matrix::Matrix;
use lapack::{dpotrf, dpotri};

impl Matrix {
    /// # Cholesky decomposition
    /// for positive definite matrix
    ///
    /// https://en.wikipedia.org/wiki/Cholesky_decomposition
    ///
    /// `A = L * L^T`
    pub fn potrf(self) -> Result<Matrix, String> {
        if self.rows != self.columns {
            return Err("dimension mismatch".to_owned());
        }

        let n = self.rows as i32;
        let mut slf = self;
        let mut info = 0;

        unsafe {
            dpotrf('U' as u8, n, &mut slf.elements, n, &mut info);
        }

        match info {
            0 => Ok(slf),
            i => Err(i.to_string()),
        }
    }

    /// # Solve equation
    /// with matrix decomposed by potrf
    pub fn potri(self) -> Result<Matrix, String> {
        if self.rows != self.columns {
            return Err("dimension mismatch".to_owned());
        }

        let n = self.rows as i32;
        let mut slf = self;
        let mut info = 0;

        unsafe {
            dpotri('U' as u8, n, &mut slf.elements, n, &mut info);
        }

        match info {
            0 => Ok(slf),
            i => Err(i.to_string()),
        }
    }
}
