use crate::matrix::Matrix;
use crate::number::c64;
use lapack::{dpotrf, zpotrf};

impl Matrix {
    /// # Cholesky decomposition
    /// for positive definite f64 matrix
    ///
    /// https://en.wikipedia.org/wiki/Cholesky_decomposition
    ///
    /// `A = L * L^T`
    pub fn potrf(self) -> Result<Matrix, String> {
        let n = self.rows;
        if n != self.columns {
            return Err("dimension mismatch".to_owned());
        }

        let mut info = 0;
        let mut slf = self;
        let n = n as i32;

        unsafe {
            dpotrf('U' as u8, n, &mut slf.elements, n, &mut info);
        }

        match info {
            0 => Ok(slf),
            i => Err(i.to_string()),
        }
    }
}

impl Matrix<c64> {
    /// # Cholesky decomposition
    /// for positive definite c64 matrix
    ///
    /// https://en.wikipedia.org/wiki/Cholesky_decomposition
    ///
    /// `A = L * L^*`
    pub fn potrf(self) -> Result<Matrix<c64>, String> {
        let n = self.rows;
        if n != self.columns {
            return Err("dimension mismatch".to_owned());
        }

        let mut info = 0;
        let mut slf = self;
        let n = n as i32;

        unsafe {
            zpotrf('U' as u8, n, &mut slf.elements, n, &mut info);
        }

        match info {
            0 => Ok(slf),
            i => Err(i.to_string()),
        }
    }
}
