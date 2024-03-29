use crate::matrix::ge::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use crate::Number;
use lapack::{dpotrf, zpotrf};

#[derive(Clone, Debug)]
pub struct POTRF<T = f64>(pub Matrix<T>)
where
    T: Number;

impl Matrix {
    /// # Cholesky decomposition
    /// for positive definite f64 matrix
    ///
    /// https://en.wikipedia.org/wiki/Cholesky_decomposition
    ///
    /// `A = L * L^T`
    pub fn potrf(self) -> Result<POTRF, MatrixError> {
        let n = self.rows;
        if n != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut info = 0;
        let mut slf = self;
        let n = n as i32;

        unsafe {
            dpotrf('L' as u8, n, &mut slf.elems, n, &mut info);
        }

        match info {
            0 => Ok(POTRF(slf)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dpotrf".to_owned(),
                info,
            }),
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
    pub fn potrf(self) -> Result<POTRF<c64>, MatrixError> {
        let n = self.rows;
        if n != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut info = 0;
        let mut slf = self;
        let n = n as i32;

        unsafe {
            zpotrf('L' as u8, n, &mut slf.elems, n, &mut info);
        }

        match info {
            0 => Ok(POTRF::<c64>(slf)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zpotrf".to_owned(),
                info,
            }),
        }
    }
}
