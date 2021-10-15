use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use crate::Number;
use lapack::{dgetrf, zgetrf};

pub struct GETRF<T = f64>(pub Matrix<T>, pub Vec<i32>)
where
    T: Number;

impl Matrix {
    /// # LU decomposition
    /// for f64
    pub fn getrf(self) -> Result<GETRF, MatrixError> {
        let m = self.rows;
        let n = self.cols;
        let mut ipiv = vec![0; m.min(n)];
        let mut info = 0;

        let mut slf = self;
        let m = m as i32;
        let n = n as i32;

        unsafe {
            dgetrf(n, m, &mut slf.elems, n, &mut ipiv, &mut info);
        }

        match info {
            0 => Ok(GETRF(slf, ipiv)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dgetrf".to_owned(),
                info,
            }),
        }
    }
}

impl Matrix<c64> {
    /// # LU decomposition
    /// for c64
    pub fn getrf(self) -> Result<GETRF<c64>, MatrixError> {
        let m = self.rows;
        let n = self.cols;
        let mut ipiv = vec![0; m.min(n)];
        let mut info = 0;

        let mut slf = self;
        let m = m as i32;
        let n = n as i32;

        unsafe {
            zgetrf(n, m, &mut slf.elems, n, &mut ipiv, &mut info);
        }

        match info {
            0 => Ok(GETRF::<c64>(slf, ipiv)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zgetrf".to_owned(),
                info,
            }),
        }
    }
}
