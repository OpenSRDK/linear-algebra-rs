use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dgetri, zgetri};
use std::error::Error;

impl Matrix {
    /// # Inverse
    /// with matrix decomposed by getrf
    pub fn getri(self, ipiv: &[i32]) -> Result<Matrix, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() {
            return Err(Box::new(MatrixError::DimensionMismatch));
        }

        let mut work = vec![f64::default(); n];
        let mut info = 0;

        let mut slf = self;
        let n = n as i32;

        unsafe {
            dgetri(n, &mut slf.elems, n, ipiv, &mut work, n, &mut info);
        }

        match info {
            0 => Ok(slf),
            _ => Err(Box::new(MatrixError::LapackRoutineError {
                routine: "dgetri".to_owned(),
                info,
            })),
        }
    }
}

impl Matrix<c64> {
    /// # Inverse
    /// with matrix decomposed by getrf
    pub fn getri(self, ipiv: &[i32]) -> Result<Matrix<c64>, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() {
            return Err(Box::new(MatrixError::DimensionMismatch));
        }

        let mut work = vec![c64::default(); n];
        let mut info = 0;

        let mut slf = self;
        let n = n as i32;

        unsafe {
            zgetri(n, &mut slf.elems, n, ipiv, &mut work, n, &mut info);
        }

        match info {
            0 => Ok(slf),
            _ => Err(Box::new(MatrixError::LapackRoutineError {
                routine: "zgetri".to_owned(),
                info,
            })),
        }
    }
}
