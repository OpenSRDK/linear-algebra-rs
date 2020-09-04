use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dpotrs, zpotrs};
use std::error::Error;

impl Matrix {
    /// # Solve equation
    /// with matrix decomposed by potrf
    /// `Ax = b`
    pub fn potrs(&self, b_t: Matrix) -> Result<Matrix, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() || n != b_t.cols {
            return Err(Box::new(MatrixError::DimensionMismatch));
        }

        let mut info = 0;

        let n = n as i32;
        let mut b_t = b_t;

        unsafe {
            dpotrs(
                'U' as u8,
                n,
                b_t.rows as i32,
                &self.elems,
                n,
                &mut b_t.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b_t),
            _ => Err(Box::new(MatrixError::LapackRoutineError {
                routine: "dpotrs".to_owned(),
                info,
            })),
        }
    }
}

impl Matrix<c64> {
    /// # Solve equation
    /// with matrix decomposed by potrf
    /// `Ax = b`
    pub fn potrs(&self, b_t: Matrix<c64>) -> Result<Matrix<c64>, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() || n != b_t.cols {
            return Err(Box::new(MatrixError::DimensionMismatch));
        }

        let mut info = 0;

        let n = n as i32;
        let mut b_t = b_t;

        unsafe {
            zpotrs(
                'U' as u8,
                n,
                b_t.rows as i32,
                &self.elems,
                n,
                &mut b_t.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b_t),
            _ => Err(Box::new(MatrixError::LapackRoutineError {
                routine: "zpotrs".to_owned(),
                info,
            })),
        }
    }
}
