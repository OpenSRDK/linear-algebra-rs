use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dgetrs, zgetrs};
use std::error::Error;

impl Matrix {
    /// # Solve equation
    /// with matrix decomposed by getrf
    /// `Ax = b`
    pub fn getrs(&self, ipiv: &[i32], b_t: Matrix) -> Result<Matrix, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() || n != b_t.cols {
            return Err(Box::new(MatrixError::DimensionMismatch));
        }

        let mut info = 0;

        let n = n as i32;
        let mut b_t = b_t;

        unsafe {
            dgetrs(
                'T' as u8,
                n,
                b_t.rows as i32,
                &self.elems,
                n,
                ipiv,
                &mut b_t.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b_t),
            _ => Err(Box::new(MatrixError::LapackRoutineError {
                routine: "dpotrf".to_owned(),
                info,
            })),
        }
    }
}

impl Matrix<c64> {
    /// # Solve equation
    /// with matrix decomposed by getrf
    /// `Ax = b`
    pub fn getrs(&self, ipiv: &[i32], b_t: Matrix<c64>) -> Result<Matrix<c64>, Box<dyn Error>> {
        let n = self.rows();
        if n != self.cols() || n != b_t.cols {
            return Err(Box::new(MatrixError::DimensionMismatch));
        }

        let mut info = 0;

        let n = n as i32;
        let mut b_t = b_t;

        unsafe {
            zgetrs(
                'T' as u8,
                n,
                b_t.rows as i32,
                &self.elems,
                n,
                ipiv,
                &mut b_t.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b_t),
            _ => Err(Box::new(MatrixError::LapackRoutineError {
                routine: "dpotrf".to_owned(),
                info,
            })),
        }
    }
}
