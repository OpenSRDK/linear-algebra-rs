use super::trf::POTRF;
use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dpotri, zpotri};

impl POTRF {
    /// # Inverse
    /// with matrix decomposed by potrf
    pub fn potri(self) -> Result<Matrix, MatrixError> {
        let POTRF(mut mat) = self;
        let n = mat.rows();
        if n != mat.cols() {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut info = 0;
        let n = n as i32;

        unsafe {
            dpotri('L' as u8, n, &mut mat.elems, n, &mut info);
        }

        match info {
            0 => Ok(mat),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dpotri".to_owned(),
                info,
            }),
        }
    }
}

impl POTRF<c64> {
    /// # Inverse
    /// with matrix decomposed by potrf
    pub fn potri(self) -> Result<Matrix<c64>, MatrixError> {
        let POTRF::<c64>(mut mat) = self;
        let n = mat.rows();
        if n != mat.cols() {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut info = 0;
        let n = n as i32;

        unsafe {
            zpotri('L' as u8, n, &mut mat.elems, n, &mut info);
        }

        match info {
            0 => Ok(mat),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zpotri".to_owned(),
                info,
            }),
        }
    }
}
