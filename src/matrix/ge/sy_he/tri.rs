use super::trf::{HETRF, SYTRF};
use crate::number::c64;
use crate::{matrix::MatrixError, Matrix};
use lapack::{dsytri, zhetri, zsytri};

impl SYTRF {
    /// # Inverse
    /// with matrix decomposed by sytrf
    pub fn sytri(self) -> Result<Matrix, MatrixError> {
        let SYTRF(mut mat, ipiv) = self;

        let n = mat.rows;
        if n != mat.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut work = vec![f64::default(); n];
        let mut info = 0;

        let n = n as i32;

        unsafe {
            dsytri('L' as u8, n, &mut mat.elems, n, &ipiv, &mut work, &mut info);
        }

        match info {
            0 => Ok(mat),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dsytri".to_owned(),
                info,
            }),
        }
    }
}

impl SYTRF<c64> {
    /// # Inverse
    /// with matrix decomposed by sytrf
    pub fn sytri(self) -> Result<Matrix<c64>, MatrixError> {
        let SYTRF::<c64>(mut mat, ipiv) = self;

        let n = mat.rows;
        if n != mat.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut work = vec![c64::default(); n];
        let mut info = 0;

        let n = n as i32;

        unsafe {
            zsytri('L' as u8, n, &mut mat.elems, n, &ipiv, &mut work, &mut info);
        }

        match info {
            0 => Ok(mat),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zsytri".to_owned(),
                info,
            }),
        }
    }
}

impl HETRF {
    /// # Inverse
    /// with matrix decomposed by hetrf
    pub fn hetri(self) -> Result<Matrix<c64>, MatrixError> {
        let HETRF(mut mat, ipiv) = self;

        let n = mat.rows;
        if n != mat.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut work = vec![c64::default(); n];
        let mut info = 0;

        let n = n as i32;

        unsafe {
            zhetri('L' as u8, n, &mut mat.elems, n, &ipiv, &mut work, &mut info);
        }

        match info {
            0 => Ok(mat),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zhetri".to_owned(),
                info,
            }),
        }
    }
}
