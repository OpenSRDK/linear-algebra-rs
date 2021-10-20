use super::trf::{HPTRF, SPTRF};
use crate::matrix::MatrixError;
use crate::number::c64;
use crate::SymmetricPackedMatrix;
use lapack::{dsptri, zhptri, zsptri};

impl SPTRF {
    /// # Inverse
    /// with matrix decomposed by sptrf
    pub fn sptri(self) -> Result<SymmetricPackedMatrix, MatrixError> {
        let SPTRF(mut mat, ipiv) = self;

        let n = mat.dim();

        let mut work = vec![f64::default(); n];
        let mut info = 0;

        let n = n as i32;

        unsafe {
            dsptri('L' as u8, n, &mut mat.elems, &ipiv, &mut work, &mut info);
        }

        match info {
            0 => Ok(mat),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dsptri".to_owned(),
                info,
            }),
        }
    }
}

impl SPTRF<c64> {
    /// # Inverse
    /// with matrix decomposed by sptrf
    pub fn sptri(self) -> Result<SymmetricPackedMatrix<c64>, MatrixError> {
        let SPTRF::<c64>(mut mat, ipiv) = self;

        let n = mat.dim();

        let mut work = vec![c64::default(); n];
        let mut info = 0;

        let n = n as i32;

        unsafe {
            zsptri('L' as u8, n, &mut mat.elems, &ipiv, &mut work, &mut info);
        }

        match info {
            0 => Ok(mat),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zsptri".to_owned(),
                info,
            }),
        }
    }
}

impl HPTRF {
    /// # Inverse
    /// with matrix decomposed by hptrf
    pub fn hptri(self) -> Result<SymmetricPackedMatrix<c64>, MatrixError> {
        let HPTRF(mut mat, ipiv) = self;

        let n = mat.dim();

        let mut work = vec![c64::default(); n];
        let mut info = 0;

        let n = n as i32;

        unsafe {
            zhptri('L' as u8, n, &mut mat.elems, &ipiv, &mut work, &mut info);
        }

        match info {
            0 => Ok(mat),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zhptri".to_owned(),
                info,
            }),
        }
    }
}
