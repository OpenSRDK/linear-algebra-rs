use super::trf::PPTRF;
use crate::matrix::MatrixError;
use crate::number::c64;
use crate::SymmetricPackedMatrix;
use lapack::{dpptri, zpptri};

impl PPTRF {
    /// # Inverse
    /// with matrix decomposed by potrf
    pub fn pptri(self) -> Result<SymmetricPackedMatrix, MatrixError> {
        let PPTRF(mut mat) = self;
        let n = mat.dim();

        let mut info = 0;
        let n = n as i32;

        unsafe {
            dpptri('L' as u8, n, &mut mat.elems, &mut info);
        }

        match info {
            0 => Ok(mat),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dpptri".to_owned(),
                info,
            }),
        }
    }
}

impl PPTRF<c64> {
    /// # Inverse
    /// with matrix decomposed by potrf
    pub fn pptri(self) -> Result<SymmetricPackedMatrix<c64>, MatrixError> {
        let PPTRF::<c64>(mut mat) = self;
        let n = mat.dim();

        let mut info = 0;
        let n = n as i32;

        unsafe {
            zpptri('L' as u8, n, &mut mat.elems, &mut info);
        }

        match info {
            0 => Ok(mat),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zpptri".to_owned(),
                info,
            }),
        }
    }
}
