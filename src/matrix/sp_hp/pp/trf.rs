use crate::matrix::MatrixError;
use crate::number::c64;
use crate::Number;
use crate::SymmetricPackedMatrix;
use lapack::{dpptrf, zpptrf};

#[derive(Clone, Debug, Default, PartialEq, Hash)]
pub struct PPTRF<T = f64>(pub SymmetricPackedMatrix<T>)
where
    T: Number;

impl SymmetricPackedMatrix {
    /// # Cholesky decomposition
    /// for positive definite f64 matrix
    ///
    /// https://en.wikipedia.org/wiki/Cholesky_decomposition
    ///
    /// `A = L * L^T`
    pub fn pptrf(self) -> Result<PPTRF, MatrixError> {
        let n = self.dim;

        let mut info = 0;
        let mut slf = self;
        let n = n as i32;

        unsafe {
            dpptrf('L' as u8, n, &mut slf.elems, &mut info);
        }

        match info {
            0 => Ok(PPTRF(slf)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dpptrf".to_owned(),
                info,
            }),
        }
    }
}

impl SymmetricPackedMatrix<c64> {
    /// # Cholesky decomposition
    /// for positive definite c64 matrix
    ///
    /// https://en.wikipedia.org/wiki/Cholesky_decomposition
    ///
    /// `A = L * L^*`
    pub fn pptrf(self) -> Result<PPTRF<c64>, MatrixError> {
        let n = self.dim;

        let mut info = 0;
        let mut slf = self;
        let n = n as i32;

        unsafe {
            zpptrf('L' as u8, n, &mut slf.elems, &mut info);
        }

        match info {
            0 => Ok(PPTRF::<c64>(slf)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zpptrf".to_owned(),
                info,
            }),
        }
    }
}
