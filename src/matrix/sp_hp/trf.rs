use crate::matrix::MatrixError;
use crate::number::c64;
use crate::Number;
use crate::SymmetricPackedMatrix;
use lapack::{dsptrf, zhptrf};

pub struct SPHPTRF<T = f64>(pub SymmetricPackedMatrix<T>)
where
    T: Number;

impl SymmetricPackedMatrix {
    /// # Cholesky decomposition
    /// for positive definite f64 SymmetricPackedMatrix
    ///
    /// https://en.wikipedia.org/wiki/Cholesky_decomposition
    ///
    /// `A = L * L^T`
    pub fn sptrf(self) -> Result<SPHPTRF, MatrixError> {
        let n = self.dim;

        let mut info = 0;
        let mut slf = self;
        let mut ipiv = vec![0; n];
        let n = n as i32;

        unsafe {
            dsptrf('L' as u8, n, &mut slf.elems, &mut ipiv, &mut info);
        }

        match info {
            0 => Ok(SPHPTRF(slf)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dsptrf".to_owned(),
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
    pub fn hptrf(self) -> Result<SPHPTRF<c64>, MatrixError> {
        let n = self.dim;

        let mut info = 0;
        let mut slf = self;
        let mut ipiv = vec![0; n];
        let mut n = n as i32;

        unsafe {
            zhptrf('L' as u8, n, &mut slf.elems, &mut ipiv, &mut info);
        }

        match info {
            0 => Ok(SPHPTRF::<c64>(slf)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zhptrf".to_owned(),
                info,
            }),
        }
    }
}
