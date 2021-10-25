use crate::matrix::MatrixError;
use crate::number::c64;
use crate::Number;
use crate::SymmetricPackedMatrix;
use lapack::{dsptrf, zhptrf, zsptrf};

#[derive(Clone, Debug)]
pub struct SPTRF<T = f64>(pub SymmetricPackedMatrix<T>, pub Vec<i32>)
where
    T: Number;

#[derive(Clone, Debug)]
pub struct HPTRF(pub SymmetricPackedMatrix<c64>, pub Vec<i32>);

impl SymmetricPackedMatrix {
    ///
    pub fn sptrf(self) -> Result<SPTRF, MatrixError> {
        let n = self.dim;

        let mut info = 0;
        let mut slf = self;
        let mut ipiv = vec![0; n];
        let n = n as i32;

        unsafe {
            dsptrf('L' as u8, n, &mut slf.elems, &mut ipiv, &mut info);
        }

        match info {
            0 => Ok(SPTRF(slf, ipiv)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dsptrf".to_owned(),
                info,
            }),
        }
    }
}

impl SymmetricPackedMatrix<c64> {
    ///
    pub fn sptrf(self) -> Result<SPTRF<c64>, MatrixError> {
        let n = self.dim;

        let mut info = 0;
        let mut slf = self;
        let mut ipiv = vec![0; n];
        let n = n as i32;

        unsafe {
            zsptrf('L' as u8, n, &mut slf.elems, &mut ipiv, &mut info);
        }

        match info {
            0 => Ok(SPTRF::<c64>(slf, ipiv)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zsptrf".to_owned(),
                info,
            }),
        }
    }

    ///
    pub fn hptrf(self) -> Result<HPTRF, MatrixError> {
        let n = self.dim;

        let mut info = 0;
        let mut slf = self;
        let mut ipiv = vec![0; n];
        let n = n as i32;

        unsafe {
            zhptrf('L' as u8, n, &mut slf.elems, &mut ipiv, &mut info);
        }

        match info {
            0 => Ok(HPTRF(slf, ipiv)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zhptrf".to_owned(),
                info,
            }),
        }
    }
}
