use crate::matrix::ge::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use crate::Number;
use lapack::{dsytrf, zhetrf, zsytrf};

#[derive(Clone, Debug)]
pub struct SYTRF<T = f64>(pub Matrix<T>, pub Vec<i32>)
where
    T: Number;

pub struct HETRF(pub Matrix<c64>, pub Vec<i32>);

impl Matrix {
    ///
    pub fn sytrf(self) -> Result<SYTRF, MatrixError> {
        let n = self.rows;
        if n != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut mat = self;
        let mut ipiv = vec![0; n];
        let lwork = 2 * mat.rows;
        let mut work = vec![0.0; lwork];
        let mut info = 0;
        let n = n as i32;

        unsafe {
            dsytrf(
                'L' as u8,
                n,
                &mut mat.elems,
                n,
                &mut ipiv,
                &mut work,
                lwork as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok(SYTRF(mat, ipiv)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dsytrf".to_owned(),
                info,
            }),
        }
    }
}

impl Matrix<c64> {
    ///
    pub fn sytrf(self) -> Result<SYTRF<c64>, MatrixError> {
        let n = self.rows;
        if n != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut mat = self;
        let mut ipiv = vec![0; n];
        let lwork = 2 * mat.rows;
        let mut work = vec![c64::default(); lwork];
        let mut info = 0;
        let n = n as i32;

        unsafe {
            zsytrf(
                'L' as u8,
                n,
                &mut mat.elems,
                n,
                &mut ipiv,
                &mut work,
                lwork as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok(SYTRF::<c64>(mat, ipiv)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zsytrf".to_owned(),
                info,
            }),
        }
    }

    ///
    pub fn hetrf(self) -> Result<HETRF, MatrixError> {
        let n = self.rows;
        if n != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut mat = self;
        let mut ipiv = vec![0; n];
        let lwork = 2 * mat.rows;
        let mut work = vec![c64::default(); lwork];
        let mut info = 0;
        let n = n as i32;

        unsafe {
            zhetrf(
                'L' as u8,
                n,
                &mut mat.elems,
                n,
                &mut ipiv,
                &mut work,
                lwork as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok(HETRF(mat, ipiv)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zhetrf".to_owned(),
                info,
            }),
        }
    }
}
