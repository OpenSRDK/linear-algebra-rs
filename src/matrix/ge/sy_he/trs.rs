use super::trf::{HETRF, SYTRF};
use crate::number::c64;
use crate::{matrix::MatrixError, ge::Matrix};
use lapack::{dsytrs, zhetrs, zsytrs};

impl SYTRF {
    /// # Solve equation
    ///
    /// with matrix decomposed by sptrf
    ///
    /// $$
    /// \mathbf{A} \mathbf{x} = \mathbf{b}
    /// $$
    ///
    /// $$
    /// \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}
    /// $$
    pub fn sytrs(&self, b: Matrix) -> Result<Matrix, MatrixError> {
        let SYTRF(mat, ipiv) = self;
        let n = mat.rows;
        if n != mat.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            dsytrs(
                'L' as u8,
                n,
                b.cols as i32,
                &mat.elems,
                n,
                &ipiv,
                &mut b.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dsytrs".to_owned(),
                info,
            }),
        }
    }
}

impl SYTRF<c64> {
    /// # Solve equation
    ///
    /// with matrix decomposed by sptrf
    ///
    /// $$
    /// \mathbf{A} \mathbf{x} = \mathbf{b}
    /// $$
    ///
    /// $$
    /// \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}
    /// $$
    pub fn sytrs(&self, b: Matrix<c64>) -> Result<Matrix<c64>, MatrixError> {
        let SYTRF::<c64>(mat, ipiv) = self;
        let n = mat.rows;
        if n != mat.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            zsytrs(
                'L' as u8,
                n,
                b.cols as i32,
                &mat.elems,
                n,
                &ipiv,
                &mut b.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zsytrs".to_owned(),
                info,
            }),
        }
    }
}

impl HETRF {
    /// # Solve equation
    ///
    /// with matrix decomposed by hptrf
    ///
    /// $$
    /// \mathbf{A} \mathbf{x} = \mathbf{b}
    /// $$
    ///
    /// $$
    /// \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}
    /// $$
    pub fn hetrs(&self, b: Matrix<c64>) -> Result<Matrix<c64>, MatrixError> {
        let HETRF(mat, ipiv) = self;
        let n = mat.rows;
        if n != mat.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            zhetrs(
                'L' as u8,
                n,
                b.cols as i32,
                &mat.elems,
                n,
                &ipiv,
                &mut b.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zhetrs".to_owned(),
                info,
            }),
        }
    }
}

#[cfg(test)]

mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = vec![2.0, 1.0, 2.0];
        let c = SymmetricPackedMatrix::from(2, a).unwrap();
        let b = mat![
            1.0, 3.0;
            2.0, 4.0
        ];
        let l = c.sptrf().unwrap();
        let x_t = l.sptrs(b).unwrap();

        println!("{:#?}", x_t);
        // assert_eq!(x_t[0][0], 0.0);
        // assert_eq!(x_t[0][1], 1.0);
        // assert_eq!(x_t[1][0], 5.0 / 3.0 - 1.0);
        // assert_eq!(x_t[1][1], 5.0 / 3.0);
    }
}
