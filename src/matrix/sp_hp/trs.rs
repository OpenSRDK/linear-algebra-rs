use super::trf::{HPTRF, SPTRF};
use crate::number::c64;
use crate::{matrix::MatrixError, Matrix};
use lapack::{dsptrs, zhptrs, zsptrs};

impl SPTRF {
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
    pub fn sptrs(&self, b: Matrix) -> Result<Matrix, MatrixError> {
        let SPTRF(mat, ipiv) = self;
        let n = mat.dim();

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            dsptrs(
                'L' as u8,
                n,
                b.cols as i32,
                &mat.elems,
                &ipiv,
                &mut b.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dsptrs".to_owned(),
                info,
            }),
        }
    }
}

impl SPTRF<c64> {
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
    pub fn sptrs(&self, b: Matrix<c64>) -> Result<Matrix<c64>, MatrixError> {
        let SPTRF::<c64>(mat, ipiv) = self;
        let n = mat.dim();

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            zsptrs(
                'L' as u8,
                n,
                b.cols as i32,
                &mat.elems,
                &ipiv,
                &mut b.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zsptrs".to_owned(),
                info,
            }),
        }
    }
}

impl HPTRF {
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
    pub fn hptrs(&self, b: Matrix<c64>) -> Result<Matrix<c64>, MatrixError> {
        let HPTRF(mat, ipiv) = self;
        let n = mat.dim();

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            zhptrs(
                'L' as u8,
                n,
                b.cols as i32,
                &mat.elems,
                &ipiv,
                &mut b.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zhptrs".to_owned(),
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
