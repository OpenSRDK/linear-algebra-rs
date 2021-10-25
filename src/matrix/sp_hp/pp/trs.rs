use super::trf::PPTRF;
use crate::matrix::ge::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dpptrs, zpptrs};

impl PPTRF {
    /// # Solve equation
    ///
    /// with matrix decomposed by potrf
    ///
    /// $$
    /// \mathbf{A} \mathbf{x} = \mathbf{b}
    /// $$
    ///
    /// $$
    /// \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}
    /// $$
    pub fn pptrs(&self, b: Matrix) -> Result<Matrix, MatrixError> {
        let PPTRF(mat) = self;
        let n = mat.dim();

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            dpptrs(
                'L' as u8,
                n,
                b.cols() as i32,
                &mat.elems,
                b.elems_mut(),
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dpptrs".to_owned(),
                info,
            }),
        }
    }
}

impl PPTRF<c64> {
    /// # Solve equation
    ///
    /// with matrix decomposed by potrf
    ///
    /// $$
    /// \mathbf{A} \mathbf{x} = \mathbf{b}
    /// $$
    ///
    /// $$
    /// \mathbf{x} = \mathbf{A}^{-1} \mathbf{b}
    /// $$
    pub fn pptrs(&self, b: Matrix<c64>) -> Result<Matrix<c64>, MatrixError> {
        let PPTRF::<c64>(mat) = self;
        let n = mat.dim();

        let mut info = 0;

        let n = n as i32;
        let mut b = b;

        unsafe {
            zpptrs(
                'L' as u8,
                n,
                b.cols() as i32,
                &mat.elems,
                b.elems_mut(),
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zpptrs".to_owned(),
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
        let l = c.pptrf().unwrap();
        let x_t = l.pptrs(b).unwrap();

        println!("{:#?}", x_t);
        // assert_eq!(x_t[0][0], 0.0);
        // assert_eq!(x_t[0][1], 1.0);
        // assert_eq!(x_t[1][0], 5.0 / 3.0 - 1.0);
        // assert_eq!(x_t[1][1], 5.0 / 3.0);
    }
}
