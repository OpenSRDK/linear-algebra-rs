use crate::{sy::trd::SYTRD, Matrix, MatrixError, SymmetricTridiagonalMatrix};
use lapack::dorgtr;

impl SYTRD {
    /// Generate an orthogonal matrix Q and symmetric tridiagonal matrix T by using the result of sytrd
    pub fn orgtr(self) -> Result<(Matrix, SymmetricTridiagonalMatrix), MatrixError> {
        let SYTRD(mut mat, tau, t) = self;
        let n = mat.rows as i32;

        let lwork = 2 * mat.rows;
        let mut work = vec![0.0; lwork];
        let mut info = 0;

        unsafe {
            dorgtr(
                'L' as u8,
                n,
                &mut mat.elems,
                n,
                &tau,
                &mut work,
                lwork as i32,
                &mut info,
            );
            if info != 0 {
                return Err(MatrixError::LapackRoutineError {
                    routine: "dorgtr".to_owned(),
                    info,
                });
            }
        }

        Ok((mat, t))
    }
}
