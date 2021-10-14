use super::trf::PTTRF;
use crate::matrix::*;
use lapack::dpttrs;
use lapack::zpttrs;

impl PTTRF {
    /// # Solve equation
    /// with matrix decomposed by pttrf
    /// `Ax = b`
    /// return x
    pub fn pttrs(&self, b: Matrix) -> Result<Matrix, MatrixError> {
        let e = self.0.e();
        let n = self.0.d().len() as i32;
        let mut b = b;
        let mut info = 0;

        unsafe { dpttrs(n, b.cols as i32, self.1.d(), &e, &mut b.elems, n, &mut info) }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dpttrs".to_owned(),
                info,
            }),
        }
    }
}

impl PTTRF<c64> {
    /// # Solve equation
    /// with matrix decomposed by pttrf
    /// `Ax = b`
    /// return x
    pub fn pttrs(&self, b: Matrix<c64>) -> Result<Matrix<c64>, MatrixError> {
        let e = self.0.e();
        let n = self.0.d().len() as i32;
        let mut b = b;
        let mut info = 0;

        unsafe {
            zpttrs(
                'L' as u8,
                n,
                b.cols as i32,
                self.1.d(),
                &e,
                &mut b.elems,
                n,
                &mut info,
            )
        }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "zpttrs".to_owned(),
                info,
            }),
        }
    }
}
