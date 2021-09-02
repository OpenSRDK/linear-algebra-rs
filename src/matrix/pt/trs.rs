use crate::bd::BidiagonalMatrix;
use crate::{matrix::*, DiagonalMatrix};
use lapack::dpttrs;

impl BidiagonalMatrix<f64> {
    /// # Solve equation
    /// with matrix decomposed by pttrf
    /// `Ax = b`
    /// return x
    pub fn pttrs(&self, d: &DiagonalMatrix, b: Matrix) -> Result<Matrix, MatrixError> {
        let e = self.e();
        let n = self.d().len() as i32;
        let mut b = b;
        let mut info = 0;

        unsafe { dpttrs(n, b.cols as i32, d.d(), &e, &mut b.elems, n, &mut info) }

        match info {
            0 => Ok(b),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dpttrs".to_owned(),
                info,
            }),
        }
    }
}
