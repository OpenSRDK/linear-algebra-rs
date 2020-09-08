use crate::bd::BidiagonalMatrix;
use crate::matrix::*;
use lapack::dpttrs;
use std::error::Error;

impl BidiagonalMatrix<f64> {
    /// # Solve equation
    /// with matrix decomposed by pttrf
    /// `Ax = b`
    /// return xt
    pub fn pttrs(&self, d: &[f64], bt: Matrix) -> Result<Matrix, Box<dyn Error>> {
        let e = self.e();
        let n = self.d().len() as i32;
        let mut bt = bt;
        let mut info = 0;

        unsafe { dpttrs(n, bt.rows as i32, &d, &e, &mut bt.elems, n, &mut info) }

        match info {
            0 => Ok(bt),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dpttrs".to_owned(),
                info,
            }
            .into()),
        }
    }
}
