use crate::bd::BidiagonalMatrix;
use crate::matrix::*;
use lapack::dgttrs;
use std::error::Error;

impl BidiagonalMatrix<f64> {
    /// # Solve equation
    /// with matrix decomposed by gttrf
    /// `Ax = b`
    /// return xt
    pub fn gttrs(
        &self,
        u: &[Vec<f64>; 3],
        ipiv: &[i32],
        bt: Matrix,
    ) -> Result<Matrix, Box<dyn Error>> {
        let e = self.e();
        let n = self.d().len() as i32;
        let mut bt = bt;
        let mut info = 0;

        unsafe {
            dgttrs(
                'N' as u8,
                n,
                bt.rows as i32,
                &e,
                &u[0],
                &u[1],
                &u[2],
                ipiv,
                &mut bt.elems,
                n,
                &mut info,
            )
        }

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
