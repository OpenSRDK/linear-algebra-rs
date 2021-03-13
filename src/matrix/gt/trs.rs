use crate::{matrix::*, BidiagonalMatrix};
use lapack::dgttrs;
use std::error::Error;

impl BidiagonalMatrix<f64> {
  /// # Solve equation
  /// with matrix decomposed by gttrf
  /// `Ax = b`
  /// return x
  pub fn gttrs(
    &self,
    u: &[Vec<f64>; 3],
    ipiv: &[i32],
    b: Matrix,
  ) -> Result<Matrix, Box<dyn Error>> {
    let e = self.e();
    let n = self.d().len() as i32;
    let mut b = b;
    let mut info = 0;

    unsafe {
      dgttrs(
        'N' as u8,
        n,
        b.cols as i32,
        &e,
        &u[0],
        &u[1],
        &u[2],
        ipiv,
        &mut b.elems,
        n,
        &mut info,
      )
    }

    match info {
      0 => Ok(b),
      _ => Err(
        MatrixError::LapackRoutineError {
          routine: "dgttrs".to_owned(),
          info,
        }
        .into(),
      ),
    }
  }
}
