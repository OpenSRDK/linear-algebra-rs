use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dgetrs, zgetrs};
use std::error::Error;

impl Matrix {
  /// # Solve equation
  /// with matrix decomposed by getrf
  /// `Ax = b`
  /// return x
  pub fn getrs(&self, ipiv: &[i32], b: Matrix) -> Result<Matrix, Box<dyn Error>> {
    let n = self.rows();
    if n != self.cols() || n != b.rows {
      return Err(MatrixError::DimensionMismatch.into());
    }

    let mut info = 0;

    let n = n as i32;
    let mut b = b;

    unsafe {
      dgetrs(
        'N' as u8,
        n,
        b.cols as i32,
        &self.elems,
        n,
        ipiv,
        &mut b.elems,
        n,
        &mut info,
      );
    }

    match info {
      0 => Ok(b),
      _ => Err(
        MatrixError::LapackRoutineError {
          routine: "dgetrs".to_owned(),
          info,
        }
        .into(),
      ),
    }
  }
}

impl Matrix<c64> {
  /// # Solve equation
  /// with matrix decomposed by getrf
  /// `Ax = b`
  /// return xt
  pub fn getrs(&self, ipiv: &[i32], bt: Matrix<c64>) -> Result<Matrix<c64>, Box<dyn Error>> {
    let n = self.rows();
    if n != self.cols() || n != bt.cols {
      return Err(MatrixError::DimensionMismatch.into());
    }

    let mut info = 0;

    let n = n as i32;
    let mut bt = bt;

    unsafe {
      zgetrs(
        'T' as u8,
        n,
        bt.rows as i32,
        &self.elems,
        n,
        ipiv,
        &mut bt.elems,
        n,
        &mut info,
      );
    }

    match info {
      0 => Ok(bt),
      _ => Err(
        MatrixError::LapackRoutineError {
          routine: "zgetrs".to_owned(),
          info,
        }
        .into(),
      ),
    }
  }
}
