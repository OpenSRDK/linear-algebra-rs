use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dpotri, zpotri};
use std::error::Error;

impl Matrix {
  /// # Inverse
  /// with matrix decomposed by potrf
  pub fn potri(self) -> Result<Matrix, Box<dyn Error>> {
    let n = self.rows();
    if n != self.cols() {
      return Err(MatrixError::DimensionMismatch.into());
    }

    let mut info = 0;
    let mut slf = self;
    let n = n as i32;

    unsafe {
      dpotri('L' as u8, n, &mut slf.elems, n, &mut info);
    }

    match info {
      0 => Ok(slf),
      _ => Err(
        MatrixError::LapackRoutineError {
          routine: "dpotri".to_owned(),
          info,
        }
        .into(),
      ),
    }
  }
}

impl Matrix<c64> {
  /// # Inverse
  /// with matrix decomposed by potrf
  pub fn potri(self) -> Result<Matrix<c64>, Box<dyn Error>> {
    let n = self.rows();
    if n != self.cols() {
      return Err(MatrixError::DimensionMismatch.into());
    }

    let mut info = 0;
    let mut slf = self;
    let n = n as i32;

    unsafe {
      zpotri('L' as u8, n, &mut slf.elems, n, &mut info);
    }

    match info {
      0 => Ok(slf),
      _ => Err(
        MatrixError::LapackRoutineError {
          routine: "zpotri".to_owned(),
          info,
        }
        .into(),
      ),
    }
  }
}
