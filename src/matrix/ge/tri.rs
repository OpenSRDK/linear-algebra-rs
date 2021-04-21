use crate::matrix::Matrix;
use crate::matrix::MatrixError;
use crate::number::c64;
use lapack::{dgetri, zgetri};

impl Matrix {
  /// # Inverse
  /// with matrix decomposed by getrf
  pub fn getri(self, ipiv: &[i32]) -> Result<Matrix, MatrixError> {
    let n = self.rows();
    if n != self.cols() {
      return Err(MatrixError::DimensionMismatch);
    }

    let mut work = vec![f64::default(); n];
    let mut info = 0;

    let mut slf = self;
    let n = n as i32;

    unsafe {
      dgetri(n, &mut slf.elems, n, ipiv, &mut work, n, &mut info);
    }

    match info {
      0 => Ok(slf),
      _ => Err(MatrixError::LapackRoutineError {
        routine: "dgetri".to_owned(),
        info,
      }),
    }
  }
}

impl Matrix<c64> {
  /// # Inverse
  /// with matrix decomposed by getrf
  pub fn getri(self, ipiv: &[i32]) -> Result<Matrix<c64>, MatrixError> {
    let n = self.rows();
    if n != self.cols() {
      return Err(MatrixError::DimensionMismatch);
    }

    let mut work = vec![c64::default(); n];
    let mut info = 0;

    let mut slf = self;
    let n = n as i32;

    unsafe {
      zgetri(n, &mut slf.elems, n, ipiv, &mut work, n, &mut info);
    }

    match info {
      0 => Ok(slf),
      _ => Err(MatrixError::LapackRoutineError {
        routine: "zgetri".to_owned(),
        info,
      }),
    }
  }
}
