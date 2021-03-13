use super::SymmetricTridiagonalMatrix;
use crate::matrix::*;
use lapack::dstevd;
use std::error::Error;

impl SymmetricTridiagonalMatrix<f64> {
  /// # Eigen decomposition
  /// return (lambda, pt)
  pub fn stevd(self) -> Result<(Vec<f64>, Matrix), Box<dyn Error>> {
    let (mut d, mut e) = self.eject();
    let n = d.len();
    let mut z = Matrix::new(n, n);
    let lwork = 1.max(1 + 4 * n + n.pow(2));
    let mut work = vec![0.0; lwork];
    let liwork = 1.max(3 + 5 * n);
    let mut iwork = vec![0; liwork];
    let mut info = 0;

    let n = n as i32;

    unsafe {
      dstevd(
        'V' as u8,
        n,
        &mut d,
        &mut e,
        &mut z.elems,
        n,
        &mut work,
        lwork as i32,
        &mut iwork,
        liwork as i32,
        &mut info,
      )
    }

    match info {
      0 => Ok((d, z)),
      _ => Err(
        MatrixError::LapackRoutineError {
          routine: "dstev".to_owned(),
          info,
        }
        .into(),
      ),
    }
  }
}
