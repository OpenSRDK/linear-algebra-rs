use crate::{matrix::Matrix, number::c64};

impl Matrix<f64> {
  /// # Pow integer
  /// for diagonal matrix
  pub fn dipowi(self, exp: i32) -> Self {
    let n = self.rows;
    let mut slf = self;

    for i in 0..n {
      slf[i][i] = slf[i][i].powi(exp);
    }

    slf
  }
}

impl Matrix<c64> {
  /// # Pow integer
  /// for diagonal matrix
  pub fn dipowi(self, exp: i32) -> Self {
    let n = self.rows;
    let mut slf = self;

    for i in 0..n {
      slf[i][i] = slf[i][i].powi(exp);
    }

    slf
  }
}

#[cfg(test)]
mod tests {
  use crate::*;
  #[test]
  fn it_works() {
    let a = Matrix::<f64>::identity(2);
    let a_inv = a.clone().diinv();

    assert_eq!(a[0][0], a_inv[0][0]);

    let a = Matrix::<c64>::identity(2);
    let a_inv = a.clone().diinv();

    assert_eq!(a[0][0], a_inv[0][0]);
  }
}
