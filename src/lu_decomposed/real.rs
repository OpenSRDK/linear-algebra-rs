use super::LUDecomposed;
use crate::{
    matrix::Matrix,
    types::{PositiveDefinite, PositiveSemiDefinite, Square, UpperTriangle},
};
use lapack::dgetri;

macro_rules! implement {
  ( $($t: ty),+ ) => {
      $(
          impl LUDecomposed<$t, f64> {
              pub fn inv(mut self) -> Result<Matrix<$t, f64>, String> {
                  let n = self.matrix.get_rows();
                  let mut work = vec![0.0; n];
                  let mut info = 0;
                  unsafe {
                      dgetri(n as i32, self.matrix.get_elements(), n as i32, &self.ipiv, &mut work, n as i32, &mut info);
                  }

                  match info {
                      0 => Ok(self.matrix),
                      i => Err(i.to_string()),
                  }
              }

              pub fn det(&self) -> f64 {
                  self.matrix.transmute_ref::<UpperTriangle>().det()
              }
          }

      )+
  };
}
implement! {Square, PositiveDefinite, PositiveSemiDefinite}
