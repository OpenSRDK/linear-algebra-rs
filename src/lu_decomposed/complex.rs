use super::LUDecomposed;
use crate::{
    matrix::Matrix,
    number::c64,
    types::{PositiveDefinite, PositiveSemiDefinite, Square, UpperTriangle},
};
use lapack::zgetri;
use std::mem::transmute;

macro_rules! implement {
  ( $($t: ty),+ ) => {
      $(
          impl LUDecomposed<$t, c64> {
              pub fn inv(mut self) -> Result<Matrix<$t, c64>, String> {
                  let n = self.matrix.get_rows();
                  let mut work = vec![c64::default(); n];
                  let mut info = 0;
                  unsafe {
                      zgetri(n as i32, transmute::<&mut [c64], &mut [blas::c64]>(self.matrix.get_elements()), n as i32, &self.ipiv, transmute::<&mut [c64], &mut [blas::c64]>(&mut work), n as i32, &mut info);
                  }

                  match info {
                    0 => Ok(self.matrix),
                    i => Err(i.to_string()),
                  }
              }

              pub fn det(&self) -> c64 {
                  self.matrix.transmute_ref::<UpperTriangle>().det()
              }
          }

      )+
  };
}
implement! {Square, PositiveDefinite, PositiveSemiDefinite}
