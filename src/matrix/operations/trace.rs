use crate::{matrix::Matrix, number::Number, types::*};

fn trace<T, U>(slf: &Matrix<T, U>) -> U
where
    T: Type,
    U: Number,
{
    (0..slf.rows).into_iter().map(|i| slf[i][i]).sum()
}

macro_rules! implement {
  ( $($t: ty),+ ) => {
      $(
          impl<U: Number> Matrix<$t, U> {
              /// # Trace
              pub fn trace(&self) -> U {
                  trace(self)
              }
          }
      )+
  };
}

implement! {Square, UpperTriangle, LowerTriangle, Diagonal, PositiveDefinite, PositiveSemiDefinite}
