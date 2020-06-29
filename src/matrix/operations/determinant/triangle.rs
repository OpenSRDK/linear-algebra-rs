use crate::{types::{Type, UpperTriangle, LowerTriangle, Diagonal}, number::Number, matrix::Matrix};

fn determinant_triangle<T, U>(slf: &Matrix<T, U>) -> U
where
    T: Type,
    U: Number,
{
    (0..slf.rows).into_iter().map(|i| slf[i][i]).product()
}

macro_rules! implement_triangle {
  ( $($t: ty),+ ) => {
      $(
          impl<U> Matrix<$t, U> where U: Number {
              pub fn determinant(&self) -> U {
                  determinant_triangle(self)
              }
          }
      )+
  };
}
implement_triangle! {UpperTriangle, LowerTriangle, Diagonal}
