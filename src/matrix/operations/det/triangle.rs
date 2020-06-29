use crate::{
    matrix::Matrix,
    number::Number,
    types::{Diagonal, LowerTriangle, Type, UpperTriangle},
};

fn det_triangle<T, U>(slf: &Matrix<T, U>) -> U
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
              /// # Determinant
              /// for $t Matrix
              pub fn det(&self) -> U {
                  det_triangle(self)
              }
          }
      )+
  };
}
implement_triangle! {UpperTriangle, LowerTriangle, Diagonal}
