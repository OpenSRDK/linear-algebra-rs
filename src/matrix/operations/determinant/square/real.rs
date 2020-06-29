use crate::matrix::Matrix;
use crate::types::{Type, UpperTriangle, Square, PositiveDefinite, PositiveSemiDefinite};
use lapack::dgetrf;

fn determinant_square<T>(slf: &Matrix<T, f64>) -> Result<f64, i32>
where
    T: Type,
{
    let mut solution_matrix = slf.clone().transmute::<UpperTriangle>();

    let mut ipiv = vec![0; slf.rows];
    let mut info = 0;

    unsafe {
        dgetrf(
            slf.rows as i32,
            slf.rows as i32,
            &mut solution_matrix.elements,
            slf.rows as i32,
            &mut ipiv,
            &mut info,
        );
    }

    match info {
        i if i == 0 => Ok(solution_matrix.determinant()),
        i => Err(i),
    }
}

macro_rules! implement_square {
  ( $($t: ty),+ ) => {
      $(
          impl Matrix<$t> {
              pub fn determinant(&self) -> Result<f64, i32> {
                  determinant_square(self)
              }
          }
      )+
  };
}

implement_square! {Square, PositiveDefinite, PositiveSemiDefinite}
