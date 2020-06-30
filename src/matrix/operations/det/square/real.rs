use crate::matrix::Matrix;
use crate::types::{PositiveDefinite, PositiveSemiDefinite, Square, Type, UpperTriangle};
use lapack::dgetrf;

fn det_square<T>(slf: &Matrix<T, f64>) -> Result<f64, String>
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
        i if i == 0 => Ok(solution_matrix.det()),
        i => Err(i.to_string()),
    }
}

macro_rules! implement_square {
  ( $($t: ty),+ ) => {
      $(
          impl Matrix<$t> {
              /// # Determinant
              /// for $t Matrix
              pub fn det(&self) -> Result<f64, String> {
                  det_square(self)
              }
          }
      )+
  };
}

implement_square! {Square, PositiveDefinite, PositiveSemiDefinite}
