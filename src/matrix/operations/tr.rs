use crate::{matrix::Matrix, number::Number};
use rayon::prelude::*;

impl<T> Matrix<T>
where
  T: Number,
{
  /// # Trace
  pub fn tr(&self) -> T {
    (0..self.rows).into_par_iter().map(|i| self[i][i]).sum()
  }
}
