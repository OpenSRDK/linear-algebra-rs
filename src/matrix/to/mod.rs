use crate::matrix::ci::CirculantMatrix;
use crate::{matrix::MatrixError, number::Number};

#[derive(Clone, Debug, Default, Hash)]
pub struct ToeplitzMatrix<T = f64>
where
  T: Number,
{
  dim: usize,
  row_elems: Vec<T>,
  col_elems: Vec<T>,
}

impl<T> ToeplitzMatrix<T>
where
  T: Number,
{
  /// must be row.len() - 1 = col.len()
  pub fn new(row_elems: Vec<T>, col_elems: Vec<T>) -> Result<Self, MatrixError> {
    let dim = row_elems.len();

    if col_elems.len() != dim.max(1) - 1 {
      return Err(MatrixError::DimensionMismatch);
    }

    Ok(Self {
      dim,
      row_elems,
      col_elems,
    })
  }

  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn row_elems(&self) -> &[T] {
    &self.row_elems
  }

  pub fn col_elems(&self) -> &[T] {
    &self.col_elems
  }

  pub fn eject(self) -> (Vec<T>, Vec<T>) {
    (self.row_elems, self.col_elems)
  }

  pub fn embedded_circulant(&self) -> CirculantMatrix<T> {
    let row = (0..self.dim)
      .into_iter()
      .chain((1..self.dim - 1).rev().into_iter())
      .map(|i| self.row_elems[i])
      .collect();

    CirculantMatrix::<T>::new(row)
  }
}
