use crate::{matrix::Matrix, number::Number};

pub trait SubMatrix<T>: Send + Sync {
  fn size(&self) -> (usize, usize);
  fn index(&self, i: usize, j: usize) -> T;
}

impl<T> SubMatrix<T> for T
where
  T: Number,
{
  fn size(&self) -> (usize, usize) {
    (1usize, 1usize)
  }

  fn index(&self, _: usize, _: usize) -> T {
    *self
  }
}

impl<T> SubMatrix<T> for Matrix<T>
where
  T: Number,
{
  fn size(&self) -> (usize, usize) {
    (self.rows(), self.cols())
  }

  fn index(&self, i: usize, j: usize) -> T {
    self[(i, j)]
  }
}

impl<T> SubMatrix<T> for &Matrix<T>
where
  T: Number,
{
  fn size(&self) -> (usize, usize) {
    (self.rows(), self.cols())
  }

  fn index(&self, i: usize, j: usize) -> T {
    self[(i, j)]
  }
}
