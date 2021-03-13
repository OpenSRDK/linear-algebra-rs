pub mod bd;
pub mod ci;
pub mod di;
pub mod ge;
pub mod gt;
pub mod kr;
pub mod operations;
pub mod operators;
pub mod po;
pub mod pt;
pub mod sp;
pub mod st;
pub mod sy;
pub mod to;
pub mod tr;

use crate::number::{c64, Number};
use rayon::prelude::*;

/// # Matrix
#[derive(Clone, Debug, Default, Hash, PartialEq)]
pub struct Matrix<T = f64>
where
  T: Number,
{
  rows: usize,
  cols: usize,
  elems: Vec<T>,
}

#[derive(thiserror::Error, Debug)]
pub enum MatrixError {
  #[error("Dimension mismatch.")]
  DimensionMismatch,
  #[error("BLAS routine error. routine: {routine}, info: {info}")]
  BlasRoutineError { routine: String, info: i32 },
  #[error("LAPACK routine error. routine: {routine}, info: {info}")]
  LapackRoutineError { routine: String, info: i32 },
}

impl<T> Matrix<T>
where
  T: Number,
{
  pub fn new(rows: usize, cols: usize) -> Self {
    Self {
      rows,
      cols,
      elems: vec![T::default(); rows * cols],
    }
  }

  pub fn from(rows: usize, elems: Vec<T>) -> Self {
    Self {
      rows,
      cols: elems.len() / rows,
      elems,
    }
  }

  pub fn same_size(&self, rhs: &Matrix<T>) -> bool {
    self.rows == rhs.rows && self.cols == rhs.cols
  }

  pub fn rows(&self) -> usize {
    self.rows
  }

  pub fn cols(&self) -> usize {
    self.cols
  }

  pub fn vec(self) -> Vec<T> {
    self.elems
  }

  pub fn slice(&self) -> &[T] {
    &self.elems
  }

  pub fn reshape(mut self, rows: usize) -> Self {
    self.rows = rows;
    self.cols = self.elems.len() / rows;

    self
  }
}

impl Matrix<f64> {
  pub fn to_complex(&self) -> Matrix<c64> {
    Matrix::<c64>::from(
      self.rows,
      self.elems.par_iter().map(|&e| c64::new(e, 0.0)).collect(),
    )
  }
}

impl Matrix<c64> {
  pub fn to_real(&self) -> Matrix<f64> {
    Matrix::from(self.rows, self.elems.par_iter().map(|e| e.re).collect())
  }
}

pub trait Vector<T>
where
  T: Number,
{
  fn row_mat(self) -> Matrix<T>;
  fn col_mat(self) -> Matrix<T>;
}

impl<T> Vector<T> for Vec<T>
where
  T: Number,
{
  fn row_mat(self) -> Matrix<T> {
    Matrix::<T>::from(1, self)
  }

  fn col_mat(self) -> Matrix<T> {
    Matrix::<T>::from(self.len(), self)
  }
}
