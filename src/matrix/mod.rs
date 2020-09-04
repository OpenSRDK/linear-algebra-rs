pub mod bd;
pub mod ci;
pub mod di;
pub mod ge;
pub mod kr;
pub mod operations;
pub mod operators;
pub mod po;
pub mod pt;
pub mod st;
pub mod sy;
pub mod to;
pub mod tr;

use crate::number::{c64, Number};
use rayon::prelude::*;

/// # Matrix
#[derive(Clone, Debug, Default, Hash)]
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
    #[error("empty")]
    Empty,
    #[error("dimension mismatch")]
    DimensionMismatch,
    #[error("BLAS routine error, routine: {routine}, info: {info}")]
    BlasRoutineError { routine: String, info: i32 },
    #[error("LAPACK routine error, routine: {routine}, info: {info}")]
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

    pub fn row(v: Vec<T>) -> Self {
        Matrix::<T>::from(1, v)
    }

    pub fn col(v: Vec<T>) -> Self {
        Matrix::<T>::from(v.len(), v)
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

    pub fn elems(self) -> Vec<T> {
        self.elems
    }

    pub fn elems_ref(&self) -> &[T] {
        &self.elems
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
