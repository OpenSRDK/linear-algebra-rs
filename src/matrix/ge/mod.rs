use crate::{
    number::{c64, Number},
    MatrixError,
};
use rayon::prelude::*;
use std::error::Error;

pub mod or_un;
pub mod sy_he;
pub mod tr;

pub mod mm;
pub mod operations;
pub mod operators;
pub mod svd;
pub mod trf;
pub mod tri;
pub mod trs;

/// # Matrix
/// ```
/// use opensrdk_linear_algebra::*;
///
/// let a = mat!(
///   1.0, 2.0;
///   3.0, 4.0
/// );
/// assert_eq!(a[0], [1.0, 3.0]);
/// assert_eq!(a[1], [2.0, 4.0]);
///
/// assert_eq!(a[(0, 0)], 1.0);
/// assert_eq!(a[(0, 1)], 2.0);
/// assert_eq!(a[(1, 0)], 3.0);
/// assert_eq!(a[(1, 1)], 4.0);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Hash)]
pub struct Matrix<T = f64>
where
    T: Number,
{
    rows: usize,
    cols: usize,
    elems: Vec<T>,
}

impl From<Box<dyn Error + Send + Sync>> for MatrixError {
    fn from(e: Box<dyn Error + Send + Sync>) -> Self {
        MatrixError::Others(e)
    }
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

    /// You can do `unwrap()` if you have a conviction that `elems.len() % rows == 0`
    pub fn from(rows: usize, elems: Vec<T>) -> Result<Self, MatrixError> {
        let cols = elems.len() / rows;

        if elems.len() != rows * cols {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self { rows, cols, elems })
    }

    pub fn is_same_size(&self, other: &Matrix<T>) -> bool {
        self.rows == other.rows && self.cols == other.cols
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

    pub fn elems(&self) -> &[T] {
        &self.elems
    }

    pub fn elems_mut(&mut self) -> &mut [T] {
        &mut self.elems
    }

    pub fn reshape(mut self, rows: usize) -> Self {
        self.rows = rows;
        self.cols = self.elems.len() / rows;

        self
    }

    pub fn eject_row(&self, index: usize) -> Vec<T> {
        (0..self.cols)
            .into_iter()
            .map(|j| self[(index, j)])
            .collect()
    }

    pub fn eject_sub_matrix(
        &self,
        start_i: usize,
        start_j: usize,
        rows: usize,
        cols: usize,
    ) -> Matrix<T> {
        Matrix::from(
            rows,
            (0..cols)
                .into_iter()
                .flat_map(|j| (0..rows).into_iter().map(move |i| (i, j)))
                .map(|(i, j)| self[(start_i + i, start_j + j)])
                .collect(),
        )
        .unwrap()
    }
}

impl Matrix<f64> {
    pub fn to_complex(&self) -> Matrix<c64> {
        Matrix::<c64>::from(
            self.rows,
            self.elems.par_iter().map(|&e| c64::new(e, 0.0)).collect(),
        )
        .unwrap()
    }
}

impl Matrix<c64> {
    pub fn to_real(&self) -> (Matrix<f64>, Matrix<f64>) {
        (
            Matrix::from(self.rows, self.elems.par_iter().map(|e| e.re).collect()).unwrap(),
            Matrix::from(self.rows, self.elems.par_iter().map(|e| e.im).collect()).unwrap(),
        )
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
        Matrix::<T>::from(1, self).unwrap()
    }

    fn col_mat(self) -> Matrix<T> {
        Matrix::<T>::from(self.len(), self).unwrap()
    }
}
