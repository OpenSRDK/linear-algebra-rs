pub mod ci;
pub mod di;
pub mod ge;
pub mod operations;
pub mod operators;
pub mod po;
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
    columns: usize,
    elements: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Number,
{
    pub fn new(rows: usize, columns: usize, elements: Vec<T>) -> Self {
        Self {
            rows,
            columns,
            elements,
        }
    }

    pub fn is_same_size(&self, rhs: &Matrix<T>) -> bool {
        self.rows == rhs.rows && self.columns == rhs.columns
    }

    pub fn get_rows(&self) -> usize {
        self.rows
    }

    pub fn get_columns(&self) -> usize {
        self.columns
    }

    pub fn get_elements(&self) -> &[T] {
        &self.elements
    }
}

impl Matrix<f64> {
    pub fn to_complex(&self) -> Matrix<c64> {
        Matrix::<c64>::new(
            self.rows,
            self.columns,
            self.elements
                .par_iter()
                .map(|&e| c64::new(e, 0.0))
                .collect(),
        )
    }
}

impl Matrix<c64> {
    pub fn to_real(&self) -> Matrix<f64> {
        Matrix::new(
            self.rows,
            self.columns,
            self.elements.par_iter().map(|e| e.re).collect(),
        )
    }
}

pub trait Vector<T: Number> {
    fn to_row_vector(&self) -> Matrix<T>;
    fn to_column_vector(&self) -> Matrix<T>;
}

impl<T: Number> Vector<T> for [T] {
    fn to_row_vector(&self) -> Matrix<T> {
        Matrix::<T>::new(1, self.len(), self.to_vec())
    }

    fn to_column_vector(&self) -> Matrix<T> {
        Matrix::<T>::new(self.len(), 1, self.to_vec())
    }
}
