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
    columns: usize,
    elements: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Number,
{
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            elements: vec![T::default(); rows * columns],
        }
    }

    pub fn from(rows: usize, elements: Vec<T>) -> Self {
        Self {
            rows,
            columns: elements.len() / rows,
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

    pub fn get_elements_ref(&self) -> &[T] {
        &self.elements
    }

    pub fn get_elements(self) -> Vec<T> {
        self.elements
    }
}

impl Matrix<f64> {
    pub fn to_complex(&self) -> Matrix<c64> {
        Matrix::<c64>::from(
            self.rows,
            self.elements
                .par_iter()
                .map(|&e| c64::new(e, 0.0))
                .collect(),
        )
    }
}

impl Matrix<c64> {
    pub fn to_real(&self) -> Matrix<f64> {
        Matrix::from(self.rows, self.elements.par_iter().map(|e| e.re).collect())
    }
}

pub trait Vector<T: Number> {
    fn to_row_vector(self) -> Matrix<T>;
    fn to_column_vector(self) -> Matrix<T>;
}

impl<T: Number> Vector<T> for Vec<T> {
    fn to_row_vector(self) -> Matrix<T> {
        Matrix::<T>::from(1, self)
    }

    fn to_column_vector(self) -> Matrix<T> {
        Matrix::<T>::from(self.len(), self)
    }
}
