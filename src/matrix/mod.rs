pub mod operations;
pub mod operators;

use crate::number::{c64, Number};
use crate::types::{Standard, Type};
use rayon::prelude::*;
use std::marker::PhantomData;

/// # Matrix
#[derive(Clone, Debug, Default, Hash)]
pub struct Matrix<T = Standard, U = f64>
where
    T: Type,
    U: Number,
{
    rows: usize,
    columns: usize,
    elements: Vec<U>,
    phantom: PhantomData<T>,
}

impl<T, U> Matrix<T, U>
where
    T: Type,
    U: Number,
{
    pub fn new(rows: usize, columns: usize, elements: Vec<U>) -> Self {
        Self {
            rows,
            columns,
            elements,
            phantom: PhantomData,
        }
    }

    pub fn is_same_size<V: Type>(&self, rhs: &Matrix<V, U>) -> bool {
        self.rows == rhs.rows && self.columns == rhs.columns
    }

    pub fn get_rows(&self) -> usize {
        self.rows
    }

    pub fn get_columns(&self) -> usize {
        self.columns
    }

    pub fn get_elements(&mut self) -> &mut [U] {
        &mut self.elements
    }
}

impl<T> Matrix<T, f64>
where
    T: Type,
{
    pub fn to_complex(&self) -> Matrix<T, c64> {
        Matrix::<T, c64>::new(
            self.rows,
            self.columns,
            self.elements
                .par_iter()
                .map(|e| c64::new(*e, 0.0))
                .collect(),
        )
    }
}

impl<T> Matrix<T, c64>
where
    T: Type,
{
    pub fn to_real(&self) -> Matrix<T, f64> {
        Matrix::<T, f64>::new(
            self.rows,
            self.columns,
            self.elements.par_iter().map(|e| e.re).collect(),
        )
    }
}

pub trait Vector<U: Number> {
    fn to_row_vector(&self) -> Matrix<Standard, U>;
    fn to_column_vector(&self) -> Matrix<Standard, U>;
}

impl<U: Number> Vector<U> for [U] {
    fn to_row_vector(&self) -> Matrix<Standard, U> {
        Matrix::<Standard, U>::new(1, self.len(), self.to_vec())
    }

    fn to_column_vector(&self) -> Matrix<Standard, U> {
        Matrix::<Standard, U>::new(self.len(), 1, self.to_vec())
    }
}
