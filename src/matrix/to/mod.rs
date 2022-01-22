use crate::matrix::ci::CirculantMatrix;
use crate::{matrix::MatrixError, number::Number};
use rayon::prelude::*;

#[derive(Clone, Debug, Default, PartialEq, Hash)]
pub struct ToeplitzMatrix<T = f64>
where
    T: Number,
{
    col_elems: Vec<T>,
    row_elems: Vec<T>,
}

impl<T> ToeplitzMatrix<T>
where
    T: Number,
{
    pub fn new(dim: usize) -> Self {
        Self {
            col_elems: vec![T::default(); dim],
            row_elems: vec![T::default(); dim.max(1) - 1],
        }
    }

    /// - `col_elems`: First column elements. The length must be `dimension`.
    /// - `row_elems`: First roe elements without first element. The length must be `dimension - 1`.
    pub fn from(col_elems: Vec<T>, row_elems: Vec<T>) -> Result<Self, MatrixError> {
        let dim = col_elems.len();

        if row_elems.len() != dim.max(1) - 1 {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self {
            col_elems,
            row_elems,
        })
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.col_elems.len()
    }

    /// First column elements.
    pub fn col_elems(&self) -> &[T] {
        &self.col_elems
    }

    /// First row elements.
    pub fn row_elems(&self) -> &[T] {
        &self.row_elems
    }

    /// Returns `(self.row_elems, self.col_elems)`
    pub fn eject(self) -> (Vec<T>, Vec<T>) {
        (self.row_elems, self.col_elems)
    }

    pub fn embedded_circulant(&self) -> CirculantMatrix<T> {
        let col_elems = (0..self.dim())
            .into_par_iter()
            .chain((1..self.dim() - 1).into_par_iter().rev())
            .map(|i| self.col_elems[i])
            .collect();

        CirculantMatrix::<T>::new(col_elems)
    }
}
