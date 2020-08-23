use crate::matrix::ci::CirculantMatrix;
use crate::number::Number;
pub struct ToeplitzMatrix<T = f64>
where
    T: Number,
{
    dim: usize,
    row: Vec<T>,
    column: Vec<T>,
}

impl<T> ToeplitzMatrix<T>
where
    T: Number,
{
    /// must be row[0] == column[0]
    pub fn new(row: Vec<T>, column: Vec<T>) -> Self {
        let dim = row.len();

        if column.len() != dim {
            panic!("dimension mismatch")
        }
        if row[0] != column[0] {
            panic!("first element mismatch")
        }

        Self { dim, row, column }
    }

    pub fn get_dim(&self) -> usize {
        self.dim
    }

    pub fn get_row(&self) -> &[T] {
        &self.row
    }

    pub fn get_column(&self) -> &[T] {
        &self.column
    }

    pub fn embedded_circulant(&self) -> CirculantMatrix<T> {
        let row = (0..self.dim)
            .into_iter()
            .chain((1..self.dim - 1).rev().into_iter())
            .map(|i| self.row[i])
            .collect();

        CirculantMatrix::<T>::new(row)
    }
}
