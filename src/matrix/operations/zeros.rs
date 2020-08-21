use crate::{matrix::Matrix, number::Number};

impl<T> Matrix<T>
where
    T: Number,
{
    pub fn zeros(rows: usize, columns: usize) -> Self {
        Self::new(rows, columns, vec![T::default(); rows * columns])
    }
}
