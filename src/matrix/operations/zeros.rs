use crate::{
    matrix::Matrix,
    number::Number,
    types::{Square, Standard},
};

impl<U> Matrix<Standard, U>
where
    U: Number,
{
    pub fn zeros(rows: usize, columns: usize) -> Self {
        Self::new(rows, columns, vec![U::default(); rows * columns])
    }
}

impl<U> Matrix<Square, U>
where
    U: Number,
{
    pub fn zeros(dim: usize) -> Self {
        Self::new(dim, dim, vec![U::default(); dim * dim])
    }
}
