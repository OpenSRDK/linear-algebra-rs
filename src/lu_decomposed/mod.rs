pub mod complex;
pub mod real;

use crate::{matrix::Matrix, number::Number, types::Type};

/// # LU decomposed
pub struct LUDecomposed<T, U = f64>
where
    T: Type,
    U: Number,
{
    matrix: Matrix<T, U>,
    ipiv: Vec<i32>,
}

impl<T, U> LUDecomposed<T, U>
where
    T: Type,
    U: Number,
{
    pub fn new(matrix: Matrix<T, U>, ipiv: Vec<i32>) -> Self {
        Self { matrix, ipiv }
    }
}
