use crate::{matrix::Matrix, number::Number};

impl<T> Matrix<T>
where
    T: Number,
{
    /// # Determinant
    /// for triangle matrix
    /// To apply this method to none triangle matrix, use LU decomposition or Cholesky decomposition.
    pub fn det(&self) -> T {
        (0..self.rows).into_iter().map(|i| self[i][i]).product()
    }
}
