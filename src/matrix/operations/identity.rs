use crate::{matrix::Matrix, number::Number};

impl<T> Matrix<T>
where
    T: Number,
{
    /// # Identity
    pub fn identity(n: usize) -> Matrix<T> {
        let mut new_matrix = Matrix::<T>::zeros(n, n);
        for i in 0..n {
            new_matrix[i][i] = T::one();
        }

        new_matrix
    }
}
