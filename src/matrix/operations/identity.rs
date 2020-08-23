use crate::{matrix::Matrix, number::Number};

impl<T> Matrix<T>
where
    T: Number,
{
    /// # Identity
    pub fn identity(n: usize) -> Self {
        let mut new_matrix = Matrix::<T>::new(n, n);
        for i in 0..n {
            new_matrix[i][i] = T::one();
        }

        new_matrix
    }
}
