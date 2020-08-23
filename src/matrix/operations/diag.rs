use crate::{matrix::Matrix, number::Number};

/// # Diagonal matrix
impl Matrix {
    pub fn diag<T>(vec: &[T]) -> Matrix<T>
    where
        T: Number,
    {
        let n = vec.len();
        let mut new_matrix = Matrix::<T>::new(n, n);
        for i in 0..n {
            new_matrix[i][i] = vec[i];
        }

        new_matrix
    }
}
