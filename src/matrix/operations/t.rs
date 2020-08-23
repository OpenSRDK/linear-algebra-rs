use crate::matrix::Matrix;
use crate::number::Number;

impl<T> Matrix<T>
where
    T: Number,
{
    /// # Transpose
    pub fn t(&self) -> Matrix<T> {
        let mut new_matrix = Matrix::<T>::new(self.cols, self.rows);

        for i in 0..new_matrix.rows {
            for j in 0..new_matrix.cols {
                new_matrix[i][j] = self[j][i];
            }
        }

        new_matrix
    }
}
