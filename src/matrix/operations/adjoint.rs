use crate::matrix::Matrix;
use crate::number::c64;

impl Matrix<c64> {
    pub fn adjoint(&self) -> Matrix<c64> {
        let mut new_matrix = Matrix::<c64>::new(self.cols, self.rows);

        for i in 0..new_matrix.rows {
            for j in 0..new_matrix.cols {
                new_matrix[i][j] = c64::new(self[j][i].re, -1.0 * self[j][i].im);
            }
        }

        new_matrix
    }
}
