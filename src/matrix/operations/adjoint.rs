use crate::matrix::Matrix;
use crate::number::c64;

impl Matrix<c64> {
    pub fn adjoint(&self) -> Matrix<c64> {
        let mut new_matrix = Matrix::<c64>::new(self.cols, self.rows);

        for j in 0..new_matrix.cols {
            for i in 0..new_matrix.rows {
                new_matrix[j][i] = c64::new(self[i][j].re, -1.0 * self[i][j].im);
            }
        }

        new_matrix
    }
}
