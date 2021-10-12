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

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let mut a = Matrix::<c64>::new(2, 3);
        a[(1, 2)] = c64::new(2.0, 3.0);
        let b = a.adjoint();

        assert_eq!(b[(2, 1)], c64::new(2.0, -3.0))
    }
}
