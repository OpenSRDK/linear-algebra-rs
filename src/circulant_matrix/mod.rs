pub mod teoplitz_matrix;

use crate::{
    diagonalized::Diagonalized,
    matrix::Vector,
    matrix::{operations::diag::diag, Matrix},
    number::{c64, Number},
    types::Square,
};
use std::f64::consts::PI;

pub struct CirculantMatrix<U>
where
    U: Number,
{
    row: Vec<U>,
}

impl<U> CirculantMatrix<U>
where
    U: Number,
{
    pub fn new(row: Vec<U>) -> Self {
        Self { row }
    }
}

impl CirculantMatrix<f64> {
    pub fn eigen_decomposition(&self) -> Diagonalized<c64> {
        let n = self.row.len();

        let mut fourier_matrix: Matrix<Square, c64> = Matrix::<Square, c64>::zeros(n);
        let omega = c64::new(0.0, 2.0 * PI / (n as f64)).exp();

        for i in 0..n {
            for j in 0..i {
                fourier_matrix[i][j] = fourier_matrix[j][i];
            }
            for j in i..n {
                fourier_matrix[i][j] = omega.powi((i * j) as i32);
            }
        }

        let eigenvalues = self.row.to_row_vector().to_complex() * &fourier_matrix;
        let eigen_diag = diag(&eigenvalues[0]);

        fourier_matrix = fourier_matrix * c64::new(1.0 / (n as f64).sqrt(), 0.0);

        let fourier_matrix_inv = fourier_matrix.adjoint();

        Diagonalized(fourier_matrix, eigen_diag, fourier_matrix_inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let a = CirculantMatrix::new(vec![1.0, 2.0, 3.0]);
        let diagonalized = a.eigen_decomposition();

        assert_eq!(diagonalized.1[0][0].re, 6.0);
    }
}
