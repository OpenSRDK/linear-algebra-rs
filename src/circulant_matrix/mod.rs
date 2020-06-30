pub mod teoplitz_matrix;

use crate::{
    diagonalized::Diagonalized,
    matrix::{operations::diag::diag, Matrix},
    number::{c64, Number},
    types::Square,
};
use rayon::prelude::*;
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
    /// Read MLP series "GP amd ML"
    pub fn eigen_decomposition(&self) -> Diagonalized<c64> {
        let n = self.row.len();

        let mut fourier_matrix: Matrix<Square, c64> = Matrix::<Square, c64>::zeros(n);

        for i in 0..n {
            for j in 0..i {
                fourier_matrix[i][j] = fourier_matrix[j][i];
            }
            for j in i..n {
                fourier_matrix[i][j] =
                    c64::new(0.0, -2.0 * PI * (i as f64) * (j as f64) / (n as f64));
            }
        }

        let fourier_matrix_t = fourier_matrix.t();

        let eigenvalues: Vec<c64> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..n)
                    .into_par_iter()
                    .map(|j| self.row[j] * fourier_matrix_t[i][j])
                    .sum()
            })
            .collect();

        let eigen_diag = diag(&eigenvalues);

        Diagonalized(fourier_matrix, eigen_diag, fourier_matrix_t)
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let a = CirculantMatrix::new(vec![1.0, 2.0]);
        let diagonalized = a.eigen_decomposition();

        assert_eq!(diagonalized.1[0][0].re, -1.0);
        assert_eq!(diagonalized.1[1][1].re, 3.0);
    }
}
