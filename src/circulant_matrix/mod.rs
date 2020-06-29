use crate::{
    diagonalized::Diagonalized,
    matrix::{operations::diag::diag, Matrix},
    number::{c64, Number},
    types::Square,
};
use rayon::prelude::*;
use std::f64::consts::PI;

pub struct ToeplitzMatrix<U>
where
    U: Number,
{
    dim: usize,
    row: Vec<U>,
    column: Vec<U>,
}

impl<U> ToeplitzMatrix<U>
where
    U: Number,
{
    pub fn new(row: Vec<U>, column: Vec<U>) -> Self {
        let dim = row.len();

        if column.len() != dim {
            panic!("different dimensions")
        }
        if dim < 2 || row[0] != column[0] {
            panic!("")
        }

        Self { dim, row, column }
    }

    pub fn embedded_circulant(&self) -> CirculantMatrix<U> {
        let row = (0..self.dim)
            .into_iter()
            .chain((1..self.dim - 1).rev().into_iter())
            .map(|i| self.row[i])
            .collect();

        CirculantMatrix::<U>::new(row)
    }
}

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
            for j in 0..n {
                fourier_matrix[i][j] =
                    c64::new(0.0, -2.0 * PI * (i as f64) * (j as f64) / (n as f64));
            }
        }

        let fourier_matrix_t = fourier_matrix.transpose();

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
