use crate::matrix::ci::CirculantMatrix;
use crate::{matrix::Matrix, number::c64};
use rayon::prelude::*;
use rustfft::FftPlanner;
use std::f64::consts::PI;

impl CirculantMatrix<f64> {
    pub fn cievd(&self) -> (Matrix<c64>, Vec<c64>) {
        let row_elems = self.row_elems();
        let n = row_elems.len();

        let mut fourier_matrix: Matrix<c64> = Matrix::<c64>::new(n, n);
        let omega = c64::new(0.0, 2.0 * PI / (n as f64)).exp();

        for i in 0..n {
            for j in 0..i {
                fourier_matrix[i][j] = fourier_matrix[j][i];
            }
            for j in i..n {
                fourier_matrix[i][j] = omega.powi((i * j) as i32);
            }
        }

        let mut buffer = row_elems
            .par_iter()
            .map(|&e| c64::new(e, 0.0))
            .collect::<Vec<c64>>();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        (fourier_matrix, buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let a = CirculantMatrix::new(vec![1.0, 2.0, 3.0]);
        let diagonalized = a.cievd();

        assert_eq!(diagonalized.1[0].re, 6.0);
    }
}
