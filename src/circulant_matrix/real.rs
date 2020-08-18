use super::CirculantMatrix;
use crate::{
    diagonalized::Diagonalized,
    matrix::{operations::diag::diag, Matrix},
    number::c64,
    types::Square,
};
use rayon::prelude::*;
use rustfft::FFTplanner;
use std::f64::consts::PI;
use std::mem::transmute;

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

        let mut input = self
            .row
            .par_iter()
            .map(|&e| c64::new(e, 0.0))
            .collect::<Vec<c64>>();

        let mut output = vec![c64::default(); n];

        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(n);
        unsafe {
            fft.process(
                transmute::<&mut [c64], &mut [rustfft::num_complex::Complex<f64>]>(&mut input),
                transmute::<&mut [c64], &mut [rustfft::num_complex::Complex<f64>]>(&mut output),
            );
        }

        let eigen_diag = diag(&output);

        fourier_matrix = fourier_matrix * c64::new(1.0 / (n as f64).sqrt(), 0.0);

        let fourier_matrix_inv = fourier_matrix.t();

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
