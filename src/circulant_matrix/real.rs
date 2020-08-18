use super::CirculantMatrix;
use crate::{
    diagonalized::Diagonalized,
    matrix::{operations::diag::diag, Matrix, Vector},
    number::c64,
    types::Square,
};
use std::f64::consts::PI;

fn fft(a: &mut [c64]) {
    let n = a.len();
    if n == 1 {
        return;
    }
    let mut b = vec![c64::default(); n / 2];
    let mut c = vec![c64::default(); n / 2];

    for i in 0..n {
        match i % 2 {
            0 => b[i / 2] = a[i],
            1 => c[i / 2] = a[i],
            _ => {}
        }
    }
    fft(&mut b);
    fft(&mut c);

    let omega = c64::new(0.0, 2.0 * PI / (n as f64)).exp();
    for i in 0..n {
        a[i] = b[i % (n / 2)] + c[i % (n / 2)] * omega.powi(i as i32);
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

        let mut c = self.row.to_row_vector().to_complex();
        fft(c.get_elements());

        let eigen_diag = diag(&c[0]);

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
