use crate::matrix::MatrixError;
use crate::{matrix::Matrix, number::Number};
use std::error::Error;

#[derive(Clone, Debug)]
pub struct KroneckerMatrices<T = f64>
where
    T: Number,
{
    k: Vec<Matrix<T>>,
    rows: usize,
    cols: usize,
}

impl<T> KroneckerMatrices<T>
where
    T: Number,
{
    pub fn new(k: Vec<Matrix<T>>) -> Self {
        let (rows, cols) = k
            .iter()
            .fold((1usize, 1usize), |v, m| (v.0 * m.rows, v.1 * m.cols));
        Self { k, rows, cols }
    }

    pub fn elems_ref(&self) -> &[Matrix<T>] {
        &self.k
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn prod(&self) -> Matrix<T> {
        let mut new_matrix = Matrix::from(self.rows, vec![T::one(); self.rows * self.cols]);
        let k_len = self.k.len();

        let mut row_block = 1;
        let mut col_block = 1;

        for p in (0..k_len).rev() {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    new_matrix[i][j] *=
                        self.k[p][i / row_block % self.k[p].rows][j / col_block % self.k[p].cols];
                }
            }

            row_block *= self.k[p].rows;
            col_block *= self.k[p].cols;
        }

        new_matrix
    }
}

impl KroneckerMatrices {
    pub fn vec_mul(&self, v: &[f64]) -> Result<Vec<f64>, Box<dyn Error>> {
        if self.k.len() == 0 || self.rows == 0 || self.cols == 0 {
            return Err(MatrixError::Empty.into());
        }

        let n = v.len();

        if self.cols != n {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let k_len = self.k.len();
        let nu = self.rows / self.k[k_len - 1].rows();
        let mut u = Matrix::from(nu, v.to_vec()).t();

        for p in (1..k_len).rev() {
            let nu = self.k[p - 1].cols();
            let ku = &self.k[p - 1] * u;
            u = Matrix::from(nu, ku.elems()).t();
        }

        let ku = &self.k[0] * u;

        Ok(ku.t().elems())
    }
}
