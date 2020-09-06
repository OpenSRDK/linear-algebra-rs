use crate::matrix::MatrixError;
use crate::{matrix::Matrix, number::Number};
use std::error::Error;

pub struct KroneckerMatrices<T = f64>
where
    T: Number,
{
    k: Vec<Matrix<T>>,
    rows_sum: usize,
    cols_sum: usize,
}

impl<T> KroneckerMatrices<T>
where
    T: Number,
{
    pub fn new(k: Vec<Matrix<T>>) -> Self {
        let (rows_sum, cols_sum) = k
            .iter()
            .fold((0usize, 0usize), |v, m| (v.0 + m.rows, v.1 + m.cols));
        Self {
            k,
            rows_sum,
            cols_sum,
        }
    }

    pub fn elems_ref(&self) -> &[Matrix<T>] {
        &self.k
    }

    pub fn rows_sum(&self) -> usize {
        self.rows_sum
    }

    pub fn cols_sum(&self) -> usize {
        self.cols_sum
    }
}

impl KroneckerMatrices {
    pub fn vec_mul(&self, v: &[f64]) -> Result<Vec<f64>, Box<dyn Error>> {
        if self.k.len() == 0 || self.rows_sum == 0 || self.cols_sum == 0 {
            return Err(MatrixError::Empty.into());
        }

        let n = v.len();

        if self.cols_sum != n {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let k_len = self.k.len();
        let nu = self.rows_sum / self.k[k_len - 1].rows();
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
