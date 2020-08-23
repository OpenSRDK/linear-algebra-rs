use crate::{matrix::Matrix, number::Number};

pub struct KroneckerMatrices<T = f64>
where
    T: Number,
{
    k: Vec<Matrix<T>>,
    rows_sum: usize,
    columns_sum: usize,
}

impl<T> KroneckerMatrices<T>
where
    T: Number,
{
    pub fn new(k: Vec<Matrix<T>>) -> Self {
        let (rows_sum, columns_sum) = k
            .iter()
            .fold((0usize, 0usize), |v, m| (v.0 + m.rows, v.1 + m.columns));
        Self {
            k,
            rows_sum,
            columns_sum,
        }
    }
}

impl KroneckerMatrices {
    pub fn vec_mul(&self, v: &[f64]) -> Result<Vec<f64>, String> {
        if self.k.len() == 0 || self.rows_sum == 0 || self.columns_sum == 0 {
            return Err("empty".to_owned());
        }

        let n = v.len();

        if self.columns_sum != n {
            return Err("dimension mismatch".to_owned());
        }

        let k_len = self.k.len();
        let nu = self.rows_sum / self.k[k_len - 1].get_rows();
        let mut u = Matrix::from(nu, v.to_vec()).t();

        for p in (1..k_len).rev() {
            let nu = self.k[p - 1].get_columns();
            let ku = &self.k[p - 1] * u;
            u = Matrix::from(nu, ku.get_elements()).t();
        }

        let ku = &self.k[0] * u;

        Ok(ku.t().get_elements())
    }
}
