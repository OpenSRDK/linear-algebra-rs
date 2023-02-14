use crate::{ge::Matrix, matrix::ge::Vector, MatrixError};
use std::error::Error;

impl Matrix {
    /// # Solve equations with Conjugate Gradient Method
    /// for positiveDefinite matrix
    pub fn posv_cgm(
        vec_mul: &dyn Fn(Vec<f64>) -> Result<Vec<f64>, Box<dyn Error + Send + Sync>>,
        b: Vec<f64>,
        iterations: usize,
    ) -> Result<Vec<f64>, MatrixError> {
        let mut x = Matrix::new(b.len(), 1);
        let mut r = b.col_mat();
        let mut p = r.clone();

        for _ in 0..iterations {
            let r_t = r.t();
            let a_p = match vec_mul(p.clone().vec()) {
                Ok(v) => Ok(v),
                Err(e) => Err(MatrixError::Others(e)),
            }?
            .col_mat();
            let alpha = r_t.dot(&p)[0][0] / p.t().dot(&a_p)[0][0];

            let old_r = r.clone();
            x = x + p.clone() * alpha;
            r = r - a_p.clone() * alpha;

            let beta = r.t().dot(&r)[0][0] / r_t.dot(&old_r)[0][0];
            p = r.clone() + p * beta;
        }

        Ok(x.vec())
    }
}
