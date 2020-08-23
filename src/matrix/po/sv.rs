use crate::matrix::Matrix;
use crate::matrix::Vector;

impl Matrix {
    /// # Solve equations with Conjugate Gradient Method
    /// for positiveDefinite matrix
    pub fn posv_cgm(
        vec_mul: impl Fn(&[f64]) -> Result<Vec<f64>, String>,
        b: Vec<f64>,
        iterations: usize,
    ) -> Result<Vec<f64>, String> {
        let mut x = Matrix::new(b.len(), 1);
        let mut r = b.to_row_vector();
        let mut p = r.clone();

        for _ in 0..iterations {
            let r_t = r.t();
            let a_p = vec_mul(p.get_elements_ref())?.to_column_vector();
            let alpha = (&r_t * &p)[0][0] / (p.t() * &a_p)[0][0];

            let old_r = r.clone();
            x = x + p.clone() * alpha;
            r = r - a_p.clone() * alpha;

            let beta = (r.t() * &r)[0][0] / (&r_t * &old_r)[0][0];
            p = r.clone() + p.clone() * beta;
        }

        Ok(x.get_elements())
    }
}
