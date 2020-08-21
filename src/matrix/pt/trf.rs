use crate::matrix::Matrix;
use lapack::dpttrf;
use rayon::prelude::*;

impl Matrix {
    /// # Cholesky decomposition
    /// for tridiagonal matrix
    pub fn dpttrf(d: Vec<f64>, e: Vec<f64>) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = d.len() as i32;
        let mut d_mut = d;
        let mut e_mut = e;
        let mut info = 0;

        unsafe { dpttrf(n, &mut d_mut, &mut e_mut, &mut info) }

        if info != 0 {
            return Err(info.to_string());
        }

        d_mut.par_iter_mut().for_each(|d_e| *d_e = d_e.powf(0.5));

        Ok((d_mut, e_mut))
    }
}
