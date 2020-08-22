use crate::matrix::{Matrix, Vector};
use lapack::{dorgtr, dsytrd};
use rayon::prelude::*;

impl Matrix {
    /// # Tridiagonalize
    /// for symmetric matrix
    pub fn sytrd(self) -> Result<(Matrix, Vec<f64>, Vec<f64>), String> {
        if self.rows == 0 || self.rows != self.columns {
            return Err("dimension mismatch".to_owned());
        }
        let n = self.rows as i32;
        let mut slf = self;
        let mut d = vec![0.0; slf.rows];
        let mut e = vec![0.0; slf.rows - 1];
        let mut tau = vec![0.0; slf.rows - 1];
        let lwork = 2 * slf.rows;
        let mut work = vec![0.0; lwork];
        let mut info = 0;

        unsafe {
            dsytrd(
                'U' as u8,
                n,
                &mut slf.elements,
                n,
                &mut d,
                &mut e,
                &mut tau,
                &mut work,
                lwork as i32,
                &mut info,
            );
            if info != 0 {
                return Err(info.to_string());
            }

            dorgtr(
                'U' as u8,
                n,
                &mut slf.elements,
                n,
                &tau,
                &mut work,
                lwork as i32,
                &mut info,
            )
        }

        let v = slf;

        Ok((v, d, e))
    }

    /// # Lanczos algorithm
    /// for symmetric matrix
    /// only k iteration
    pub fn sytrd_k(&self, k: usize) -> Result<(Vec<f64>, Vec<f64>, Matrix), String> {
        let n = self.rows;
        if n == 0 || n != self.columns {
            return Err("dimension mismatch".to_owned());
        }
        let k = k.min(n);

        let mut d = vec![0.0; k];
        let mut e = vec![0.0; k - 1];

        let mut u = vec![vec![0.0; n]; k];
        u[0][0] = 1.0;
        let mut u_prev = vec![0.0; n].to_column_vector();
        let mut e_prev = 0.0;

        for i in 0..k {
            let u_mat = u[i].clone().to_column_vector();
            d[i] = (u_mat.t() * self * &u_mat)[0][0];

            if i + 1 == k {
                break;
            }

            let v: Matrix = self * &u_mat - e_prev * u_prev.clone() - d[i] * u_mat.clone();
            e[i] = v
                .elements
                .par_iter()
                .map(|&v_e| v_e.powi(2))
                .sum::<f64>()
                .sqrt();
            u[i + 1] = ((1.0 / e[i]) * v).elements;

            u_prev = u_mat;
            e_prev = e[i];
        }
        let q_t = Matrix::new(k, n, u.concat());

        Ok((d, e, q_t))
    }
}
