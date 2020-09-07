use crate::{
    matrix::{Matrix, MatrixError, Vector},
    st::SymmetricTridiagonalMatrix,
};
use lapack::{dorgtr, dsytrd};
use rayon::prelude::*;
use std::error::Error;

impl Matrix {
    /// # Tridiagonalize
    /// for symmetric matrix
    pub fn sytrd(self) -> Result<(Matrix, SymmetricTridiagonalMatrix), Box<dyn Error>> {
        if self.rows == 0 || self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch.into());
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
                &mut slf.elems,
                n,
                &mut d,
                &mut e,
                &mut tau,
                &mut work,
                lwork as i32,
                &mut info,
            );
            if info != 0 {
                return Err(MatrixError::LapackRoutineError {
                    routine: "dsytrd".to_owned(),
                    info,
                }
                .into());
            }

            dorgtr(
                'U' as u8,
                n,
                &mut slf.elems,
                n,
                &tau,
                &mut work,
                lwork as i32,
                &mut info,
            )
        }

        let v = slf;
        let t = SymmetricTridiagonalMatrix::new(d, e);

        Ok((v, t))
    }

    /// # Lanczos algorithm
    /// for symmetric matrix
    /// only k iteration
    pub fn sytrd_k(
        n: usize,
        k: usize,
        vec_mul: &dyn Fn(&[f64]) -> Result<Vec<f64>, Box<dyn Error>>,
        probe: Option<&[f64]>,
    ) -> Result<(SymmetricTridiagonalMatrix, Matrix), Box<dyn Error>> {
        if n == 0 {
            return Err(MatrixError::Empty.into());
        }
        let k = k.min(n);

        let mut d = vec![0.0; k];
        let mut e = vec![0.0; k - 1];

        let mut u = vec![vec![0.0; n]; k];

        match probe {
            Some(v) => {
                if v.len() != n {
                    return Err(MatrixError::DimensionMismatch.into());
                }
                let norm = v.par_iter().map(|&vi| vi.powi(2)).sum::<f64>().sqrt();
                u[0].par_iter_mut()
                    .zip(v.par_iter())
                    .for_each(|(m, &vi)| *m = vi / norm);
            }
            None => {
                u[0][0] = 1.0;
            }
        }

        let mut u_prev = vec![0.0; n];
        let mut e_prev = 0.0;

        for i in 0..k {
            let u_t = u[i].clone().row_mat();

            let a_u = vec_mul(&u[i])?.col_mat();
            d[i] = (u_t * &a_u)[0][0];

            if i + 1 == k {
                break;
            }

            let v: Matrix = a_u - e_prev * u_prev.col_mat() - d[i] * u[i].clone().col_mat();
            e[i] = v
                .elems
                .par_iter()
                .map(|&v_e| v_e.powi(2))
                .sum::<f64>()
                .sqrt();
            u[i + 1] = ((1.0 / e[i]) * v).elems;

            u_prev = u[i].clone();
            e_prev = e[i];
        }
        let q_t = Matrix::from(k, u.concat());
        let t = SymmetricTridiagonalMatrix::new(d, e);

        Ok((t, q_t))
    }
}
