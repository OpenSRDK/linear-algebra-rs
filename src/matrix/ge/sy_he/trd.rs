use crate::{
    matrix::{Matrix, MatrixError, Vector},
    st_ht::SymmetricTridiagonalMatrix,
};
use lapack::dsytrd;
use std::error::Error;

pub struct SYTRD(pub Matrix, pub Vec<f64>, pub SymmetricTridiagonalMatrix);

impl Matrix {
    /// # Tridiagonalize
    /// for symmetric matrix
    pub fn sytrd(self) -> Result<SYTRD, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }
        let mut mat = self;
        let n = mat.rows as i32;
        let mut d = vec![0.0; mat.rows];
        let mut e = vec![0.0; mat.rows.max(1) - 1];
        let mut tau = vec![0.0; mat.rows.max(1) - 1];
        let lwork = 2 * mat.rows;
        let mut work = vec![0.0; lwork];
        let mut info = 0;

        unsafe {
            dsytrd(
                'L' as u8,
                n,
                &mut mat.elems,
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
                });
            }
        }

        let t = SymmetricTridiagonalMatrix::new(d, e)?;

        Ok(SYTRD(mat, tau, t))
    }

    /// # Lanczos algorithm
    /// for symmetric matrix
    /// only k iteration
    pub fn sytrd_k(
        n: usize,
        k: usize,
        vec_mul: &dyn Fn(Vec<f64>) -> Result<Vec<f64>, Box<dyn Error + Send + Sync>>,
        probe: Option<&[f64]>,
    ) -> Result<(Matrix, SymmetricTridiagonalMatrix), MatrixError> {
        let k = k.min(n);

        let mut d = vec![0.0; k];
        let mut e = vec![0.0; k.max(1) - 1];

        let mut v = vec![vec![0.0; n]; k];

        if 0 < k {
            match probe {
                Some(vec) => {
                    if vec.len() != n {
                        return Err(MatrixError::DimensionMismatch);
                    }
                    let norm = vec.iter().map(|wi| wi.powi(2)).sum::<f64>().sqrt();
                    v[0] = vec.iter().map(|vi| vi / norm).collect();
                }
                None => {
                    v[0][0] = 1.0;
                }
            }

            let a_v = match vec_mul(v[0].clone()) {
                Ok(v) => Ok(v),
                Err(e) => Err(MatrixError::Others(e)),
            }?
            .col_mat();
            let v_mat = v[0].clone().col_mat();

            d[0] = (a_v.t() * &v_mat)[0][0];
            let mut w_prev = a_v - d[0] * v_mat;

            for i in 1..k {
                e[i - 1] = w_prev
                    .slice()
                    .iter()
                    .map(|wi| wi.powi(2))
                    .sum::<f64>()
                    .sqrt();

                v[i].clone_from_slice((w_prev * (1.0 / e[i - 1])).slice());

                let a_v = match vec_mul(v[i].clone()) {
                    Ok(v) => Ok(v),
                    Err(e) => Err(MatrixError::Others(e)),
                }?
                .col_mat();
                let v_mat = v[i].clone().col_mat();

                d[i] = (a_v.t() * &v_mat)[0][0];
                w_prev = a_v - d[i] * v_mat - e[i - 1] * v[i - 1].clone().col_mat();
            }
        }

        let q = Matrix::from(n, v.concat()).unwrap();
        let t = SymmetricTridiagonalMatrix::new(d, e)?;

        Ok((q, t))
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat![
            1.0, 3.0, 6.0, 12.0;
            2.0, 4.0, 8.0, 16.0;
            3.0, 6.0, 12.0, 24.0;
            4.0, 8.0, 16.0, 30.0
        ];
        let (q, t) =
            Matrix::sytrd_k(4, 3, &|v: Vec<f64>| Ok((&a * v.col_mat()).vec()), None).unwrap();

        let aback = &q * &t.mat() * &q.t();

        println!("{:#?}", aback);
        println!("{:#?}", &q * &q.t());
    }
}
