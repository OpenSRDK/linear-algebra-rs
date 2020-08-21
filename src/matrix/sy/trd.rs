use crate::matrix::Matrix;
use lapack::{dorgtr, dsytrd};

impl Matrix {
    /// # Lanczos algorithm
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
}
