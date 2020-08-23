use crate::matrix::Matrix;
use crate::number::c64;
use lapack::{dpotrs, zpotrs};

impl Matrix {
    /// # Solve equation
    /// with matrix decomposed by potrf
    /// `Ax = b`
    pub fn potrs(&self, b_t: Matrix) -> Result<Matrix, String> {
        let n = self.rows();
        if n != self.cols() || n != b_t.cols {
            return Err("dimension mismatch".to_owned());
        }

        let mut info = 0;

        let n = n as i32;
        let mut b_t = b_t;

        unsafe {
            dpotrs(
                'U' as u8,
                n,
                b_t.rows as i32,
                &self.elems,
                n,
                &mut b_t.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b_t),
            i => Err(i.to_string()),
        }
    }
}

impl Matrix<c64> {
    /// # Solve equation
    /// with matrix decomposed by potrf
    /// `Ax = b`
    pub fn potrs(&self, b_t: Matrix<c64>) -> Result<Matrix<c64>, String> {
        let n = self.rows();
        if n != self.cols() || n != b_t.cols {
            return Err("dimension mismatch".to_owned());
        }

        let mut info = 0;

        let n = n as i32;
        let mut b_t = b_t;

        unsafe {
            zpotrs(
                'U' as u8,
                n,
                b_t.rows as i32,
                &self.elems,
                n,
                &mut b_t.elems,
                n,
                &mut info,
            );
        }

        match info {
            0 => Ok(b_t),
            i => Err(i.to_string()),
        }
    }
}
