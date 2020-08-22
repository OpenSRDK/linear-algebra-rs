use crate::matrix::Matrix;
use crate::number::c64;
use lapack::{dpotrs, zpotrs};

impl Matrix {
    /// # Solve equation
    /// with matrix decomposed by potrf
    /// `Ax = b`
    pub fn potrs(&self, b_t: Matrix) -> Result<Matrix, String> {
        let n = self.get_rows();
        if n != self.get_columns() || n != b_t.columns {
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
                &self.elements,
                n,
                &mut b_t.elements,
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
        let n = self.get_rows();
        if n != self.get_columns() || n != b_t.columns {
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
                &self.elements,
                n,
                &mut b_t.elements,
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
