use crate::matrix::Matrix;
use crate::number::c64;
use lapack::{dgetrs, zgetrs};

impl Matrix {
    /// # Solve equation
    /// with matrix decomposed by getrf
    /// `Ax = b`
    pub fn getrs(&self, ipiv: &[i32], b_t: Matrix) -> Result<Matrix, String> {
        let n = self.get_rows();
        if n != self.get_columns() || n != b_t.columns {
            return Err("dimension mismatch".to_owned());
        }

        let mut info = 0;

        let n = n as i32;
        let mut b_t = b_t;

        unsafe {
            dgetrs(
                'T' as u8,
                n,
                b_t.rows as i32,
                &self.elements,
                n,
                ipiv,
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
    /// with matrix decomposed by getrf
    /// `Ax = b`
    pub fn getrs(&self, ipiv: &[i32], b_t: Matrix<c64>) -> Result<Matrix<c64>, String> {
        let n = self.get_rows();
        if n != self.get_columns() || n != b_t.columns {
            return Err("dimension mismatch".to_owned());
        }

        let mut info = 0;

        let n = n as i32;
        let mut b_t = b_t;

        unsafe {
            zgetrs(
                'T' as u8,
                n,
                b_t.rows as i32,
                &self.elements,
                n,
                ipiv,
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
