use crate::matrix::Matrix;
use crate::number::c64;
use lapack::{dgetri, zgetri};

impl Matrix {
    /// # Inverse
    /// with matrix decomposed by getrf
    pub fn getri(self, ipiv: &[i32]) -> Result<Matrix, String> {
        let n = self.get_rows();
        if n != self.get_columns() {
            return Err("dimension mismatch".to_owned());
        }

        let mut work = vec![f64::default(); n];
        let mut info = 0;

        let mut slf = self;
        let n = n as i32;

        unsafe {
            dgetri(n, &mut slf.elements, n, ipiv, &mut work, n, &mut info);
        }

        match info {
            0 => Ok(slf),
            i => Err(i.to_string()),
        }
    }
}

impl Matrix<c64> {
    /// # Inverse
    /// with matrix decomposed by getrf
    pub fn getri(self, ipiv: &[i32]) -> Result<Matrix<c64>, String> {
        let n = self.get_rows();
        if n != self.get_columns() {
            return Err("dimension mismatch".to_owned());
        }

        let mut work = vec![c64::default(); n];
        let mut info = 0;

        let mut slf = self;
        let n = n as i32;

        unsafe {
            zgetri(n, &mut slf.elements, n, ipiv, &mut work, n, &mut info);
        }

        match info {
            0 => Ok(slf),
            i => Err(i.to_string()),
        }
    }
}
