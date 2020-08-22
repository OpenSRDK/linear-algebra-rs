use crate::matrix::Matrix;
use crate::number::c64;
use lapack::{dpotri, zpotri};

impl Matrix {
    /// # Inverse
    /// with matrix decomposed by potrf
    pub fn potri(self) -> Result<Matrix, String> {
        let n = self.get_rows();
        if n != self.get_columns() {
            return Err("dimension mismatch".to_owned());
        }

        let mut info = 0;
        let mut slf = self;
        let n = n as i32;

        unsafe {
            dpotri('U' as u8, n, &mut slf.elements, n, &mut info);
        }

        match info {
            0 => Ok(slf),
            i => Err(i.to_string()),
        }
    }
}

impl Matrix<c64> {
    /// # Inverse
    /// with matrix decomposed by potrf
    pub fn potri(self) -> Result<Matrix<c64>, String> {
        let n = self.get_rows();
        if n != self.get_columns() {
            return Err("dimension mismatch".to_owned());
        }

        let mut info = 0;
        let mut slf = self;
        let n = n as i32;

        unsafe {
            zpotri('U' as u8, n, &mut slf.elements, n, &mut info);
        }

        match info {
            0 => Ok(slf),
            i => Err(i.to_string()),
        }
    }
}
