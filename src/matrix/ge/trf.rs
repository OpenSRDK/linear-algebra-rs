use crate::matrix::Matrix;
use crate::number::c64;
use lapack::{dgetrf, dgetri, zgetrf, zgetri};

impl Matrix {
    /// # LU decomposition
    /// for f64
    pub fn getrf(self) -> Result<(Matrix, Vec<i32>), String> {
        let m = self.rows as i32;
        let n = self.columns as i32;
        let mut slf = self;
        let mut ipiv = vec![0; m.min(n) as usize];
        let mut info = 0;

        unsafe {
            dgetrf(n, m, &mut slf.elements, n, &mut ipiv, &mut info);
        }

        match info {
            0 => Ok((slf, ipiv)),
            i => Err(i.to_string()),
        }
    }

    pub fn getri(self, ipiv: &[i32]) -> Result<Matrix, String> {
        let n = self.get_rows();
        if n != self.get_columns() {
            return Err("dimension mismatch".to_owned());
        }

        let mut slf = self;
        let mut work = vec![f64::default(); n];
        let mut info = 0;
        unsafe {
            dgetri(
                n as i32,
                &mut slf.elements,
                n as i32,
                ipiv,
                &mut work,
                n as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok(slf),
            i => Err(i.to_string()),
        }
    }
}

impl Matrix<c64> {
    /// # LU decomposition
    /// for c64
    pub fn getrf(self) -> Result<(Matrix<c64>, Vec<i32>), String> {
        let m = self.rows as i32;
        let n = self.columns as i32;
        let mut slf = self;
        let mut ipiv = vec![0; m.min(n) as usize];
        let mut info = 0;

        unsafe {
            zgetrf(n, m, &mut slf.elements, n, &mut ipiv, &mut info);
        }

        match info {
            0 => Ok((slf, ipiv)),
            i => Err(i.to_string()),
        }
    }

    pub fn getri(self, ipiv: &[i32]) -> Result<Matrix<c64>, String> {
        let n = self.get_rows();
        if n != self.get_columns() {
            return Err("dimension mismatch".to_owned());
        }

        let mut slf = self;
        let mut work = vec![c64::default(); n];
        let mut info = 0;
        unsafe {
            zgetri(
                n as i32,
                &mut slf.elements,
                n as i32,
                ipiv,
                &mut work,
                n as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok(slf),
            i => Err(i.to_string()),
        }
    }
}
