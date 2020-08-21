use crate::matrix::Matrix;
use lapack::dgetrf;

impl Matrix {
    /// # LU decomposition
    /// for f64
    pub fn dgetrf(self) -> Result<(Matrix, Vec<i32>), String> {
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
}
