use crate::matrix::Matrix;
use lapack::dgesvd;

impl Matrix {
    /// # Singular Value Decomposition
    ///
    /// https://en.wikipedia.org/wiki/Singular_value_decomposition
    ///
    /// `M = U * Sigma * V^T`
    /// `(u, sigma, vt)`
    pub fn gesvd(&self) -> Result<(Matrix, Matrix, Matrix), String> {
        if self.rows != self.columns {
            return Err("dimension mismatch".to_owned());
        }

        let mut info = 0;
        let mut u = Matrix::new(self.rows, self.rows);
        let mut sigma = Matrix::new(self.rows, self.columns);
        let mut vt = Matrix::new(self.columns, self.columns);
        let lwork = 2 * self.rows;

        unsafe {
            dgesvd(
                'A' as u8,
                'A' as u8,
                self.rows as i32,
                self.columns as i32,
                &mut self.t().elements,
                self.rows as i32,
                &mut sigma.elements,
                &mut u.elements,
                self.rows as i32,
                &mut vt.elements,
                self.columns as i32,
                &mut vec![0.0; lwork],
                lwork as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok((u, sigma, vt)),
            i => Err(i.to_string()),
        }
    }
}
