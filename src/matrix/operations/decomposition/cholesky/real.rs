use crate::matrix::Matrix;
use crate::types::{LowerTriangle, PositiveDefinite};
use lapack::dpftrf;

impl Matrix<PositiveDefinite> {
    /// # Cholesky decomposition
    /// for f64
    ///
    /// https://en.wikipedia.org/wiki/Cholesky_decomposition
    ///
    /// `A = L * L^T`
    pub fn cholesky(&self) -> Result<Matrix<LowerTriangle>, String> {
        if self.rows != self.columns {
            return Err("dimension mismatch".to_owned());
        }

        let mut a = self.clone().transmute();

        let mut info = 0;

        unsafe {
            dpftrf(
                'T' as u8,
                'L' as u8,
                self.rows as i32,
                &mut a.elements,
                &mut info,
            );
        }

        match info {
            0 => Ok(a),
            i => Err(i.to_string()),
        }
    }
}
