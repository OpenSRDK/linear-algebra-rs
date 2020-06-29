use crate::matrix::Matrix;
use crate::types::{PositiveDefinite, Standard};
use lapack::dpftrf;

impl Matrix<PositiveDefinite> {
    pub fn cholesky_decomposition(&self) -> Result<Matrix, i32> {
        if self.rows != self.columns {
            return Err(0);
        }

        let mut a = self.clone().transmute::<Standard>();

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
            i => Err(i),
        }
    }
}
