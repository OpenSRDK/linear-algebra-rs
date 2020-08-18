use crate::{
    matrix::{operations::identity::identity, Matrix},
    types::{Diagonal, PositiveSemiDefinite, Square},
};
use lapack::dgesvd;

impl Matrix<PositiveSemiDefinite> {
    /// # Singular Value Decomposition
    ///
    /// https://en.wikipedia.org/wiki/Singular_value_decomposition
    ///
    /// `M = U * Sigma * V^T`
    pub fn svd(&self) -> Result<(Matrix<Square>, Matrix<Diagonal>, Matrix<Square>), String> {
        if self.rows != self.columns {
            return Err("dimension mismatch".to_owned());
        }

        let mut info = 0;
        let mut u = Matrix::<Square>::zeros(self.rows);
        let mut s = identity(self.rows);
        let lwork = 2 * self.rows;

        unsafe {
            dgesvd(
                'N' as u8,
                'A' as u8,
                self.rows as i32,
                self.columns as i32,
                &mut self.t().elements,
                self.rows as i32,
                &mut s.elements,
                &mut vec![],
                self.rows as i32,
                &mut u.elements,
                self.columns as i32,
                &mut vec![0.0; lwork],
                lwork as i32,
                &mut info,
            );
        }

        match info {
            0 => {
                let u_t = u.t();
                Ok((u, s, u_t))
            }
            i => Err(i.to_string()),
        }
    }
}
