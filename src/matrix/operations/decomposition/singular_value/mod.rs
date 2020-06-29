use crate::{types::{Diagonal, PositiveSemiDefinite, Square}, matrix::{Matrix, operations::identity::identity}};
use lapack::*;

impl Matrix<PositiveSemiDefinite> {
    pub fn singular_value_decomposition(
        &self,
    ) -> Result<(Matrix<Square>, Matrix<Diagonal>, Matrix<Square>), i32> {
        if self.rows != self.columns {
            return Err(0);
        }

        let mut info = 0;
        let mut  u = Matrix::<Square>::zeros(self.rows);
        let mut s = identity(self.rows);
        let lwork = 2 * self.rows;

        unsafe {
            dgesvd(
                'N' as u8,
                'A' as u8,
                self.rows as i32,
                self.columns as i32,
                &mut self.transpose().elements,
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
              let u_t = u.transpose();
              Ok((u, s, u_t))
            },
            i => Err(i),
        }
    }
}
