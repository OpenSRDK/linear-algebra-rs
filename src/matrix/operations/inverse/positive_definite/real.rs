use crate::{types::{PositiveDefinite}, matrix::{operations::identity::identity, Matrix}};
use lapack::*;

impl Matrix<PositiveDefinite> {
    pub fn inverse(&self) -> Result<Matrix<PositiveDefinite>, i32> {
        if self.rows != self.columns {
            return Err(0);
        }

        let mut elements = self.elements.clone();
        let mut solution_matrix = identity(self.rows).transmute();

        let mut info = 0;

        unsafe {
            dposv(
                'L' as u8,
                self.rows as i32,
                self.rows as i32,
                &mut elements,
                self.rows as i32,
                &mut solution_matrix.elements,
                self.rows as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok(solution_matrix),
            i => Err(i),
        }
    }
}
