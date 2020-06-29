use crate::{
    matrix::{operations::identity::identity, Matrix},
    types::PositiveDefinite,
};
use lapack::dposv;

impl Matrix<PositiveDefinite> {
    /// # Inverse
    /// for Positive Definite Matrix
    pub fn inv(&self) -> Result<Matrix<PositiveDefinite>, i32> {
        if self.rows != self.columns {
            return Err(0);
        }

        let mut elements = self.elements.clone();
        let mut solution_matrix = identity(self.rows).transmute();

        let mut info = 0;

        unsafe {
            dposv(
                'U' as u8,
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
