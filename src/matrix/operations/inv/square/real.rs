use crate::{
    matrix::{operations::identity::identity, Matrix},
    types::{Square, Standard},
};
use lapack::dgesv;

impl Matrix<Square> {
    /// # Inverse
    /// for Square Matrix
    pub fn inv(&self) -> Result<Matrix, String> {
        let mut elements = self.elements.clone();
        let mut solution_matrix = identity(self.rows).transmute::<Standard>();

        let mut ipiv = vec![0; self.rows];
        let mut info = 0;

        unsafe {
            dgesv(
                self.rows as i32,
                self.rows as i32,
                &mut elements,
                self.rows as i32,
                &mut ipiv,
                &mut solution_matrix.elements,
                self.rows as i32,
                &mut info,
            );
        }

        match info {
            0 => Ok(solution_matrix),
            i => Err(i.to_string()),
        }
    }
}
