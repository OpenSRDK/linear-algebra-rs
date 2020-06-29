use crate::{matrix::Matrix, types::Square};
use lapack::*;

impl Matrix<Square> {
    pub fn solve_linear_equations(self, constants: &Matrix) -> Result<Matrix, i32> {
        if self.rows != constants.rows || constants.columns != 1 {
            return Err(0);
        }

        let mut solution_matrix = constants.clone();
        let mut ipiv = vec![0; self.rows];
        let mut info = 0;

        unsafe {
            dgesv(
                self.rows as i32,
                1,
                &mut self.transpose().elements,
                self.rows as i32,
                &mut ipiv,
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
