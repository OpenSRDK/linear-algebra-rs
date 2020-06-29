use crate::{matrix::Matrix, types::Square};
use lapack::dgesv;

impl Matrix<Square> {
    /// # Solve equations
    /// for Square Matrix
    pub fn solve_eqs(self, constants: &Matrix) -> Result<Matrix, i32> {
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
                &mut self.t().elements,
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

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let mut a = Matrix::<Square, f64>::zeros(2);
        a[0][0] = 1.0;
        a[0][1] = 2.0;
        a[1][0] = 2.0;
        a[1][1] = 1.0;

        let mut b = Matrix::<Standard, f64>::zeros(2, 1);
        b[0][0] = 3.0;
        b[1][0] = 3.0;
        let x = a.solve_eqs(&b);

        assert_eq!(x.unwrap()[0][0], 1.0)
    }
}
