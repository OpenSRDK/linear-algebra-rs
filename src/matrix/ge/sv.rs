use crate::matrix::Matrix;
use lapack::dgesv;

impl Matrix {
    /// # Solve equations
    /// for square matrix
    pub fn gesv(self, constants: Matrix) -> Result<Matrix, String> {
        if self.rows != constants.rows || constants.columns != 1 {
            return Err("dimension mismatch".to_owned());
        }

        let mut solution_matrix = constants;
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
            i => Err(i.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let mut a = Matrix::zeros(2, 2);
        a[0][0] = 1.0;
        a[0][1] = 2.0;
        a[1][0] = 2.0;
        a[1][1] = 1.0;

        let mut b = Matrix::zeros(2, 1);
        b[0][0] = 3.0;
        b[1][0] = 3.0;
        let x = a.gesv(b);

        assert_eq!(x.unwrap()[0][0], 1.0)
    }
}
