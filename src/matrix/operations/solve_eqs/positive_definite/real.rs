use crate::{
    matrix::Matrix,
    types::{PositiveDefinite, Standard},
};

impl Matrix<PositiveDefinite> {
    /// # Solve equations Conjugate Gradient Method
    /// for Positive definite Matrix
    pub fn solve_eqs_cgm(self, constants: &Matrix) -> Result<Matrix, i32> {
        if self.rows != constants.rows || constants.columns != 1 {
            return Err(0);
        }

        let mut x = Matrix::<Standard>::zeros(constants.rows, 1);
        let mut r = constants.clone();
        let mut p = r.clone();

        loop {
            let r_t = r.t();
            let a_p = &self * &p;
            let alpha = (&r_t * &p)[0][0] / (p.t() * &a_p)[0][0];

            let old_r = r.clone();
            x = x + p.clone() * alpha;
            r = r - a_p.clone() * alpha;

            let max_r = r.elements.iter().fold(0.0 / 0.0, |m, v| v.max(m));
            if max_r < 0.001 {
                break;
            }

            let beta = (r.t() * &r)[0][0] / (&r_t * &old_r)[0][0];
            p = r.clone() + p.clone() * beta;
        }

        Ok(x)
    }
}
