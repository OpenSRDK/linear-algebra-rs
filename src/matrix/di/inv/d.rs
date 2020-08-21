use crate::matrix::Matrix;

impl Matrix<f64> {
    /// # Inverse
    /// for diagonal matrix
    pub fn diinv(self) -> Self {
        let n = self.rows;
        let mut slf = self;

        for i in 0..n {
            slf[i][i] = slf[i][i].powi(-1);
        }

        slf
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let a = Matrix::<f64>::identity(2);
        let a_inv = a.clone().diinv();

        assert_eq!(a[0][0], a_inv[0][0])
    }
}
