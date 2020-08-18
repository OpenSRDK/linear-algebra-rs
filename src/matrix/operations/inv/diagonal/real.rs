use crate::{matrix::Matrix, types::Diagonal};

impl Matrix<Diagonal, f64> {
    /// # Inverse
    /// for Diagonal Matrix
    pub fn inv(mut self) -> Self {
        let n = self.rows;

        for i in 0..n {
            self[i][i] = self[i][i].powi(-1);
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let a = identity::<f64>(2);
        let a_inv = a.clone().inv();

        assert_eq!(a[0][0], a_inv[0][0])
    }
}
