use crate::{matrix::Matrix, number::c64};

impl Matrix<f64> {
    /// # Inverse
    /// for diagonal matrix
    pub fn diinv(self) -> Self {
        self.dipowi(-1)
    }
}

impl Matrix<c64> {
    /// # Inverse
    /// for diagonal matrix
    pub fn diinv(self) -> Self {
        self.dipowi(-1)
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    #[test]
    fn it_works() {
        let a = Matrix::<f64>::identity(2);
        let a_inv = a.clone().diinv();

        assert_eq!(a[0][0], a_inv[0][0]);

        let a = Matrix::<c64>::identity(2);
        let a_inv = a.clone().diinv();

        assert_eq!(a[0][0], a_inv[0][0]);
    }
}
