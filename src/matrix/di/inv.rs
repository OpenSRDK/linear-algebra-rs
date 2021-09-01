use crate::{number::c64, DiagonalMatrix};

impl DiagonalMatrix<f64> {
    /// # Inverse
    /// for diagonal matrix
    pub fn diinv(self) -> Self {
        self.dipowi(-1)
    }
}

impl DiagonalMatrix<c64> {
    /// # Inverse
    /// for diagonal matrix
    pub fn diinv(self) -> Self {
        self.dipowi(-1)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = DiagonalMatrix::<f64>::identity(2);
        let a_inv = a.clone().diinv();

        assert_eq!(a[0], a_inv[0]);

        let a = DiagonalMatrix::<c64>::identity(2);
        let a_inv = a.clone().diinv();

        assert_eq!(a[0], a_inv[0]);
    }
}
