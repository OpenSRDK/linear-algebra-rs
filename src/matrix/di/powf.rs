use crate::{number::c64, DiagonalMatrix};
use rayon::prelude::*;

impl DiagonalMatrix<f64> {
    /// # Pow integer
    /// for diagonal matrix
    pub fn powf(self, exp: f64) -> Self {
        let mut slf = self;
        slf.d.par_iter_mut().for_each(|di| *di = di.powf(exp));

        slf
    }
}

impl DiagonalMatrix<c64> {
    /// # Pow integer
    /// for diagonal matrix
    pub fn powf(self, exp: f64) -> Self {
        let mut slf = self;
        slf.d.par_iter_mut().for_each(|di| *di = di.powf(exp));

        slf
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = DiagonalMatrix::<f64>::identity(2);
        let a_inv = a.clone().powf(-1.0);

        assert_eq!(a[0], a_inv[0]);

        let a = DiagonalMatrix::<c64>::identity(2);
        let a_inv = a.clone().powf(-1.0);

        assert_eq!(a[0], a_inv[0]);
    }
}
