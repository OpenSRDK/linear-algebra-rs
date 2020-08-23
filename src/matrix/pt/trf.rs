use crate::{bd::BidiagonalMatrix, matrix::st::SymmetricTridiagonalMatrix};
use lapack::dpttrf;
use rayon::prelude::*;

impl SymmetricTridiagonalMatrix<f64> {
    /// # Cholesky decomposition
    /// for tridiagonal matrix
    pub fn pttrf(self) -> Result<BidiagonalMatrix, String> {
        let (mut d, mut e) = self.get_elements();
        let n = d.len() as i32;
        let mut info = 0;

        unsafe { dpttrf(n, &mut d, &mut e, &mut info) }

        if info != 0 {
            return Err(info.to_string());
        }
        d.par_iter_mut().for_each(|d_e| *d_e = d_e.powf(0.5));

        let bd = BidiagonalMatrix::new(d, e);

        Ok(bd)
    }
}
