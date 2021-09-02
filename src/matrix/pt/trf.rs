use crate::matrix::MatrixError;
use crate::DiagonalMatrix;
use crate::{bd::BidiagonalMatrix, matrix::st::SymmetricTridiagonalMatrix};
use lapack::dpttrf;

impl SymmetricTridiagonalMatrix<f64> {
    /// # Cholesky decomposition
    /// for tridiagonal matrix
    /// `T = L * D * L^T`
    pub fn pttrf(self) -> Result<(BidiagonalMatrix, DiagonalMatrix<f64>), MatrixError> {
        let (mut d, mut e) = self.eject();
        let n = d.len() as i32;
        let mut info = 0;

        unsafe { dpttrf(n, &mut d, &mut e, &mut info) }

        if info != 0 {
            return Err(MatrixError::LapackRoutineError {
                routine: "dpttrf".to_owned(),
                info,
            });
        }

        let bd = BidiagonalMatrix::new(vec![1.0; n as usize], e)?;
        let d = DiagonalMatrix::new(d);

        Ok((bd, d))
    }
}
