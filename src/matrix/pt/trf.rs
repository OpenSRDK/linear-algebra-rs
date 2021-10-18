use crate::matrix::*;
use crate::DiagonalMatrix;
use crate::{bd::BidiagonalMatrix, matrix::st_ht::SymmetricTridiagonalMatrix};
use lapack::dpttrf;
use lapack::zpttrf;

pub struct PTTRF<T = f64>(pub BidiagonalMatrix<T>, pub DiagonalMatrix)
where
    T: Number;

impl SymmetricTridiagonalMatrix {
    /// # Cholesky decomposition
    /// for tridiagonal matrix
    /// `T = L * D * L^T`
    pub fn pttrf(self) -> Result<PTTRF, MatrixError> {
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

        Ok(PTTRF(bd, d))
    }
}

impl SymmetricTridiagonalMatrix<c64> {
    /// # Cholesky decomposition
    /// for tridiagonal matrix
    /// `T = L * D * L^T`
    pub fn pttrf(self) -> Result<PTTRF<c64>, MatrixError> {
        let (d, mut e) = self.eject();
        let mut d = d.into_par_iter().map(|di| di.re).collect::<Vec<_>>();
        let n = d.len() as i32;
        let mut info = 0;

        unsafe { zpttrf(n, &mut d, &mut e, &mut info) }

        if info != 0 {
            return Err(MatrixError::LapackRoutineError {
                routine: "zpttrf".to_owned(),
                info,
            });
        }

        let bd = BidiagonalMatrix::<c64>::new(vec![c64::one(); n as usize], e)?;
        let d = DiagonalMatrix::new(d);

        Ok(PTTRF::<c64>(bd, d))
    }
}
