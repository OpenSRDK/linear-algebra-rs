use super::TridiagonalMatrix;
use crate::matrix::MatrixError;
use crate::number::*;
use lapack::dgttrf;
use lapack::zgttrf;

#[derive(Clone, Debug)]
pub struct GTTRF<T = f64>(pub Vec<T>, pub [Vec<T>; 3], pub Vec<i32>)
where
    T: Number;

impl TridiagonalMatrix {
    /// # LU decomposition
    /// for tridiagonal matrix
    pub fn gttrf(self) -> Result<GTTRF, MatrixError> {
        let (mut dl, mut d, mut du) = self.eject();
        let n = d.len();
        let mut du2 = vec![0.0; n.max(2) - 2];
        let mut ipiv = vec![0; n];
        let mut info = 0;

        let n = n as i32;

        unsafe { dgttrf(n, &mut dl, &mut d, &mut du, &mut du2, &mut ipiv, &mut info) }

        if info != 0 {
            return Err(MatrixError::LapackRoutineError {
                routine: "dgttrf".to_owned(),
                info,
            });
        }

        let u = [d, du, du2];

        Ok(GTTRF(dl, u, ipiv))
    }
}

impl TridiagonalMatrix<c64> {
    /// # LU decomposition
    /// for tridiagonal matrix
    pub fn gttrf(self) -> Result<GTTRF<c64>, MatrixError> {
        let (mut dl, mut d, mut du) = self.eject();
        let n = d.len();
        let mut du2 = vec![c64::default(); n.max(2) - 2];
        let mut ipiv = vec![0; n];
        let mut info = 0;

        let n = n as i32;

        unsafe { zgttrf(n, &mut dl, &mut d, &mut du, &mut du2, &mut ipiv, &mut info) }

        if info != 0 {
            return Err(MatrixError::LapackRoutineError {
                routine: "dgttrf".to_owned(),
                info,
            });
        }

        let u = [d, du, du2];

        Ok(GTTRF::<c64>(dl, u, ipiv))
    }
}
