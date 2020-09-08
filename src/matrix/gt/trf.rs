use super::TridiagonalMatrix;
use crate::bd::BidiagonalMatrix;
use crate::matrix::MatrixError;
use lapack::dgttrf;
use std::error::Error;

impl TridiagonalMatrix<f64> {
    /// # LU decomposition
    /// for tridiagonal matrix
    pub fn gttrf(self) -> Result<(BidiagonalMatrix, [Vec<f64>; 3], Vec<i32>), Box<dyn Error>> {
        let (mut dl, mut d, mut du) = self.elems();
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
            }
            .into());
        }

        let l = BidiagonalMatrix::new(vec![1.0; n as usize], dl)?;
        let u = [d, du, du2];

        Ok((l, u, ipiv))
    }
}
