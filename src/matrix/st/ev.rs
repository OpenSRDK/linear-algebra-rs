use super::SymmetricTridiagonalMatrix;
use crate::matrix::*;
use lapack::dstev;
use std::error::Error;

impl SymmetricTridiagonalMatrix<f64> {
    /// # Eigen decomposition
    /// return (lambda, pt)
    pub fn stev(self) -> Result<(Vec<f64>, Matrix), Box<dyn Error>> {
        let (mut d, mut e) = self.eject();
        let n = d.len();
        let mut z = Matrix::new(n, n);
        let mut work = vec![0.0; 1.max(2 * (n - 2))];
        let mut info = 0;

        let n = n as i32;

        unsafe {
            dstev(
                'V' as u8,
                n,
                &mut d,
                &mut e,
                &mut z.elems,
                n,
                &mut work,
                &mut info,
            )
        }

        match info {
            0 => Ok((d, z)),
            _ => Err(MatrixError::LapackRoutineError {
                routine: "dstev".to_owned(),
                info,
            }
            .into()),
        }
    }
}
