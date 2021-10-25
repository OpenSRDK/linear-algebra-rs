use super::SymmetricTridiagonalMatrix;
use crate::{matrix::ge::Matrix, matrix::*};
use lapack::dstev;

impl SymmetricTridiagonalMatrix {
    /// # Eigen decomposition
    /// return (lambda, pt)
    pub fn stev(self) -> Result<(Vec<f64>, Matrix), MatrixError> {
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
                z.elems_mut(),
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
            }),
        }
    }
}
