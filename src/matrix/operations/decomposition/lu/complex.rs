use crate::matrix::Matrix;
use crate::{lu_decomposed::LUDecomposed, number::c64, types::Type};
use lapack::zgetrf;
use std::intrinsics::transmute;

impl<T> Matrix<T, c64>
where
    T: Type,
{
    /// # LU decomposition
    /// for f64
    pub fn lud(mut self) -> Result<LUDecomposed<T, c64>, String> {
        let mut ipiv = vec![0; self.rows.min(self.columns)];
        let mut info = 0;

        unsafe {
            zgetrf(
                self.rows as i32,
                self.columns as i32,
                transmute::<&mut [c64], &mut [blas::c64]>(&mut self.elements),
                self.rows as i32,
                &mut ipiv,
                &mut info,
            );
        }

        match info {
            0 => Ok(LUDecomposed::new(self.transmute(), ipiv)),
            i => Err(i.to_string()),
        }
    }
}
