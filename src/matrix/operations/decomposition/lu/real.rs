use crate::matrix::Matrix;
use crate::{lu_decomposed::LUDecomposed, types::Type};
use lapack::dgetrf;

impl<T> Matrix<T>
where
    T: Type,
{
    /// # LU decomposition
    /// for f64
    pub fn lud(mut self) -> Result<LUDecomposed<T>, String> {
        let mut ipiv = vec![0; self.rows.min(self.columns)];
        let mut info = 0;

        unsafe {
            dgetrf(
                self.rows as i32,
                self.columns as i32,
                &mut self.elements,
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
