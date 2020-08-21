use crate::matrix::Matrix;
use blas::dgemm;

impl Matrix {
    pub fn gemm(self, lhs: &Matrix, rhs: &Matrix, alpha: f64, beta: f64) -> Result<Matrix, String> {
        if self.rows != lhs.rows || self.columns != rhs.columns || lhs.columns != rhs.rows {
            return Err("dimension mismatch".to_owned());
        }

        let m = lhs.rows as i32;
        let n = lhs.columns as i32;
        let k = rhs.columns as i32;

        let mut slf = self;

        unsafe {
            dgemm(
                'N' as u8,
                'N' as u8,
                k,
                n,
                m,
                alpha,
                rhs.elements.as_slice(),
                k,
                lhs.elements.as_slice(),
                m,
                beta,
                &mut slf.elements,
                k,
            );
        }

        Ok(slf)
    }
}
