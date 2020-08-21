use crate::{matrix::Matrix, number::c64};
use blas::dgemm;
use blas::zgemm;

impl Matrix {
    pub fn gemm(self, lhs: &Matrix, rhs: &Matrix, alpha: f64, beta: f64) -> Result<Matrix, String> {
        if self.rows != lhs.rows || self.columns != rhs.columns || lhs.columns != rhs.rows {
            return Err("dimension mismatch".to_owned());
        }

        let m = lhs.rows as i32;
        let k = lhs.columns as i32;
        let n = rhs.columns as i32;

        let mut slf = self;

        unsafe {
            dgemm(
                'T' as u8,
                'T' as u8,
                m,
                n,
                k,
                alpha,
                lhs.elements.as_slice(),
                k,
                rhs.elements.as_slice(),
                n,
                beta,
                &mut slf.elements,
                m,
            );
        }

        Ok(slf)
    }
}

impl Matrix<c64> {
    pub fn gemm(
        self,
        lhs: &Matrix<c64>,
        rhs: &Matrix<c64>,
        alpha: c64,
        beta: c64,
    ) -> Result<Matrix<c64>, String> {
        if self.rows != lhs.rows || self.columns != rhs.columns || lhs.columns != rhs.rows {
            return Err("dimension mismatch".to_owned());
        }

        let m = lhs.rows as i32;
        let k = lhs.columns as i32;
        let n = rhs.columns as i32;

        let mut slf = self;

        unsafe {
            zgemm(
                'T' as u8,
                'T' as u8,
                m,
                n,
                k,
                alpha,
                rhs.elements.as_slice(),
                k,
                lhs.elements.as_slice(),
                n,
                beta,
                &mut slf.elements,
                m,
            );
        }

        Ok(slf)
    }
}
