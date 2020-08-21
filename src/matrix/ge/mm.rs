use crate::{matrix::Matrix, number::c64};
use blas::dgemm;
use blas::zgemm;
use std::mem::transmute;

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
        let n = lhs.columns as i32;
        let k = rhs.columns as i32;

        let mut slf = self;

        unsafe {
            zgemm(
                'N' as u8,
                'N' as u8,
                k,
                n,
                m,
                transmute::<c64, blas::c64>(alpha),
                transmute::<&[c64], &[blas::c64]>(rhs.elements.as_slice()),
                k,
                transmute::<&[c64], &[blas::c64]>(lhs.elements.as_slice()),
                m,
                transmute::<c64, blas::c64>(beta),
                transmute::<&mut [c64], &mut [blas::c64]>(&mut slf.elements),
                k,
            );
        }

        Ok(slf)
    }
}
