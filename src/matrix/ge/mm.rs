use crate::matrix::MatrixError;
use crate::{matrix::Matrix, number::c64};
use blas::dgemm;
use blas::zgemm;
use std::error::Error;

impl Matrix {
    pub fn gemm(
        self,
        lhs: &Matrix,
        rhs: &Matrix,
        alpha: f64,
        beta: f64,
    ) -> Result<Matrix, Box<dyn Error>> {
        if self.rows != lhs.rows || self.cols != rhs.cols || lhs.cols != rhs.rows {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let m = lhs.rows as i32;
        let k = lhs.cols as i32;
        let n = rhs.cols as i32;

        let mut slf = self;

        unsafe {
            dgemm(
                'T' as u8,
                'T' as u8,
                m,
                n,
                k,
                alpha,
                lhs.elems.as_slice(),
                k,
                rhs.elems.as_slice(),
                n,
                beta,
                &mut slf.elems,
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
    ) -> Result<Matrix<c64>, Box<dyn Error>> {
        if self.rows != lhs.rows || self.cols != rhs.cols || lhs.cols != rhs.rows {
            return Err(MatrixError::DimensionMismatch.into());
        }

        let m = lhs.rows as i32;
        let k = lhs.cols as i32;
        let n = rhs.cols as i32;

        let mut slf = self;

        unsafe {
            zgemm(
                'T' as u8,
                'T' as u8,
                m,
                n,
                k,
                alpha,
                rhs.elems.as_slice(),
                k,
                lhs.elems.as_slice(),
                n,
                beta,
                &mut slf.elems,
                m,
            );
        }

        Ok(slf)
    }
}
