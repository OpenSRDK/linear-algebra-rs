use crate::matrix::MatrixError;
use crate::{number::c64, Matrix};
use blas::dgemm;
use blas::zgemm;

impl Matrix {
    /// C = self
    /// A = lhs
    /// B = rhs
    /// return alpha*op( A )*op( B ) + beta*C,
    pub fn gemm(
        self,
        lhs: &Matrix,
        rhs: &Matrix,
        alpha: f64,
        beta: f64,
    ) -> Result<Matrix, MatrixError> {
        if self.rows != lhs.rows || self.cols != rhs.cols || lhs.cols != rhs.rows {
            return Err(MatrixError::DimensionMismatch);
        }

        let m = lhs.rows as i32;
        let k = lhs.cols as i32;
        let n = rhs.cols as i32;

        let mut slf = self;

        unsafe {
            dgemm(
                'N' as u8,
                'N' as u8,
                m,
                n,
                k,
                alpha,
                lhs.elems.as_slice(),
                m,
                rhs.elems.as_slice(),
                k,
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
    ) -> Result<Matrix<c64>, MatrixError> {
        if self.rows != lhs.rows || self.cols != rhs.cols || lhs.cols != rhs.rows {
            return Err(MatrixError::DimensionMismatch);
        }

        let m = lhs.rows as i32;
        let k = lhs.cols as i32;
        let n = rhs.cols as i32;

        let mut slf = self;

        unsafe {
            zgemm(
                'N' as u8,
                'N' as u8,
                m,
                n,
                k,
                alpha,
                rhs.elems.as_slice(),
                m,
                lhs.elems.as_slice(),
                k,
                beta,
                &mut slf.elems,
                m,
            );
        }

        Ok(slf)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 2.0;
            3.0, 4.0
        );
        let b = mat!(
            2.0, 1.0;
            4.0, 3.0
        );
        let c = mat!(
            1.0, 3.0;
            5.0, 7.0
        );
        let alpha = 2.0;
        let beta = 3.0;
        let result = c.clone().gemm(&a, &b, alpha, beta).unwrap();
        let result2 = alpha * a.dot(&b) + beta * c;
        assert_eq!(result[(0, 0)], result2[(0, 0)]);
    }
}
