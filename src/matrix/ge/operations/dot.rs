use crate::c64;
use crate::Matrix;
use blas::dgemm;
use blas::zgemm;

impl Matrix<f64> {
    pub fn dot(&self, rhs: &Self) -> Self {
        let lhs = self;
        if lhs.cols != rhs.rows {
            panic!("Dimension mismatch.")
        }

        let m = lhs.rows as i32;
        let k = lhs.cols as i32;
        let n = rhs.cols as i32;

        let mut new_matrix = Matrix::new(lhs.rows, rhs.cols);

        unsafe {
            dgemm(
                'N' as u8,
                'N' as u8,
                m,
                n,
                k,
                1.0,
                lhs.elems.as_slice(),
                m,
                rhs.elems.as_slice(),
                k,
                0.0,
                &mut new_matrix.elems,
                m,
            );
        }

        new_matrix
    }
}

impl Matrix<c64> {
    pub fn dot(&self, rhs: &Self) -> Self {
        let lhs = self;
        if lhs.cols != rhs.rows {
            panic!("Dimension mismatch.")
        }

        let m = lhs.rows as i32;
        let k = lhs.cols as i32;
        let n = rhs.cols as i32;

        let mut new_matrix = Matrix::<c64>::new(lhs.rows, rhs.cols);

        unsafe {
            zgemm(
                'N' as u8,
                'N' as u8,
                m,
                n,
                k,
                blas::c64::new(1.0, 0.0),
                &lhs.elems,
                m,
                &rhs.elems,
                k,
                blas::c64::new(0.0, 0.0),
                &mut new_matrix.elems,
                m,
            );
        }

        new_matrix
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
        )
        .dot(&mat!(
            5.0, 6.0;
            7.0, 8.0
        ));
        assert_eq!(a[(0, 0)], 19.0);
    }
}
