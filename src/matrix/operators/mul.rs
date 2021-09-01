use crate::matrix::Matrix;
use crate::number::{c64, Number};
use blas::{dgemm, zgemm};
use rayon::prelude::*;
use std::ops::Mul;

fn mul_scalar<T>(slf: T, rhs: Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    let mut rhs = rhs;
    rhs.elems
        .par_iter_mut()
        .map(|r| {
            *r *= slf;
        })
        .collect::<Vec<_>>();

    rhs
}

fn mul_f64(lhs: &Matrix, rhs: &Matrix) -> Matrix {
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

fn mul_c64(lhs: &Matrix<c64>, rhs: &Matrix<c64>) -> Matrix<c64> {
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

macro_rules! impl_mul_scalar {
    {$t: ty} => {
        impl Mul<Matrix<$t>> for $t {
            type Output = Matrix<$t>;

            fn mul(self, rhs: Matrix<$t>) -> Self::Output {
                mul_scalar(self, rhs)
            }
        }

        impl Mul<$t> for Matrix<$t> {
            type Output = Matrix<$t>;

            fn mul(self, rhs: $t) -> Self::Output {
                mul_scalar(rhs, self)
            }
        }
    };
}

impl_mul_scalar! {f64}
impl_mul_scalar! {c64}

macro_rules! impl_mul {
  {$t: ty, $e: expr} => {
      impl Mul<Matrix<$t>> for Matrix<$t> {
          type Output = Matrix<$t>;

          fn mul(self, rhs: Matrix<$t>) -> Self::Output {
              $e(&self, &rhs)
          }
      }

      impl Mul<&Matrix<$t>> for Matrix<$t> {
          type Output = Matrix<$t>;

          fn mul(self, rhs: &Matrix<$t>) -> Self::Output {
              $e(&self, rhs)
          }
      }

      impl Mul<Matrix<$t>> for &Matrix<$t> {
          type Output = Matrix<$t>;

          fn mul(self, rhs: Matrix<$t>) -> Self::Output {
              $e(self, &rhs)
          }
      }

      impl Mul<&Matrix<$t>> for &Matrix<$t> {
          type Output = Matrix<$t>;

          fn mul(self, rhs: &Matrix<$t>) -> Self::Output {
              $e(self, rhs)
          }
      }
  };
}

impl_mul! {f64, mul_f64}
impl_mul! {c64, mul_c64}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn it_works() {
        let a = mat!(
            1.0, 2.0;
            3.0, 4.0
        ) * mat!(
            5.0, 6.0;
            7.0, 8.0
        );
        assert_eq!(a[(0, 0)], 19.0);
    }
}
