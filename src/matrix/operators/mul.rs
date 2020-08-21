use crate::matrix::Matrix;
use crate::number::{c64, Number};
use blas::{dgemm, zgemm};
use rayon::prelude::*;
use std::ops::Mul;

fn mul<T>(slf: T, rhs: Matrix<T>) -> Matrix<T>
where
    T: Number,
{
    let mut rhs = rhs;
    rhs.elements
        .par_iter_mut()
        .map(|r| {
            *r *= slf;
        })
        .collect::<Vec<_>>();

    rhs
}

fn mul_f64(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    if lhs.columns != rhs.rows {
        panic!("dimension mismatch")
    }

    let m = lhs.rows as i32;
    let k = lhs.columns as i32;
    let n = rhs.columns as i32;

    let mut new_matrix = Matrix::zeros(lhs.rows, rhs.columns);

    unsafe {
        dgemm(
            'T' as u8,
            'T' as u8,
            m,
            n,
            k,
            1.0,
            lhs.elements.as_slice(),
            k,
            rhs.elements.as_slice(),
            n,
            0.0,
            &mut new_matrix.elements,
            m,
        );
    }

    new_matrix
}

fn mul_c64(lhs: &Matrix<c64>, rhs: &Matrix<c64>) -> Matrix<c64> {
    if lhs.columns != rhs.rows {
        panic!("dimension mismatch")
    }

    let m = lhs.rows as i32;
    let k = lhs.columns as i32;
    let n = rhs.columns as i32;

    let mut new_matrix = Matrix::<c64>::zeros(lhs.rows, rhs.columns);

    unsafe {
        zgemm(
            'T' as u8,
            'T' as u8,
            m,
            n,
            k,
            blas::c64::new(1.0, 0.0),
            &lhs.elements,
            k,
            &rhs.elements,
            n,
            blas::c64::new(0.0, 0.0),
            &mut new_matrix.elements,
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
                mul(self, rhs)
            }
        }

        impl Mul<$t> for Matrix<$t> {
            type Output = Matrix<$t>;

            fn mul(self, rhs: $t) -> Self::Output {
                mul(rhs, self)
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
        let a = mat![1.0, 2.0, 3.0];
        let b = mat![
            1.0, 3.0;
            2.0, 4.0;
            3.0, 6.0
        ];
        let c = a * b;

        assert_eq!(c[0][0], 14.0)
    }
}
