use crate::matrix::Matrix;
use crate::number::{c64, Number};
use blas::{dgemm, zgemm};
use rayon::prelude::*;
use std::mem::transmute;
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
    let n = lhs.columns as i32;
    let k = rhs.columns as i32;

    let mut new_matrix = Matrix::zeros(lhs.rows, rhs.columns);

    unsafe {
        dgemm(
            'N' as u8,
            'N' as u8,
            k,
            n,
            m,
            1.0,
            rhs.elements.as_slice(),
            k,
            lhs.elements.as_slice(),
            m,
            0.0,
            &mut new_matrix.elements,
            k,
        );
    }

    new_matrix
}

fn mul_c64(lhs: &Matrix<c64>, rhs: &Matrix<c64>) -> Matrix<c64> {
    if lhs.columns != rhs.rows {
        panic!("dimension mismatch")
    }

    let m = lhs.rows as i32;
    let n = lhs.columns as i32;
    let k = rhs.columns as i32;

    let mut new_matrix = Matrix::<c64>::zeros(lhs.rows, rhs.columns);

    unsafe {
        zgemm(
            'N' as u8,
            'N' as u8,
            k,
            n,
            m,
            blas::c64::new(1.0, 0.0),
            transmute::<&[c64], &[blas::c64]>(&rhs.elements),
            k,
            transmute::<&[c64], &[blas::c64]>(&lhs.elements),
            m,
            blas::c64::new(0.0, 0.0),
            transmute::<&mut [c64], &mut [blas::c64]>(&mut new_matrix.elements),
            k,
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
